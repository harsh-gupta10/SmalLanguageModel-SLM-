import os
import torch
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import IterableDataset, DataLoader
import json
from tqdm.auto import tqdm
import torch.nn.functional as F
import gc # Import garbage collector
import random # For sampling

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
TOKENIZER_MODEL_PATH = 'model/tokenizer/multilingual_spm.model'
TEACHER_MODEL_PATH = 'model/teacher/' # Path to your downloaded Qwen3-0.6B
OUTPUT_DATA_DIR = 'data/processed/'
MAX_SEQUENCE_LENGTH = 64 # Keep this reduced (consider 256 if needed)

# --- Sampling Parameters ---
SAMPLE_PROBABILITY = 0.001 # <--- *** Adjust this for your target dataset size ***

# --- Batching Parameters ---
BATCH_SIZE_TEACHER_INFERENCE = 8 # *** Increased for better GPU utilization *** (Try 16 if VRAM allows, but carefully)
MAX_RAM_FOR_LOGITS_GB = 6 # *** CRITICAL: Max GB of RAM to use for accumulating logits before saving ***


# --- Generator for Raw Text Data with Sampling ---
def raw_text_generator_with_sampling(raw_data_dir, input_file_names, sample_probability):
    """Yields sentences from all input files, one by one, with random sampling,
       prefixing them with language tokens."""
    print(f"Streaming raw text data from {raw_data_dir} with sample probability {sample_probability}...")

    # Map file names to their corresponding language tokens
    lang_map = {
        'lang_english.txt': '<en>',
        'lang_hindi.txt': '<hi>',
        'lang_sanskrit.txt': '<sa>'
    }

    for f_name in input_file_names:
        f_path = os.path.join(raw_data_dir, f_name)
        lang_token = lang_map.get(f_name, '') # Get the correct language token
        if not lang_token:
            print(f"Warning: No language token defined for file {f_name}. Skipping.")
            continue

        if not os.path.exists(f_path):
            print(f"Error: Input file not found at {f_path}. Skipping.")
            continue
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                if random.random() < sample_probability:
                    line = line.strip()
                    if line:
                        yield f"{lang_token} {line}"
                        
# --- Iterable Dataset for Teacher Inference ---
class InferenceIterableDataset(IterableDataset):
    def __init__(self, raw_text_gen, tokenizer_processor, max_seq_len):
        self.raw_text_gen = raw_text_gen
        self.sp = tokenizer_processor
        self.max_seq_len = max_seq_len
        self.estimated_total_sampled = 0 # To track how many we've yielded

    def __iter__(self):
        for text in self.raw_text_gen:
            token_ids = self.sp.encode(text)
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            self.estimated_total_sampled += 1
            yield torch.tensor(token_ids, dtype=torch.long)
            
def prepare_distillation_data():
    print("Loading custom tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL_PATH)

    input_file_names = ['lang_english.txt', 'lang_hindi.txt', 'lang_sanskrit.txt']
    
    # Estimate total number of sentences for tqdm progress bar (before sampling)
    total_raw_sentences = 0
    for f_name in input_file_names:
        f_path = os.path.join(RAW_DATA_DIR, f_name)
        if os.path.exists(f_path):
            with open(f_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_raw_sentences += 1
    estimated_sampled_sentences = int(total_raw_sentences * SAMPLE_PROBABILITY)
    print(f"Estimated total raw sentences: {total_raw_sentences}")
    print(f"Estimated number of sampled sentences for distillation: {estimated_sampled_sentences}")


    # Create the generator for raw texts with sampling
    text_gen = raw_text_generator_with_sampling(RAW_DATA_DIR, input_file_names, SAMPLE_PROBABILITY)
    
    # Create the iterable dataset
    inference_dataset = InferenceIterableDataset(text_gen, sp, MAX_SEQUENCE_LENGTH)

    # Custom collate function for DataLoader
    def teacher_collate_fn(batch):
        max_len = max(len(item) for item in batch)
        padded_input_ids = []
        attention_mask_batch = []
        for item in batch:
            padding_len = max_len - len(item)
            padded_input_ids.append(F.pad(item, (0, padding_len), value=sp.pad_id()))
            attention_mask_batch.append(torch.cat([torch.ones(len(item), dtype=torch.long), 
                                                    torch.zeros(padding_len, dtype=torch.long)], dim=0))
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_mask_batch)
        }

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE_TEACHER_INFERENCE,
        shuffle=False,
        collate_fn=teacher_collate_fn,
        num_workers=0 # Keep at 0 for IterableDataset unless you handle worker init well
    )

    print("Loading Qwen3-0.6B teacher model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Teacher model will be loaded on {device}.")
        
        if device.type == 'cuda':
            if torch.cuda.is_bf16_supported():
                teacher_dtype = torch.bfloat16
                print("Using bfloat16 for teacher model weights.")
            else:
                teacher_dtype = torch.float16
                print("Using float16 for teacher model weights.")
        else: # On CPU, use float32 for stability and performance (Colab's CPU is slow for half-precision)
            teacher_dtype = torch.float32 
            print("Using float32 for teacher model weights (on CPU).")

        teacher_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_PATH,
            torch_dtype=teacher_dtype
        )
    except Exception as e:
        print(f"Error loading teacher model from {TEACHER_MODEL_PATH}: {e}")
        print("Please ensure the Qwen3-0.6B model files (config.json, model.safetensors etc.) are in 'model/teacher/' or update TEACHER_MODEL_PATH.")
        return

    teacher_model.eval()
    teacher_model.to(device)
    print(f"Teacher model loaded successfully. Model parameters: {teacher_model.num_parameters() / 1e6:.2f} M")

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    teacher_logits_current_group = []
    original_tokenized_ids_current_group = []
    chunk_file_idx = 0

    # Calculate optimal BATCHES_PER_SAVE based on MAX_RAM_FOR_LOGITS_GB
    logit_size_per_sequence_float32_MB = MAX_SEQUENCE_LENGTH * teacher_model.config.vocab_size * 4 / (1024**2) # 4 bytes for float32
    max_sequences_in_ram = int(MAX_RAM_FOR_LOGITS_GB * 1024 / logit_size_per_sequence_float32_MB)
    
    # Ensure at least one batch is saved at a time.
    # batches_per_save = max(1, max_sequences_in_ram // BATCH_SIZE_TEACHER_INFERENCE)
    batches_per_save = 10

    
    print(f"Calculated logit size per sequence (float32): {logit_size_per_sequence_float32_MB:.2f} MB")
    print(f"Max sequences to accumulate in RAM: {max_sequences_in_ram}")
    print(f"Calculated BATCHES_PER_SAVE: {batches_per_save}")


    print("Generating teacher logits...")
    tqdm_loader = tqdm(inference_dataloader, total=estimated_sampled_sentences // BATCH_SIZE_TEACHER_INFERENCE if estimated_sampled_sentences > 0 else None, desc="Generating Teacher Logits")

    for batch_idx, batch in enumerate(tqdm_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

            if batch_idx == 0:
                 print(f"Size of single logit tensor (on {device}, bfloat16 for logit, float32 for storage): {logits.element_size() * logits.nelement() / (1024**2):.2f} MB (GPU output)")
                 print(f"Estimated peak RAM for accumulating logits: {max_sequences_in_ram * logit_size_per_sequence_float32_MB / 1024:.2f} GB")

            for k in range(input_ids.shape[0]): 
                original_len = torch.sum(attention_mask[k]).item()
                
                original_tokenized_ids_current_group.append(input_ids[k, :original_len].cpu().numpy())
                teacher_logits_current_group.append(logits[k, :original_len, :].to(torch.float16).cpu().numpy())

                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        
        if (batch_idx + 1) % batches_per_save == 0:
            chunk_idx_str = str(chunk_file_idx).zfill(6) 
            
            torch.save(original_tokenized_ids_current_group, os.path.join(OUTPUT_DATA_DIR, f'distillation_input_ids_chunk_{chunk_idx_str}.pt'))
            torch.save(teacher_logits_current_group, os.path.join(OUTPUT_DATA_DIR, f'distillation_teacher_logits_chunk_{chunk_idx_str}.pt'))
            
            # print(f"\nSaved chunk {chunk_idx_str} ({len(original_tokenized_ids_current_group)} items). Clearing memory...")
            
            teacher_logits_current_group = [] 
            original_tokenized_ids_current_group = []
            chunk_file_idx += 1

            if device.type == 'cuda':
                torch.cuda.empty_cache() 
            gc.collect() 

    # Save any remaining items in the last partial group
    if len(original_tokenized_ids_current_group) > 0:
        chunk_idx_str = str(chunk_file_idx).zfill(6) 
        torch.save(original_tokenized_ids_current_group, os.path.join(OUTPUT_DATA_DIR, f'distillation_input_ids_chunk_{chunk_idx_str}.pt'))
        torch.save(teacher_logits_current_group, os.path.join(OUTPUT_DATA_DIR, f'distillation_teacher_logits_chunk_{chunk_idx_str}.pt'))
        print(f"\nSaved final chunk {chunk_idx_str} ({len(original_tokenized_ids_current_group)} items).")

    print("\nAll distillation data chunks prepared and saved!")

if __name__ == '__main__':
    prepare_distillation_data()