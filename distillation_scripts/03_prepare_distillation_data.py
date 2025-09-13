import os
import torch
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
import torch.nn.functional as F
import gc # Import garbage collector

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
TOKENIZER_MODEL_PATH = 'model/tokenizer/multilingual_spm.model'
TEACHER_MODEL_PATH = 'model/teacher/' # Path to your downloaded Qwen3-0.6B
OUTPUT_DATA_DIR = 'data/processed/'
MAX_SEQUENCE_LENGTH = 1024
BATCH_SIZE_TEACHER_INFERENCE = 1 # *** CRITICAL: Reduced for lower VRAM/RAM usage during teacher inference ***
CHUNK_SIZE = 1000 # Number of processed items to save to disk at once to manage RAM

# --- Custom Dataset Class for Teacher Inference ---
class InferenceTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def prepare_distillation_data():
    print("Loading custom tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL_PATH)

    print(f"Loading raw text data from {RAW_DATA_DIR}...")
    input_files = [
        os.path.join(RAW_DATA_DIR, 'lang_english.txt'),
        os.path.join(RAW_DATA_DIR, 'lang_hindi.txt'),
        os.path.join(RAW_DATA_DIR, 'lang_sanskrit.txt')
    ]

    all_raw_texts = [] # Accumulate all raw texts first to get total count
    for f_path in input_files:
        if not os.path.exists(f_path):
            print(f"Error: Input file not found at {f_path}. Please ensure it exists.")
            return

        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_raw_texts.append(line)
    
    if not all_raw_texts:
        print("No raw text data found to process. Exiting.")
        return

    print(f"Loaded {len(all_raw_texts)} raw sentences.")
    
    print("Loading Qwen3-0.6B teacher model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Teacher model will be loaded on {device}.")
        
        # Determine dtype based on device capability
        if device.type == 'cuda' and torch.cuda.is_bf16_supported():
            teacher_dtype = torch.bfloat16
            print("Using bfloat16 for teacher model weights.")
        elif device.type == 'cuda':
            teacher_dtype = torch.float16
            print("Using float16 for teacher model weights.")
        else:
            teacher_dtype = torch.float32 # CPU usually uses float32
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
    print(f"Teacher model loaded successfully.")

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Process raw texts in chunks
    for i in tqdm(range(0, len(all_raw_texts), CHUNK_SIZE), desc="Processing data chunks"):
        chunk_raw_texts = all_raw_texts[i : i + CHUNK_SIZE]
        
        print(f"\nProcessing chunk {i // CHUNK_SIZE + 1} (items {i} to {min(i + CHUNK_SIZE, len(all_raw_texts) - 1)})...")

        # Tokenize current chunk
        tokenized_inputs_ids_chunk = []
        for text in chunk_raw_texts:
            token_ids = sp.encode(text)
            if len(token_ids) > MAX_SEQUENCE_LENGTH:
                token_ids = token_ids[:MAX_SEQUENCE_LENGTH]
            tokenized_inputs_ids_chunk.append(torch.tensor(token_ids, dtype=torch.long))

        # Create DataLoader for current chunk
        inference_dataset_chunk = InferenceTextDataset(tokenized_inputs_ids_chunk)
        
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

        inference_dataloader_chunk = DataLoader(
            inference_dataset_chunk,
            batch_size=BATCH_SIZE_TEACHER_INFERENCE,
            shuffle=False,
            collate_fn=teacher_collate_fn
        )

        teacher_logits_current_chunk = []
        original_tokenized_ids_current_chunk = []

        for batch_idx, batch in enumerate(tqdm(inference_dataloader_chunk, desc=f"Generating logits for chunk {i // CHUNK_SIZE + 1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

                for k in range(input_ids.shape[0]):
                    original_len = torch.sum(attention_mask[k]).item()
                    
                    original_tokenized_ids_current_chunk.append(input_ids[k, :original_len].cpu().numpy())
                    teacher_logits_current_chunk.append(logits[k, :original_len, :].cpu().numpy())

        # Save current chunk to disk
        chunk_idx_str = str(i // CHUNK_SIZE).zfill(5) # e.g., 00000, 00001
        torch.save(original_tokenized_ids_current_chunk, os.path.join(OUTPUT_DATA_DIR, f'distillation_input_ids_chunk_{chunk_idx_str}.pt'))
        torch.save(teacher_logits_current_chunk, os.path.join(OUTPUT_DATA_DIR, f'distillation_teacher_logits_chunk_{chunk_idx_str}.pt'))
        
        print(f"Chunk {chunk_idx_str} saved. Clearing memory...")
        del tokenized_inputs_ids_chunk, inference_dataset_chunk, inference_dataloader_chunk
        del teacher_logits_current_chunk, original_tokenized_ids_current_chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache() # Clear GPU cache
        gc.collect() # Force Python garbage collection

    print("\nAll distillation data chunks prepared and saved!")

if __name__ == '__main__':
    prepare_distillation_data()