import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import json

from peft import LoraConfig, get_peft_model, TaskType

# --- Configuration ---
# 1. Path to the model you want to continue fine-tuning
PRETRAINED_MODEL_PATH = 'model/checkpoints/pretrained/checkpoint-24000'
TOKENIZER_PATH = PRETRAINED_MODEL_PATH

# 2. Input JSON data files for the new task (English and Hindi)
DATA_FILES = ['finetuning/data/02_english.json', 'finetuning/data/02_hindi.json']

# 3. Output directory for the new LoRA adapters
OUTPUT_MODEL_DIR = 'model/checkpoints/finetuned/task_2/'

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20
BATCH_SIZE = 24 # Adjust if you face memory issues

# --- PEFT & LoRA Configuration (Generally can be kept the same) ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# --- New Prompt Template ---
# This template is generic and works for both English and Hindi
# by directly using the instruction from the data file.
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# --- Dataset and Collate Function ---

class NounVerbSwapDataset(Dataset):
    """
    A PyTorch Dataset to handle the noun-verb swap JSON data.
    It reads JSON files containing a list of dictionaries.
    """
    def __init__(self, data_files, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        print("Loading and processing data for the new task...")
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # json.load reads the entire JSON array from the file
                    file_data = json.load(f)
                    self.data.extend(file_data)
                    print(f"Loaded {len(file_data)} samples from {os.path.basename(file_path)}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}. The file might be empty or malformed.")
            except FileNotFoundError:
                print(f"Error: Data file not found at {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get data from the item, providing default empty strings
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # Format the full prompt with the response included
        full_prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output_text
        ) + self.tokenizer.eos_token
        
        tokenized_full = self.tokenizer(full_prompt, truncation=True, max_length=512, padding=False)

        # Format the prompt without the response to calculate its length
        prompt_without_response = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output="" # The response part is empty
        )
        tokenized_prompt_only = self.tokenizer(prompt_without_response, truncation=True, max_length=512, padding=False)
        
        prompt_len = len(tokenized_prompt_only['input_ids'])

        # Create labels, masking out the prompt part
        labels = list(tokenized_full['input_ids'])
        for i in range(prompt_len):
            labels[i] = -100 # -100 is the ignore_index for the loss function

        return {
            "input_ids": torch.tensor(tokenized_full['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_full['attention_mask'], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def custom_collate_fn(batch, pad_token_id):
    """Pads sequences to the max length in a batch."""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels
    }

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def finetune_with_lora():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the base model and tokenizer
    print(f"Loading base model from: {PRETRAINED_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # --- Prepare model for LoRA training ---
    print("\nInitializing LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    
    print("\nModel architecture with LoRA adapters:")
    print_trainable_parameters(model)
    
    # --- Dataset and DataLoader ---
    train_dataset = NounVerbSwapDataset(DATA_FILES, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: custom_collate_fn(b, tokenizer.pad_token_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # --- Training Loop ---
    progress_bar = tqdm(range(num_training_steps), desc="Fine-Tuning with LoRA")
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average Loss for Epoch {epoch+1}: {avg_loss:.4f}")

    # --- Saving the LoRA Adapters ---
    print(f"\nFine-tuning finished! Saving LoRA adapters to {OUTPUT_MODEL_DIR}")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("Adapters and tokenizer saved successfully.")


if __name__ == '__main__':
    finetune_with_lora()