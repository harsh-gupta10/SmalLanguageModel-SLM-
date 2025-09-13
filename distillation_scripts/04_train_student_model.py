import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import Dataset # Hugging Face datasets library is convenient

# --- Configuration ---
STUDENT_CONFIG_PATH = 'model/student/config.json'
PROCESSED_DATA_DIR = 'data/processed/'
OUTPUT_MODEL_DIR = 'model/student_trained/'
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
DISTILLATION_ALPHA = 0.5 # Weight for distillation loss vs. CE loss
TEMPERATURE = 2.0      # Temperature for softening teacher logits

def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    # Student logits and teacher logits should be for the same tokens (sequence length, vocab_size)
    # Labels are the hard targets for cross-entropy

    # Ensure teacher_logits and student_logits are aligned with labels
    # For CausalLM, logits are for the *next* token prediction.
    # So, we typically compare student_logits[:, :-1, :] with teacher_logits[:, :-1, :]
    # and labels[:, 1:] for the hard CE loss.

    # Shift logits and labels for next token prediction
    shifted_student_logits = student_logits[:, :-1, :].contiguous()
    shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    # Hard labels (Cross-entropy loss)
    loss_hard = F.cross_entropy(shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
                                shifted_labels.view(-1),
                                ignore_index=-100) # -100 is common for padding or ignored tokens

    # Soft targets (KL Divergence loss)
    # Apply temperature to soften the teacher probabilities
    student_probs = F.log_softmax(shifted_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shifted_teacher_logits / temperature, dim=-1)

    loss_soft = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature**2)

    # Combined loss
    return alpha * loss_soft + (1.0 - alpha) * loss_hard


def train_student_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading student model configuration...")
    student_config = AutoConfig.from_pretrained(STUDENT_CONFIG_PATH)

    print("Initializing student model from configuration...")
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to(device)
    student_model.train()

    print(f"Student model has {student_model.num_parameters() / 1e6:.2f} M parameters.")

    print("Loading processed distillation data...")
    tokenized_inputs = torch.load(os.path.join(PROCESSED_DATA_DIR, 'tokenized_inputs.pt'))
    teacher_logits_list = torch.load(os.path.join(PROCESSED_DATA_DIR, 'teacher_logits.pt'))

    # Convert to Hugging Face Dataset format for easier handling
    # Ensure proper padding if input sequences are not of equal length
    # This part needs careful implementation for batching and padding
    # For simplicity, we'll assume a batching function or DataLoader handles padding
    
    # Example: Create a simple dummy dataset and Dataloader.
    # In a real scenario, you'd pad `tokenized_inputs` and `teacher_logits`
    # to the max sequence length within a batch.
    
    # Let's create a dummy dataset for illustration.
    # Your `DistillationDataset` from 03_prepare_distillation_data.py would be used here.
    
    # Assuming `tokenized_inputs` and `teacher_logits_list` are lists of tensors/arrays
    # and you need a custom collate_fn for padding.
    
    # For now, let's simplify and assume all inputs are already padded or
    # we'll pad dynamically in a collate_fn.
    
    # This part requires robust data loading and batching with padding.
    # Example using a custom collate_fn:
    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        teacher_logits = [item['teacher_logits'] for item in batch]
        
        # Determine max length in batch
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad input_ids and teacher_logits
        padded_input_ids = []
        padded_teacher_logits = []
        attention_mask = []

        # Ensure vocab_size matches student config (student_config.vocab_size)
        # The shape of padded_teacher_logits should be (batch_size, max_len, vocab_size)
        # The shape of input_ids should be (batch_size, max_len)

        for i_ids, t_logits in zip(input_ids, teacher_logits):
            padding_len = max_len - len(i_ids)
            
            # Pad input_ids with pad_token_id
            padded_input_ids.append(
                F.pad(i_ids, (0, padding_len), value=student_config.pad_token_id)
            )
            # Pad teacher_logits with zeros or a neutral value if desired,
            # but typically, KL_Div will ignore padding positions if mask is used.
            # Or, more robustly, ensure teacher_logits are also padded correctly during generation.
            
            # A robust way is to pad teacher_logits with a tensor of shape (padding_len, vocab_size)
            # filled with a neutral value (e.g., negative infinity for log-probs, or zeros for probs)
            # or simply zeros, and use an attention_mask to ignore.
            
            # For simplicity, assuming t_logits were generated for padded sequences of fixed MAX_SEQUENCE_LENGTH
            # or they are already pre-padded.
            # If t_logits corresponds to original tokenized_inputs without padding,
            # then you need to pad t_logits to (max_len, vocab_size).
            
            # Let's assume teacher_logits were generated for sequences up to MAX_SEQUENCE_LENGTH
            # and therefore need to be padded to `max_len` here.
            
            vocab_size = student_config.vocab_size # Using student's vocab size
            
            # Pad teacher_logits: shape (seq_len, vocab_size)
            # Use a tensor of zeros for padding logits
            padding_logits = torch.zeros((padding_len, vocab_size), dtype=torch.float32)
            padded_teacher_logits.append(torch.cat([t_logits, padding_logits], dim=0))


            attention_mask.append(torch.cat([torch.ones(len(i_ids), dtype=torch.long), 
                                              torch.zeros(padding_len, dtype=torch.long)], dim=0))

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_mask),
            'teacher_logits': torch.stack(padded_teacher_logits)
        }

    # Create an in-memory dataset or use your `DistillationDataset`
    # For demonstration, let's assume `tokenized_inputs` and `teacher_logits_list`
    # are lists of PyTorch tensors, each representing a single sequence.
    
    # This step requires you to re-structure `tokenized_inputs` and `teacher_logits_list`
    # into a list of dictionaries as expected by your `DistillationDataset` or `datasets.Dataset`.
    
    # Example:
    train_data = []
    for i in range(len(tokenized_inputs)):
        train_data.append({
            'input_ids': torch.tensor(tokenized_inputs[i], dtype=torch.long),
            'teacher_logits': torch.tensor(teacher_logits_list[i], dtype=torch.float32)
        })
    
    train_dataset = Dataset.from_list(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)


    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)

            # Student model forward pass
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits # Shape: (batch_size, seq_len, vocab_size)

            # Calculate distillation loss
            # The labels for the hard CE loss are simply the input_ids shifted
            loss = distillation_loss(student_logits=student_logits,
                                    teacher_logits=teacher_logits,
                                    labels=input_ids, # Use input_ids as hard labels for next token prediction
                                    alpha=DISTILLATION_ALPHA,
                                    temperature=TEMPERATURE)

            loss.backward()

            # Gradient accumulation
            if (progress_bar.n + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training finished! Saving student model...")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    student_model.save_pretrained(OUTPUT_MODEL_DIR)
    student_config.save_pretrained(OUTPUT_MODEL_DIR) # Save the student config with the model

if __name__ == '__main__':
    train_student_model()