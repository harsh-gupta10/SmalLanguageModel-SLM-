import os
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler
from tqdm.auto import tqdm
import gc # Import garbage collector for explicit memory management
import glob # For finding chunk files
import math # For math.ceil in _estimate_total_items

# --- Configuration ---
STUDENT_CONFIG_PATH = 'model/student/config.json'
PROCESSED_DATA_DIR = 'data/processed/'
OUTPUT_MODEL_DIR = 'model/student_trained/'

# Training Hyperparameters
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2 # Student training batch size (adjust based on GPU memory)
GRADIENT_ACCUMULATION_STEPS = 4 # Accumulate gradients over this many batches

# Distillation Hyperparameters
DISTILLATION_ALPHA = 0.5 # Weight for distillation loss vs. CE loss (0.0 to 1.0)
TEMPERATURE = 2.0      # Temperature for softening teacher logits

# --- IMPORTANT: Teacher's original vocab size ---
# This is crucial for correctly processing teacher_logits
# from Qwen3-0.6B, which had vocab_size = 151936
TEACHER_VOCAB_SIZE = 151936

def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    """
    Calculates the combined distillation loss (KL divergence + Cross-entropy).

    Args:
        student_logits (torch.Tensor): Logits from the student model.
                                       Shape: (batch_size, seq_len, STUDENT_VOCAB_SIZE)
        teacher_logits (torch.Tensor): Logits from the teacher model.
                                       Shape: (batch_size, seq_len, TEACHER_VOCAB_SIZE)
        labels (torch.Tensor): Ground truth token IDs for hard cross-entropy loss.
                               Shape: (batch_size, seq_len)
        alpha (float): Weighting factor for distillation loss.
        temperature (float): Temperature for softening probability distributions.

    Returns:
        torch.Tensor: The calculated combined loss.
    """
    # Shift logits and labels for next token prediction, as is standard for CausalLM
    shifted_student_logits = student_logits[:, :-1, :].contiguous()
    shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    # --- Hard labels (Cross-entropy loss) ---
    # The student's logits are directly used for its own vocabulary
    loss_hard = F.cross_entropy(shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
                                shifted_labels.view(-1),
                                ignore_index=-100) # -100 is common for padding or ignored tokens

    # --- Soft targets (KL Divergence loss) ---
    student_vocab_size = shifted_student_logits.size(-1)
    
    # If student's vocab is smaller, pad its logits to match teacher's vocab size
    # This is necessary for KL_Div to compare distributions over the same dimension.
    if student_vocab_size < TEACHER_VOCAB_SIZE:
        # Create a tensor of negative infinity for padding
        # -inf ensures that log_softmax assigns virtually zero probability to padded tokens.
        padding = -torch.inf * torch.ones(
            shifted_student_logits.size(0),
            shifted_student_logits.size(1),
            TEACHER_VOCAB_SIZE - student_vocab_size,
            device=shifted_student_logits.device,
            dtype=shifted_student_logits.dtype
        )
        # Concatenate student logits with padding to match teacher's vocab size
        shifted_student_logits_padded = torch.cat([shifted_student_logits, padding], dim=-1)
    else: # Should not happen in this specific distillation setup
        shifted_student_logits_padded = shifted_student_logits

    # Compute log-probabilities for the student (over the padded vocab space)
    student_log_probs = F.log_softmax(shifted_student_logits_padded / temperature, dim=-1)
    # Compute probabilities for the teacher (over its full vocab space)
    teacher_probs = F.softmax(shifted_teacher_logits / temperature, dim=-1)

    # KL_Div expects log-probabilities for the first input and probabilities for the second.
    loss_soft = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature**2)

    # Combined loss
    return alpha * loss_soft + (1.0 - alpha) * loss_hard


# --- Custom Iterable Dataset for loading distillation chunks ---
class DistillationChunkIterableDataset(IterableDataset):
    """
    An IterableDataset that streams distillation data from chunk files,
    avoiding loading the entire dataset into memory.
    """
    def __init__(self, processed_data_dir, student_config_pad_token_id):
        self.processed_data_dir = processed_data_dir
        # Find all chunk files for input IDs and teacher logits
        self.input_ids_files = sorted(glob.glob(os.path.join(processed_data_dir, 'distillation_input_ids_chunk_*.pt')))
        self.teacher_logits_files = sorted(glob.glob(os.path.join(processed_data_dir, 'distillation_teacher_logits_chunk_*.pt')))
        
        if len(self.input_ids_files) == 0:
            raise FileNotFoundError(f"No distillation chunk files found in {processed_data_dir}")
        if len(self.input_ids_files) != len(self.teacher_logits_files):
            raise ValueError("Mismatch in number of input_ids and teacher_logits chunk files. "
                             "Ensure each input_ids chunk has a corresponding teacher_logits chunk.")
        
        self.pad_token_id = student_config_pad_token_id
        self.total_items = self._estimate_total_items()
        print(f"Total estimated items for training: {self.total_items}")

    def _estimate_total_items(self):
        """
        Estimates the total number of individual examples across all chunk files.
        This can be slow if there are many tiny chunk files.
        """
        print("Estimating total items for progress bar (may take a moment for many chunks)...")
        total = 0
        for f_path in tqdm(self.input_ids_files, desc="Estimating total items"):
            try:
                # --- FIX HERE: Add weights_only=False ---
                chunk_data = torch.load(f_path, weights_only=False)
                total += len(chunk_data)
            except Exception as e:
                print(f"Warning: Could not estimate length for {f_path}: {e}")
        return total

    def __iter__(self):
        """
        Iterates through chunk files, loads them, and yields individual examples.
        Memory is managed by deleting chunks after use.
        """
        # Get worker info for multiprocessing (if num_workers > 0)
        worker_info = torch.utils.data.get_worker_info()
        start_file_idx = 0
        end_file_idx = len(self.input_ids_files)

        if worker_info is not None:
            # Divide files among workers
            per_worker_files = int(math.ceil((end_file_idx - start_file_idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_file_idx = start_file_idx + worker_id * per_worker_files
            end_file_idx = min(start_file_idx + per_worker_files, end_file_idx)
            print(f"Worker {worker_id} processing files from index {start_file_idx} to {end_file_idx-1}")


        # Iterate through the assigned chunk files
        for i in range(start_file_idx, end_file_idx):
            input_ids_chunk_file = self.input_ids_files[i]
            teacher_logits_chunk_file = self.teacher_logits_files[i]
            
            # Load chunk data
            try:
                # --- FIX HERE: Add weights_only=False ---
                chunk_input_ids = torch.load(input_ids_chunk_file, weights_only=False)
                chunk_teacher_logits = torch.load(teacher_logits_chunk_file, weights_only=False)
            except Exception as e:
                print(f"Error loading chunk {input_ids_chunk_file} or {teacher_logits_chunk_file}: {e}")
                continue # Skip this corrupted or unreadable chunk

            if len(chunk_input_ids) != len(chunk_teacher_logits):
                print(f"Warning: Mismatch in item count for chunk {i} ({input_ids_chunk_file}). Skipping.")
                del chunk_input_ids, chunk_teacher_logits
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            # Yield individual items from the loaded chunk
            for input_ids, teacher_logits in zip(chunk_input_ids, chunk_teacher_logits):
                # Convert NumPy arrays back to PyTorch tensors here
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'teacher_logits': torch.tensor(teacher_logits, dtype=torch.float32)
                }
            
            # Clear chunk from memory after processing all its items
            del chunk_input_ids, chunk_teacher_logits
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def train_student_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading student model configuration...")
    student_config = AutoConfig.from_pretrained(STUDENT_CONFIG_PATH)

    print("Initializing student model from configuration...")
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to(device)
    student_model.train() # Set model to training mode

    print(f"Student model has {student_model.num_parameters() / 1e6:.2f} M parameters.")

    # Instantiate the custom IterableDataset to stream data
    train_dataset = DistillationChunkIterableDataset(PROCESSED_DATA_DIR, student_config.pad_token_id)

    # Custom collate function to handle padding for student training batches
    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        teacher_logits = [item['teacher_logits'] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_teacher_logits = []
        attention_mask = []

        for i_ids, t_logits_tensor in zip(input_ids, teacher_logits):
            padding_len = max_len - len(i_ids)
            
            # Pad input_ids with the student's pad_token_id
            padded_input_ids.append(F.pad(i_ids, (0, padding_len), value=student_config.pad_token_id))
            
            # Pad teacher_logits on the sequence length dimension
            # The t_logits_tensor already has shape (original_len, TEACHER_VOCAB_SIZE)
            if padding_len > 0:
                # Create padding tensor of zeros for the logits
                padding_logits_tensor = torch.zeros(
                    (padding_len, TEACHER_VOCAB_SIZE),
                    dtype=torch.float32,
                    device=t_logits_tensor.device if t_logits_tensor.is_cuda else torch.device('cpu') # Ensure device consistency
                )
                padded_teacher_logits.append(torch.cat([t_logits_tensor, padding_logits_tensor], dim=0))
            else:
                padded_teacher_logits.append(t_logits_tensor)

            # Create attention mask
            attention_mask.append(torch.cat([torch.ones(len(i_ids), dtype=torch.long), 
                                              torch.zeros(padding_len, dtype=torch.long)], dim=0))

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_mask),
            'teacher_logits': torch.stack(padded_teacher_logits)
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # IterableDataset typically handles shuffling via file order or within its __iter__
        collate_fn=custom_collate_fn,
        num_workers=0 # Set to 0 for IterableDataset unless advanced multiprocessing is implemented
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    # Approximate total training steps for the scheduler and progress bar
    # This might be slightly off if total_items isn't perfectly divisible, but it's good enough.
    num_training_steps = NUM_EPOCHS * (train_dataset.total_items // BATCH_SIZE)
    if train_dataset.total_items % BATCH_SIZE != 0:
        num_training_steps += NUM_EPOCHS # Add steps for potential last partial epoch in each loop


    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps), desc="Student Training")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        # For IterableDataset, you need to re-create the iterator for each epoch
        # if you want to re-iterate over the full dataset.
        # This will re-read all chunk files from disk.
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_logits = batch['teacher_logits'].to(device) # Move teacher logits to GPU

            # Student model forward pass
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits # Shape: (batch_size, seq_len, STUDENT_VOCAB_SIZE)

            # Calculate distillation loss
            loss = distillation_loss(student_logits=student_logits,
                                    teacher_logits=teacher_logits,
                                    labels=input_ids,
                                    alpha=DISTILLATION_ALPHA,
                                    temperature=TEMPERATURE)

            # Backpropagation
            loss.backward()

            # Gradient accumulation
            if (progress_bar.n + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("\nTraining finished! Saving student model...")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    student_model.save_pretrained(OUTPUT_MODEL_DIR)
    student_config.save_pretrained(OUTPUT_MODEL_DIR) # Save the student config along with the model weights

if __name__ == '__main__':
    train_student_model()