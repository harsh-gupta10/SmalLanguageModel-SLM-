import os
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler
from tqdm.auto import tqdm
import gc
import glob
import math

# --- Configuration ---
STUDENT_CONFIG_PATH = 'model/student/config.json'
PROCESSED_DATA_DIR = 'data/processed/'
OUTPUT_MODEL_DIR = 'model/student_trained/'
CHECKPOINT_DIR = 'model/student_trained/checkpoints/'

# Training Hyperparameters
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# Distillation Hyperparameters
DISTILLATION_ALPHA = 0.5
TEMPERATURE = 2.0
TEACHER_VOCAB_SIZE = 151936

# Checkpoint settings
CHECKPOINT_FREQ = 500  # Save every 500 steps


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    shifted_student_logits = student_logits[:, :-1, :].contiguous()
    shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    loss_hard = F.cross_entropy(
        shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100
    )

    student_vocab_size = shifted_student_logits.size(-1)
    if student_vocab_size < TEACHER_VOCAB_SIZE:
        padding = -torch.inf * torch.ones(
            shifted_student_logits.size(0),
            shifted_student_logits.size(1),
            TEACHER_VOCAB_SIZE - student_vocab_size,
            device=shifted_student_logits.device,
            dtype=shifted_student_logits.dtype
        )
        shifted_student_logits_padded = torch.cat([shifted_student_logits, padding], dim=-1)
    else:
        shifted_student_logits_padded = shifted_student_logits

    student_log_probs = F.log_softmax(shifted_student_logits_padded / temperature, dim=-1)
    teacher_probs = F.softmax(shifted_teacher_logits / temperature, dim=-1)

    loss_soft = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return alpha * loss_soft + (1.0 - alpha) * loss_hard


class DistillationChunkIterableDataset(IterableDataset):
    def __init__(self, processed_data_dir, student_config_pad_token_id):
        self.processed_data_dir = processed_data_dir
        self.input_ids_files = sorted(glob.glob(os.path.join(processed_data_dir, 'distillation_input_ids_chunk_*.pt')))
        self.teacher_logits_files = sorted(glob.glob(os.path.join(processed_data_dir, 'distillation_teacher_logits_chunk_*.pt')))
        
        if len(self.input_ids_files) == 0:
            raise FileNotFoundError(f"No distillation chunk files found in {processed_data_dir}")
        if len(self.input_ids_files) != len(self.teacher_logits_files):
            raise ValueError("Mismatch in number of input_ids and teacher_logits chunk files.")

        self.pad_token_id = student_config_pad_token_id
        self.total_items = self._estimate_total_items()
        print(f"Total estimated items for training: {self.total_items}")

    def _estimate_total_items(self):
        total = 0
        for f_path in tqdm(self.input_ids_files, desc="Estimating total items"):
            try:
                chunk_data = torch.load(f_path, weights_only=False)
                total += len(chunk_data)
            except Exception as e:
                print(f"Warning: Could not estimate length for {f_path}: {e}")
        return total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start_file_idx = 0
        end_file_idx = len(self.input_ids_files)

        if worker_info is not None:
            per_worker_files = int(math.ceil((end_file_idx - start_file_idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_file_idx = start_file_idx + worker_id * per_worker_files
            end_file_idx = min(start_file_idx + per_worker_files, end_file_idx)
            print(f"Worker {worker_id} processing files {start_file_idx}–{end_file_idx-1}")

        for i in range(start_file_idx, end_file_idx):
            input_ids_chunk_file = self.input_ids_files[i]
            teacher_logits_chunk_file = self.teacher_logits_files[i]

            try:
                chunk_input_ids = torch.load(input_ids_chunk_file, weights_only=False)
                chunk_teacher_logits = torch.load(teacher_logits_chunk_file, weights_only=False)
            except Exception as e:
                print(f"Error loading chunk {input_ids_chunk_file} or {teacher_logits_chunk_file}: {e}")
                continue

            if len(chunk_input_ids) != len(chunk_teacher_logits):
                print(f"Warning: Mismatch in item count for chunk {i}. Skipping.")
                del chunk_input_ids, chunk_teacher_logits
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            for input_ids, teacher_logits in zip(chunk_input_ids, chunk_teacher_logits):
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'teacher_logits': torch.tensor(teacher_logits, dtype=torch.float32)
                }

            del chunk_input_ids, chunk_teacher_logits
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def save_checkpoint(model, config, optimizer, scheduler, step, epoch):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-epoch{epoch}-step{step}")
    os.makedirs(ckpt_path, exist_ok=True)

    model.save_pretrained(ckpt_path)
    config.save_pretrained(ckpt_path)

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }, os.path.join(ckpt_path, "training_state.pt"))

    print(f"Saved checkpoint at {ckpt_path}")


def load_checkpoint_if_available(model, optimizer, scheduler, device):
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-epoch*-step*")))
    if not checkpoints:
        print("No checkpoints found, starting fresh.")
        return model, optimizer, scheduler, 0, 0

    latest_ckpt = checkpoints[-1]
    print(f"Resuming from checkpoint: {latest_ckpt}")

    model = AutoModelForCausalLM.from_pretrained(latest_ckpt).to(device)
    training_state = torch.load(os.path.join(latest_ckpt, "training_state.pt"), map_location=device)

    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    step = training_state["step"]
    epoch = training_state["epoch"]

    return model, optimizer, scheduler, step, epoch


def train_student_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    student_config = AutoConfig.from_pretrained(STUDENT_CONFIG_PATH)
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to(device)
    student_model.train()

    print(f"Student model has {student_model.num_parameters() / 1e6:.2f} M parameters.")

    train_dataset = DistillationChunkIterableDataset(PROCESSED_DATA_DIR, student_config.pad_token_id)

    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        teacher_logits = [item['teacher_logits'] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids, padded_teacher_logits, attention_mask = [], [], []

        for i_ids, t_logits_tensor in zip(input_ids, teacher_logits):
            padding_len = max_len - len(i_ids)
            padded_input_ids.append(F.pad(i_ids, (0, padding_len), value=student_config.pad_token_id))
            if padding_len > 0:
                padding_logits_tensor = torch.zeros(
                    (padding_len, TEACHER_VOCAB_SIZE),
                    dtype=torch.float32,
                    device=t_logits_tensor.device if t_logits_tensor.is_cuda else torch.device('cpu')
                )
                padded_teacher_logits.append(torch.cat([t_logits_tensor, padding_logits_tensor], dim=0))
            else:
                padded_teacher_logits.append(t_logits_tensor)
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
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    num_training_steps = NUM_EPOCHS * (train_dataset.total_items // BATCH_SIZE)
    if train_dataset.total_items % BATCH_SIZE != 0:
        num_training_steps += NUM_EPOCHS

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    student_model, optimizer, lr_scheduler, global_step, start_epoch = load_checkpoint_if_available(
        student_model, optimizer, lr_scheduler, device
    )

    progress_bar = tqdm(range(global_step, num_training_steps), desc="Student Training")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            loss = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=input_ids,
                alpha=DISTILLATION_ALPHA,
                temperature=TEMPERATURE
            )

            loss.backward()

            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (global_step + 1) % CHECKPOINT_FREQ == 0:
                save_checkpoint(student_model, student_config, optimizer, lr_scheduler, global_step + 1, epoch + 1)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("\nTraining finished! Saving final student model...")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    student_model.save_pretrained(OUTPUT_MODEL_DIR)
    student_config.save_pretrained(OUTPUT_MODEL_DIR)


if __name__ == '__main__':
    train_student_model()
