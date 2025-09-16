import os
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---

# 1. Define Project and Model Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "model" / "tokenizer"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PRETRAINED_CHECKPOINTS_DIR = PROJECT_ROOT / "model" / "checkpoints" / "pretrained"
LOG_DIR = PRETRAINED_CHECKPOINTS_DIR / "logs"

# 2. Model Configuration (Target: ~125M parameters using Qwen3 Architecture)
MODEL_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_key_value_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 2048,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "use_cache": False,
    "torch_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
}

# 3. Training Hyperparameters
TRAINING_ARGS = {
    "output_dir": str(PRETRAINED_CHECKPOINTS_DIR),
    "logging_dir": str(LOG_DIR),
    "report_to": "tensorboard",
    "max_steps": 40000,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": True,
    "eval_steps": 2000,
    "save_steps": 2000,
    "save_total_limit": 10,
    "logging_steps": 100,
    # MODIFICATION 1: Lower the learning rate to break through the plateau
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "lr_scheduler_type": "cosine",
    "fp16": MODEL_CONFIG["torch_dtype"] == "float16",
    "bf16": MODEL_CONFIG["torch_dtype"] == "bfloat16",
    "optim": "adamw_torch",
    "load_best_model_at_end": False,
    # MODIFICATION 2: Set to False to avoid deleting checkpoints on resume
    "overwrite_output_dir": False,
    "do_train": True,
    "do_eval": True,
}

# 4. Language Tag Configuration
LANGUAGE_TAG_MAP = {
    "english": "<en>",
    "hindi": "<hi>",
    "sanskrit": "<sa>"
}

def main():
    """
    Main function to orchestrate the pretraining of the language model.
    """
    logger.info("Starting Qwen3 pretraining script...")

    PRETRAINED_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Custom Tokenizer ---
    logger.info(f"Loading tokenizer from directory: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")

    # --- 2. Configure and Initialize Model ---
    # This section remains the same. The Trainer will handle loading the checkpoint weights.
    logger.info("Configuring and initializing Qwen3 model from scratch.")
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        **MODEL_CONFIG,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    logger.info(f"Model initialized with Qwen3 architecture and ~{params:.2f}M trainable parameters.")

    # --- 3. Load and Prepare Datasets from Raw Files (Streaming Mode) ---
    raw_files = list(RAW_DATA_DIR.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No raw .txt files found in {RAW_DATA_DIR}.")

    logger.info("Loading raw text data in STREAMING mode and adding language tags...")

    def get_lang_tag_from_filename(filepath: Path):
        file_stem = filepath.stem.lower()
        for lang_keyword, tag in LANGUAGE_TAG_MAP.items():
            if lang_keyword in file_stem:
                return tag
        return None

    all_streaming_datasets = []
    for f_path in raw_files:
        lang_tag = get_lang_tag_from_filename(f_path)
        if lang_tag:
            logger.info(f"Loading {f_path.name} with tag '{lang_tag}' as a stream.")
            stream_dataset = load_dataset("text", data_files=[str(f_path)], split='train', streaming=True)
            tagged_stream = stream_dataset.map(lambda example: {"text": f"{lang_tag}{example['text']}"})
            all_streaming_datasets.append(tagged_stream)
        else:
            logger.warning(f"Could not infer language for file: {f_path.name}. Skipping this file.")

    if not all_streaming_datasets:
        raise ValueError("No datasets were loaded.")

    dataset = interleave_datasets(all_streaming_datasets)
    # MODIFICATION 3 (Optional but recommended): Increase buffer for better shuffling if you have RAM
    dataset = dataset.shuffle(seed=42, buffer_size=20000)

    val_size = 1000
    eval_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)
    logger.info(f"Dataset prepared in streaming mode. Using {val_size} examples for validation.")

    # --- 4. Tokenize and Group the Dataset ---
    logger.info("Tokenizing and formatting the dataset for language modeling...")
    block_size = min(MODEL_CONFIG["max_position_embeddings"], 2048)
    
    def tokenize_and_group(examples):
        tokenized_output = tokenizer(examples["text"])
        concatenated_examples = {k: sum(tokenized_output[k], []) for k in tokenized_output.keys()}
        total_length = len(concatenated_examples[list(tokenized_output.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_train_dataset = train_dataset.map(tokenize_and_group, batched=True, remove_columns=["text"])
    lm_eval_dataset = eval_dataset.map(tokenize_and_group, batched=True, remove_columns=["text"])
    logger.info("Tokenization and grouping mapping is set up for the streams.")
    
    # --- 5. Set up the Trainer ---
    logger.info("Setting up the Hugging Face Trainer.")
    training_args = TrainingArguments(**TRAINING_ARGS)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset, tokenizer=tokenizer, data_collator=data_collator,
    )

    # --- 6. Start Training ---
    logger.info("Starting or resuming model pretraining...")
    try:
        # MODIFICATION 4: Tell the trainer to resume from the latest checkpoint
        # It will automatically find `checkpoint-12000` inside your output directory
        train_result = trainer.train(resume_from_checkpoint=True)

        logger.info("Training complete. Saving final model and metrics.")
        trainer.save_model()
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = val_size
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        emergency_path = PRETRAINED_CHECKPOINTS_DIR / "interrupted_checkpoint"
        trainer.save_model(output_dir=str(emergency_path))
        logger.info(f"Saved an emergency checkpoint to {emergency_path}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()