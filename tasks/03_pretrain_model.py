import os
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    # LlamaTokenizer, # Removed: No longer needed as a bridge
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
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
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
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "evaluation_strategy": "steps",
    "eval_steps": 2500,
    "save_strategy": "steps",
    "save_steps": 2500,
    "save_total_limit": 3,
    "logging_steps": 100,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "lr_scheduler_type": "cosine",
    "fp16": MODEL_CONFIG["torch_dtype"] == "float16",
    "bf16": MODEL_CONFIG["torch_dtype"] == "bfloat16",
    "load_best_model_at_end": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "do_eval": True,
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
    
    # The tokenizer should now be directly loadable as a Hugging Face tokenizer
    # after running 02_train_tokeniser.py successfully.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token: '{tokenizer.eos_token}'")
        logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {TOKENIZER_DIR}. Please ensure 02_train_tokeniser.py "
                     f"has been run successfully to create Hugging Face compatible tokenizer files. Error: {e}")
        raise

    # --- 2. Configure and Initialize Model ---
    logger.info("Configuring and initializing Qwen3 model from scratch.")
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        **MODEL_CONFIG,
        vocab_size=len(tokenizer), # Ensure model vocab size matches the trained tokenizer
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    logger.info(f"Model initialized with Qwen3 architecture and ~{params:.2f}M trainable parameters.")
    if not 100 <= params <= 150:
        logger.warning(
            f"WARNING: Model parameter count ({params:.2f}M) is outside the target range of 100M-150M."
        )

    # --- 3. Load and Prepare Datasets ---
    processed_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
    if not processed_files:
        raise FileNotFoundError(f"No processed .txt files found in {PROCESSED_DATA_DIR}.")

    logger.info("Loading processed text data...")
    dataset = load_dataset("text", data_files=[str(f) for f in processed_files], split='train')
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
    logger.info(f"Dataset split. Train size: {len(split_dataset['train'])}, Validation size: {len(split_dataset['test'])}")

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

    lm_datasets = split_dataset.map(
        tokenize_and_group,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
        desc=f"Grouping texts into chunks of {block_size}",
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["test"]
    logger.info(f"Tokenization complete. Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")
    
    # --- 5. Set up the Trainer ---
    logger.info("Setting up the Hugging Face Trainer.")
    training_args = TrainingArguments(**TRAINING_ARGS)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=data_collator,
    )

    # --- 6. Start Training ---
    logger.info("Starting model pretraining...")
    try:
        train_result = trainer.train()
        logger.info("Training complete. Saving final model and metrics.")
        trainer.save_model()
        trainer.save_state()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info("Performing final evaluation on the validation set...")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
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