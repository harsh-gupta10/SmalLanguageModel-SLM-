import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import re
import os
import json
from tqdm import tqdm
import math

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths to Model, Tokenizer, and Data ---
MODEL_DIR = PROJECT_ROOT / "model" / "checkpoints" / "pretrained" / "checkpoint-24000"
TOKENIZER_DIR = PROJECT_ROOT / "model" / "tokenizer"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- Evaluation Settings ---
PERPLEXITY_STRIDE = 512
PERPLEXITY_MAX_LENGTH = 2048

# (Helper functions like get_file_stats, count_tokens_in_file, clean_generated_text remain the same)
# ... [Keeping the helper functions from the previous script here for brevity] ...
def get_file_stats(filepath: Path):
    """Calculates the size and line count of a file."""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = sum(1 for line in f)
    return {"size_mb": size_mb, "num_lines": lines}

def count_tokens_in_file(filepath: Path, tokenizer) -> int:
    """Counts the total number of tokens in a text file."""
    if not filepath.exists():
        return 0
    total_tokens = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Tokenizing {filepath.name}", unit=" lines"):
            total_tokens += len(tokenizer.encode(line))
    return total_tokens

def clean_generated_text(text: str, language: str) -> str:
    """Cleans the generated text."""
    text = re.sub(r'<(en|hi|sa)>', '', text)
    if language == "English":
        text = re.sub(r'[\u0900-\u097F]+', '', text)
    elif language in ["Hindi", "Sanskrit"]:
        text = re.sub(r'[a-zA-Z]+', '', text)
    text = re.sub(r'[\s.,\'"-?!’]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Main Evaluation Functions ---

def get_model_statistics(model):
    """Calculates and returns statistics about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = model.config

    stats = {
        "Total Parameters": f"{total_params / 1e6:.2f}M",
        "Trainable Parameters": f"{trainable_params / 1e6:.2f}M",
        "Model Architecture": config.model_type,
        "Number of Layers": config.num_hidden_layers,
        "Hidden Size": config.hidden_size,
        "Number of Attention Heads": config.num_attention_heads,
        "Vocabulary Size (from model config)": config.vocab_size
    }
    return stats

def get_data_and_tokenizer_statistics(tokenizer, data_dir: Path):
    """Calculates statistics for the dataset and tokenizer."""
    print("\n--- Gathering Data and Tokenizer Statistics ---")
    if not data_dir.exists():
        print(f"FATAL: Data directory not found at {data_dir}")
        return {}

    stats = {
        "Tokenizer": {
            "Class": tokenizer.__class__.__name__,
            "Vocabulary Size": tokenizer.vocab_size
        },
        "Dataset (Train Split)": {}
    }

    languages = ["english", "hindi", "sanskrit"]
    for lang in languages:
        filepath = data_dir / f"train_{lang}.txt"
        if filepath.exists():
            print(f"Found train file for {lang}: {filepath}")
            file_stats = get_file_stats(filepath)
            token_count = count_tokens_in_file(filepath, tokenizer)
            stats["Dataset (Train Split)"][lang.capitalize()] = {
                "File Size (MB)": f"{file_stats['size_mb']:.2f}",
                "Number of Sentences/Lines": file_stats['num_lines'],
                "Number of Tokens": f"{token_count / 1e6:.2f}M"
            }
        else:
            print(f"WARNING: Train file not found at {filepath}. Skipping stats for this language.")

    return stats

def calculate_perplexity(model, tokenizer, device):
    """Calculates perplexity on the test sets."""
    print("\n--- Starting Perplexity Calculation ---")
    results = {}
    total_nlls = []
    total_tokens = 0

    languages = ["english", "hindi", "sanskrit"]
    for lang in languages:
        test_filepath = DATA_DIR / f"test_{lang}.txt"
        if not test_filepath.exists():
            print(f"WARNING: Test file not found at {test_filepath}, skipping perplexity for this language.")
            continue

        print(f"\nEvaluating on {lang.capitalize()} test set...")
        with open(test_filepath, "r", encoding="utf-8") as f:
            text = f.read()

        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, PERPLEXITY_STRIDE), desc=f"Calculating PPL for {lang}"):
            end_loc = min(begin_loc + PERPLEXITY_MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            total_nlls.append(neg_log_likelihood)
            total_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if nlls:
            ppl = torch.exp(torch.stack(nlls).sum() / (end_loc - 1))
            results[f"Perplexity ({lang.capitalize()})"] = ppl.item()
        else:
            results[f"Perplexity ({lang.capitalize()})"] = "N/A"

    if total_nlls:
        overall_ppl = torch.exp(torch.stack(total_nlls).sum() / total_tokens)
        results["Perplexity (Overall)"] = overall_ppl.item()
    else:
        results["Perplexity (Overall)"] = "N/A"

    return results

def generate_qualitative_examples(model, tokenizer):
    """Generates text from prompts to showcase model capabilities."""
    print("\n--- Generating Qualitative Examples ---")
    
    # ========================================================================
    # FIXED: Removed the `device` argument when `device_map="auto"` is used.
    # ========================================================================
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    prompts = {
        "English": "<en>Once upon a time in a land of algorithms,",
        "Hindi": "<hi>एक समय की बात है, डिजिटल दुनिया में",
        "Sanskrit": "<sa>कदाचित्कालः आसीत्, यत्र संगणकाः"
    }

    for lang, prompt in prompts.items():
        print(f"\n--- Language: {lang} ---")
        print(f"Prompt: {prompt}")

        generated_outputs = text_generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        )

        print("\nCleaned Generated Text:")
        for output in generated_outputs:
            raw_text = output['generated_text']
            cleaned_text = clean_generated_text(raw_text, lang)
            print(cleaned_text)


def main():
    """Main function to run the complete evaluation."""
    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model from: {MODEL_DIR}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found. Check path: {MODEL_DIR}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"
    )
    device = model.device
    print(f"Model loaded successfully on device(s): {device}")

    print(f"Loading tokenizer from: {TOKENIZER_DIR}")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Tokenizer directory not found. Check path: {TOKENIZER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # --- 2. Gather and Print Statistics ---
    # model_stats = get_model_statistics(model)
    # data_stats = get_data_and_tokenizer_statistics(tokenizer, DATA_DIR)

    # full_report = {
    #     "Model Statistics": model_stats,
    #     **data_stats
    # }

    # print("\n\n--- EVALUATION REPORT ---")
    # print(json.dumps(full_report, indent=2))

    # --- 3. Calculate Perplexity ---
    perplexity_results = calculate_perplexity(model, tokenizer, device)
    print("\n--- Perplexity Results ---")
    print(json.dumps(perplexity_results, indent=2))

    # --- 4. Generate Qualitative Examples ---
    generate_qualitative_examples(model, tokenizer)

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()