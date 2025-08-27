# tasks/02a_evaluate_tokenizer.py

import os
import json
import sentencepiece as spm
from tqdm import tqdm

# --- Configuration ---
TOKENIZER_MODEL = 'model/tokenizer/multilingual_spm.model'
RAW_DATA_DIR = 'data/raw'

# Define the language files to evaluate
LANG_FILES = {
    'english': 'lang_english.txt',
    'hindi': 'lang_hindi.txt',
    'sanskrit': 'lang_sanskrit.txt'
}

# Number of lines to sample from each file for a quick and effective evaluation.
# Using the full file can be very slow. 500,000 lines is a very robust sample.
SAMPLE_SIZE = 500000 

def evaluate_coverage():
    """
    Calculates the token coverage for each language by finding the rate of <unk> tokens.
    """
    print("--- Starting Quantitative Evaluation: Token Coverage ---")

    if not os.path.exists(TOKENIZER_MODEL):
        print(f"Error: Tokenizer model not found at {TOKENIZER_MODEL}")
        return

    # 1. Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    unk_token_id = sp.unk_id()
    
    report = {}

    print(f"Analyzing a sample of {SAMPLE_SIZE:,} lines from each language file...")

    # 2. Iterate through each language file
    for lang, filename in LANG_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found for '{lang}': {filepath}. Skipping.")
            continue

        print(f"\nProcessing '{lang}'...")
        total_tokens = 0
        unk_tokens = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            # 3. Process a large sample of lines from the file
            for i, line in tqdm(enumerate(f), total=SAMPLE_SIZE):
                if i >= SAMPLE_SIZE:
                    break
                
                # Tokenize the line into IDs
                ids = sp.encode_as_ids(line)
                
                # 4. Count total tokens and unknown tokens
                total_tokens += len(ids)
                unk_tokens += ids.count(unk_token_id)
        
        # 5. Calculate metrics
        if total_tokens > 0:
            unk_rate = (unk_tokens / total_tokens) * 100
            coverage = 100 - unk_rate
            report[lang] = {
                'total_tokens_sampled': total_tokens,
                'unknown_tokens_found': unk_tokens,
                'unknown_token_rate_%': round(unk_rate, 4),
                'token_coverage_%': round(coverage, 4)
            }
        else:
            report[lang] = "No tokens found to analyze."

    # 6. Print and save the final report
    print("\n--- Tokenizer Coverage Report ---")
    print(json.dumps(report, indent=4))

    report_path = 'model/tokenizer/coverage_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    print(f"\nFull report saved to {report_path}")

def qualitative_analysis():
    """
    Performs a simple qualitative check by printing parts of the vocabulary.
    """
    print("\n--- Starting Qualitative Analysis: Vocabulary Inspection ---")
    vocab_file = TOKENIZER_MODEL.replace('.model', '.vocab')

    if not os.path.exists(vocab_file):
        print(f"Error: Vocab file not found at {vocab_file}")
        return
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        lines = [line.split('\t')[0] for line in f] # Get just the token, not the score

    print("\nSample of initial vocabulary (special tokens and common subwords):")
    for token in lines[0:20]:
        print(f"  {token}")

    print("\nSample of mid-range vocabulary (should see mix of scripts):")
    mid_start = len(lines) // 2
    for token in lines[mid_start : mid_start + 20]:
        print(f"  {token}")
        
    print("\nSample of final vocabulary (likely rare subwords):")
    for token in lines[-20:]:
        print(f"  {token}")

    print("\nQualitative check: Do you see a mix of English characters and Devanagari script (e.g., 'ार', 'ation') in the samples above? If so, it's a good sign.")


if __name__ == '__main__':
    evaluate_coverage()
    qualitative_analysis()