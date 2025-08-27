# tasks/01_preprocess_data.py

import os
import json
import random
import ftfy
import regex as re
import sentencepiece as spm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
TOKENIZER_MODEL = 'model/tokenizer/multilingual_spm.model'

# Define the language files and their keys
LANG_FILES = {
    'english': 'lang_english.txt',
    'hindi': 'lang_hindi.txt',
    'sanskrit': 'lang_sanskrit.txt'
}

# Define the train, validation, and test split ratios
VALIDATION_SPLIT = 0.01
TEST_SPLIT = 0.01

def clean_text(text: str) -> str:
    """
    Applies basic text cleaning to a single line of text.
    - Fixes unicode errors with ftfy.
    - Normalizes whitespace.
    - Strips leading/trailing whitespace.
    """
    # Fix unicode encoding and mojibake
    text = ftfy.fix_text(text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_file(filepath: str) -> set:
    """
    Reads a file, cleans its lines, and returns a set of unique, non-empty lines.
    Using a set handles deduplication within the file itself.
    """
    unique_lines = set()
    print(f"Processing file: {os.path.basename(filepath)}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            cleaned = clean_text(line)
            if cleaned:  # Ensure the line is not empty after cleaning
                unique_lines.add(cleaned)
    return unique_lines

def calculate_token_stats():
    """
    Loads the trained tokenizer and calculates the token count and percentage for each language.
    """
    print("\n--- Calculating Token Statistics ---")
    
    if not os.path.exists(TOKENIZER_MODEL):
        print(f"Error: Tokenizer model not found at {TOKENIZER_MODEL}")
        return

    # Load the tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    
    stats = {}
    total_tokens = 0

    # Calculate tokens for each language file
    for lang, filename in LANG_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        print(f"Counting tokens in {filename}...")
        lang_token_count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            # Reading in chunks for memory efficiency with large files
            for line in tqdm(f):
                lang_token_count += len(sp.encode_as_ids(line))
        stats[lang] = {'token_count': lang_token_count}
        total_tokens += lang_token_count

    stats['total_tokens'] = total_tokens
    
    # Calculate percentages
    if total_tokens > 0:
        for lang in LANG_FILES.keys():
            percentage = (stats[lang]['token_count'] / total_tokens) * 100
            stats[lang]['percentage'] = round(percentage, 2)

    # Print and save the report
    print("\n--- Token Statistics Report ---")
    print(json.dumps(stats, indent=4))
    
    report_path = os.path.join(PROCESSED_DATA_DIR, 'token_statistics.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"\nStatistics report saved to {report_path}")


def main():
    """
    Main function to orchestrate the preprocessing workflow.
    """
    print("--- Starting Phase 1: Data Preprocessing ---")

    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # --- 1. Clean and Deduplicate Data in Parallel ---
    filepaths = [os.path.join(RAW_DATA_DIR, fname) for fname in LANG_FILES.values()]
    
    # Use a multiprocessing Pool to process files concurrently
    num_procs = min(len(filepaths), cpu_count())
    print(f"\nStarting file processing with {num_procs} processes...")
    with Pool(processes=num_procs) as pool:
        # pool.map will return a list of sets (one for each file)
        list_of_sets = pool.map(process_file, filepaths)

    # Combine all sets into one master set for global deduplication
    print("\nCombining and performing global deduplication...")
    all_unique_lines = set().union(*list_of_sets)
    
    # Convert set to list for shuffling and splitting
    all_lines = list(all_unique_lines)
    print(f"Total unique lines after global deduplication: {len(all_lines):,}")

    # --- 2. Shuffle and Split Data ---
    print("\nShuffling data...")
    random.shuffle(all_lines)
    
    # Calculate split indices
    total_size = len(all_lines)
    test_idx = int(total_size * TEST_SPLIT)
    val_idx = int(total_size * (VALIDATION_SPLIT + TEST_SPLIT))
    
    test_set = all_lines[:test_idx]
    validation_set = all_lines[test_idx:val_idx]
    train_set = all_lines[val_idx:]

    print(f"Train set size:      {len(train_set):,}")
    print(f"Validation set size: {len(validation_set):,}")
    print(f"Test set size:       {len(test_set):,}")

    # --- 3. Write Processed Files ---
    print("\nWriting processed files to disk...")
    
    splits = {
        'train.txt': train_set,
        'validation.txt': validation_set,
        'test.txt': test_set
    }
    
    for filename, dataset in splits.items():
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in tqdm(dataset, desc=f"Writing {filename}"):
                f.write(line + '\n')
    
    print("\n--- Preprocessing complete! ---")

    # --- 4. Calculate Token Statistics ---
    # This is done last as it's a reporting step.
    calculate_token_stats()


if __name__ == '__main__':
    main()