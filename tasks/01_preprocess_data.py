# tasks/01_preprocess_data.py

import os
import json
import random
import ftfy
import regex as re
import sentencepiece as spm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
TOKENIZER_MODEL = 'model/tokenizer/multilingual_spm.model'

LANG_FILES = {
    'english': 'lang_english.txt',
    'hindi': 'lang_hindi.txt',
    'sanskrit': 'lang_sanskrit.txt'
}

VALIDATION_SPLIT = 0.01
TEST_SPLIT = 0.01

def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_file(filepath: str) -> set:
    unique_lines = set()
    print(f"Processing file: {os.path.basename(filepath)}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            cleaned = clean_text(line)
            if cleaned:
                unique_lines.add(cleaned)
    return unique_lines

def calculate_token_stats():
    print("\n--- Calculating Token Statistics ---")
    
    if not os.path.exists(TOKENIZER_MODEL):
        print(f"Error: Tokenizer model not found at {TOKENIZER_MODEL}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    
    stats = {}
    total_tokens = 0

    for lang, filename in LANG_FILES.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        print(f"Counting tokens in {filename}...")
        lang_token_count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lang_token_count += len(sp.encode_as_ids(line))
        stats[lang] = {'token_count': lang_token_count}
        total_tokens += lang_token_count

    stats['total_tokens'] = total_tokens
    
    if total_tokens > 0:
        for lang in LANG_FILES.keys():
            percentage = (stats[lang]['token_count'] / total_tokens) * 100
            stats[lang]['percentage'] = round(percentage, 2)

    print("\n--- Token Statistics Report ---")
    print(json.dumps(stats, indent=4))
    
    report_path = os.path.join(PROCESSED_DATA_DIR, 'token_statistics.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"\nStatistics report saved to {report_path}")


def main():
    print("--- Starting Phase 1: Data Preprocessing ---")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    filepaths = [os.path.join(RAW_DATA_DIR, fname) for fname in LANG_FILES.values()]
    
    num_procs = min(len(filepaths), cpu_count())
    print(f"\nStarting file processing with {num_procs} processes...")
    with Pool(processes=num_procs) as pool:
        list_of_sets = pool.map(process_file, filepaths)

    print("\nCombining and performing global deduplication...")
    all_unique_lines = set().union(*list_of_sets)
    
    all_lines = list(all_unique_lines)
    print(f"Total unique lines after global deduplication: {len(all_lines):,}")

    print("\nShuffling data...")
    random.shuffle(all_lines)
    
    total_size = len(all_lines)
    test_idx = int(total_size * TEST_SPLIT)
    val_idx = int(total_size * (VALIDATION_SPLIT + TEST_SPLIT))
    
    test_set = all_lines[:test_idx]
    validation_set = all_lines[test_idx:val_idx]
    train_set = all_lines[val_idx:]

    print(f"Train set size:      {len(train_set):,}")
    print(f"Validation set size: {len(validation_set):,}")
    print(f"Test set size:       {len(test_set):,}")

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

    calculate_token_stats()


if __name__ == '__main__':
    main()