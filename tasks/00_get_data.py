# tasks/01_get_data.py

import os
from datasets import load_dataset

RAW_DATA_DIR = "data/raw"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

def collect_english_data():
    output_path = os.path.join(RAW_DATA_DIR, "lang_english.txt")
    target_size_gb = 6.5  # Target for ~1.5 billion tokens (50%)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f"'{output_path}' already exists. Skipping download.")
        return

    print("--- Starting English Data Collection ---")

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    print(f"Streaming English data to '{output_path}'...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example['text'] + "\n")
            if os.path.getsize(output_path) / (1024**3) >= target_size_gb:
                print(f"Reached target size of {target_size_gb:.2f} GB. Stopping.")
                break
    print("--- Finished English Data Collection ---")

def collect_hindi_data():
    output_path = os.path.join(RAW_DATA_DIR, "lang_hindi.txt")
    target_size_gb = 4.5  # Target for ~1.2 billion tokens (40%)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f"'{output_path}' already exists. Skipping download.")
        return

    print("--- Starting Hindi Data Collection ---")
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", streaming=True)
    print(f"Streaming Hindi data to '{output_path}'...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example['text'] + "\n")
            if os.path.getsize(output_path) / (1024**3) >= target_size_gb:
                print(f"Reached target size of {target_size_gb:.2f} GB. Stopping.")
                break
    print("--- Finished Hindi Data Collection ---")
    
def collect_sanskrit_data():
    output_path = os.path.join(RAW_DATA_DIR, "lang_sanskrit.txt")
    target_size_gb = 1.5 # Target for ~300 million tokens (10%)

    print("--- Starting Sanskrit Data Collection ---")
    dataset = load_dataset(
        "ai4bharat/sangraha",
        data_files="verified/san/*.parquet",
        split="train",
        streaming=True
    )
    print(f"Streaming Sanskrit data to '{output_path}'...")

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example['text'] + "\n")
            if os.path.getsize(output_path) / (1024**3) >= target_size_gb:
                print(f"Reached target size of {target_size_gb:.2f} GB. Stopping.")
                break

    print("--- Finished Sanskrit Data Collection ---")

if __name__ == "__main__":
    print("Starting raw data collection process...")
    
    collect_english_data()
    collect_hindi_data()
    collect_sanskrit_data()
    
    print("\nRaw data collection for all languages is complete.")
    print(f"Data saved in: '{RAW_DATA_DIR}'")
