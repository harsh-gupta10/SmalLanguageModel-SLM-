import os
import re
import gc
import unicodedata # For more robust script checking
from tqdm import tqdm
from langdetect import detect, LangDetectException
from multiprocessing import Pool, cpu_count, Manager # Import Manager here

# --- Configuration ---
INPUT_DIR = 'data/raw'
OUTPUT_DIR = 'data/cleaned' # Cleaned files will be saved here

# Input file names
INPUT_FILES = {
    'english': 'lang_english.txt',
    'hindi': 'lang_hindi.txt',
    'sanskrit': 'lang_sanskrit.txt',
}

# Cleaning Parameters
MIN_LINE_LENGTH = 10  # Minimum number of characters for a line to be considered valid
MAX_NON_ALPHANUM_RATIO = 0.3 # Max ratio of non-alphanumeric chars (e.g., symbols, numbers)

# Multiprocessing Parameters
NUM_PROCESSES = cpu_count() # Use all available CPU cores
CHUNK_SIZE = 10000          # Number of lines to process per worker at a time

# --- Regex for quick script checks ---
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]+')
LATIN_REGEX = re.compile(r'[a-zA-Z0-9]+')


# --- Helper Functions ---
def has_devanagari(text):
    return bool(DEVANAGARI_REGEX.search(text))

def has_latin(text):    
    return bool(LATIN_REGEX.search(text))

def clean_line(line):
    """Applies general cleaning to a single line of text."""
    line = re.sub(r'http\S+|www\S+|https\S+', '', line, flags=re.MULTILINE) # Remove URLs
    line = re.sub(r'\S*@\S*\s?', '', line) # Remove emails
    line = re.sub(r'\s+', ' ', line).strip() # Remove extra whitespace
    return line

def passes_general_filters(line):
    """Checks line length and non-alphanumeric ratio."""
    if len(line) < MIN_LINE_LENGTH:
        return False
    
    alphanum_count = sum(c.isalnum() or c.isspace() for c in line)
    total_chars = len(line)
    
    if total_chars == 0:
        return False
    
    non_alphanum_ratio = (total_chars - alphanum_count) / total_chars
    if non_alphanum_ratio > MAX_NON_ALPHANUM_RATIO:
        return False
        
    return True

# Worker function for multiprocessing pool
def process_line_worker(line_and_lang_key):
    """
    Worker function to clean and filter a single line.
    Returns the cleaned line if valid, otherwise None.
    """
    line, lang_key = line_and_lang_key # Unpack data

    cleaned_line = clean_line(line)

    if not passes_general_filters(cleaned_line):
        return None

    # --- Quick Script Pre-filtering ---
    # This helps skip langdetect calls for lines clearly not in the target script
    if lang_key == 'english':
        if not has_latin(cleaned_line): return None
    elif (lang_key == 'hindi' or lang_key == 'sanskrit'):
        if not has_devanagari(cleaned_line): return None
    # Add a check for mixed scripts, e.g., if a Hindi line has too many Latin chars
    # For robust mixed script filtering, one might count char types and set a threshold.
    # For now, relying on langdetect to handle mixed content more finely.

    # --- Language Specific Filtering using langdetect ---
    try:
        detected_lang = detect(cleaned_line)

        is_valid = False
        if lang_key == 'english':
            if detected_lang == 'en':
                is_valid = True
        elif lang_key == 'hindi':
            if detected_lang == 'hi':
                is_valid = True
        elif lang_key == 'sanskrit':
            # langdetect often misclassifies Sanskrit as Hindi. Allow both.
            if detected_lang == 'sa' or detected_lang == 'hi':
                is_valid = True
        
        if is_valid:
            return cleaned_line # Return the cleaned line if valid
        else:
            return None # Failed language check
            
    except LangDetectException:
        # Couldn't detect language, skip line
        return None
    except Exception as e:
        # Catch any other unexpected errors during processing a line
        # print(f"Error processing line: {e} - '{cleaned_line}'") # Uncomment for verbose error logging
        return None


def preprocess_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # This set will hold ALL unique cleaned lines across all languages.
    # It will be populated after all multiprocessing is done.
    final_global_cleaned_lines_set = set()

    # Store results for each language temporarily in a list (may contain duplicates initially)
    # The deduplication will happen in a final pass.
    all_cleaned_lines_collected = {lang: [] for lang in INPUT_FILES.keys()}

    # Use a pool of processes for parallel work
    print(f"Starting multiprocessing pool with {NUM_PROCESSES} workers...")
    with Pool(processes=NUM_PROCESSES) as pool:
        for lang_key, input_filename in INPUT_FILES.items():
            input_filepath = os.path.join(INPUT_DIR, input_filename)
            
            if not os.path.exists(input_filepath):
                print(f"Skipping {input_filename}: File not found.")
                continue
                
            print(f"\nProcessing {input_filename} ({lang_key})...")
            
            # Count total lines for tqdm
            with open(input_filepath, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

            lines_kept_count_per_lang = 0
            
            # Prepare arguments for the worker function
            lines_to_process = ((line, lang_key) for line in open(input_filepath, 'r', encoding='utf-8'))
            
            # Use imap_unordered to get results as they are ready
            for result in tqdm(pool.imap_unordered(process_line_worker, lines_to_process, chunksize=CHUNK_SIZE), total=total_lines, desc=f"Cleaning {lang_key} lines"):
                if result is not None:
                    all_cleaned_lines_collected[lang_key].append(result)
                    lines_kept_count_per_lang += 1
                
                # Periodically collect garbage in the main process (less critical with this design)
                if lines_kept_count_per_lang % 100000 == 0:
                    gc.collect()

            print(f"Finished processing {input_filename}. Kept {lines_kept_count_per_lang} lines before global deduplication.")

    print("\nAll files processed by workers. Performing final deduplication and saving...")

    # Now, perform global deduplication and save to disk
    total_saved_lines = 0
    for lang_key, cleaned_lines_list in all_cleaned_lines_collected.items():
        output_filepath = os.path.join(OUTPUT_DIR, f'cleaned_{INPUT_FILES[lang_key]}')
        
        lines_for_this_file = []
        for line in cleaned_lines_list:
            if line not in final_global_cleaned_lines_set:
                final_global_cleaned_lines_set.add(line)
                lines_for_this_file.append(line)
        
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for cleaned_line in lines_for_this_file:
                outfile.write(cleaned_line + '\n')
        
        print(f"Saved {len(lines_for_this_file)} unique cleaned lines for {lang_key} to {output_filepath}")
        total_saved_lines += len(lines_for_this_file)
        
        del cleaned_lines_list, lines_for_this_file # Free memory
        gc.collect()
    
    print(f"\nCleaning complete! Total unique lines saved across all languages: {len(final_global_cleaned_lines_set)}")
    print(f"Total lines written to files (may include within-language duplicates if not fully unique): {total_saved_lines}")

if __name__ == '__main__':
    preprocess_files()