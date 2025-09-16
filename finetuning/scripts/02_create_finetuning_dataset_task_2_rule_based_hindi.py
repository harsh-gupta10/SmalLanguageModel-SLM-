import stanza
import random
import json
import os
import re

CORPUS_PATH = "data/raw/lang_hindi.txt" 

BASE_DATA_DIR = os.path.join(os.getcwd(), "finetuning_hindi")
SAVE_DATA_DIR = os.path.join(BASE_DATA_DIR, "data")
OUTPUT_FILENAME = os.path.join(SAVE_DATA_DIR, "hindi_noun_verb_swap_dataset.json")

NUM_SAMPLES_TO_GENERATE = 5000

os.makedirs(SAVE_DATA_DIR, exist_ok=True)


def download_stanza_model():
    print("Initializing Stanza pipeline for Hindi...")
    try:
        return stanza.Pipeline('hi', processors='tokenize,pos', download_method=None)
    except Exception:
        print("Stanza model for Hindi not found. Downloading...")
        stanza.download('hi')
        return stanza.Pipeline('hi', processors='tokenize,pos')

def generate_dataset_from_stream(corpus_path, num_samples, nlp):
    dataset = []
    processed_count = 0

    print(f"Starting dataset generation from {corpus_path}...")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if processed_count >= num_samples:
                print(f"Target number of samples ({num_samples}) reached. Stopping.")
                break

            line = line.strip()
            if not line:
                continue

            doc = nlp(line)
            
            for sentence in doc.sentences:
                nouns = [word for word in sentence.words if word.upos in ["NOUN", "PROPN"]]
                verbs = [word for word in sentence.words if word.upos == "VERB"]

                if not nouns or not verbs:
                    continue
                
                # Determine how many pairs to swap (at least 1)
                max_swaps = min(len(nouns), len(verbs))
                num_to_swap = random.randint(1, max_swaps)

                # Randomly select the words to be swapped
                nouns_to_swap = random.sample(nouns, k=num_to_swap)
                verbs_to_swap = random.sample(verbs, k=num_to_swap)

                # Create shuffled lists of replacements
                noun_replacements = list(nouns_to_swap)
                verb_replacements = list(verbs_to_swap)
                random.shuffle(noun_replacements)
                random.shuffle(verb_replacements)
                
                # Map original word positions to their new replacement words
                swap_map = {}
                for i in range(num_to_swap):
                    original_noun = nouns_to_swap[i]
                    original_verb = verbs_to_swap[i]
                    # Stanza word IDs are 1-based, list indices are 0-based
                    swap_map[original_noun.id - 1] = verb_replacements[i]
                    swap_map[original_verb.id - 1] = noun_replacements[i]

                original_sentence_text = sentence.text
                
                # Build the new sentence by replacing words at the mapped indices
                temp_sentence_parts = [word.text for word in sentence.words]
                for index, replacement_word in swap_map.items():
                    temp_sentence_parts[index] = replacement_word.text
                
                # Reconstruct the sentence string
                modified_sentence = " ".join(temp_sentence_parts)
                # Clean up potential spacing issues around punctuation
                modified_sentence = re.sub(r'\s([?.!,।])', r'\1', modified_sentence)

                if original_sentence_text != modified_sentence:
                    dataset.append({
                        "instruction": "दिए गए वाक्य में संज्ञा को क्रिया से और क्रिया को संज्ञा से पहचानें और बदलें।",
                        "input": original_sentence_text,
                        "output": modified_sentence
                    })
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Generated {processed_count}/{num_samples} samples...")
                    if processed_count >= num_samples:
                        break
                        
    return dataset


def main():
    """
    Main function to coordinate the dataset generation process.
    """
    # Verify that the input corpus file exists before starting
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus file not found at '{CORPUS_PATH}'")
        print("Please make sure the file exists or update the CORPUS_PATH variable in the script.")
        return

    # Initialize the NLP model
    nlp = download_stanza_model()

    # Generate the dataset by streaming from the file
    generated_data = generate_dataset_from_stream(CORPUS_PATH, NUM_SAMPLES_TO_GENERATE, nlp)

    # Save the generated data to a JSON file
    if generated_data:
        print(f"\nDataset generation complete. Saving {len(generated_data)} samples to {OUTPUT_FILENAME}")
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
        print("File saved successfully.")
        
        # Print a few samples to verify the output
        print("\n--- Sample Generated Data ---")
        for i in range(min(5, len(generated_data))):
            print(json.dumps(generated_data[i], indent=2, ensure_ascii=False))
    else:
        print("No data was generated. Please check the input corpus and script settings.")


if __name__ == "__main__":
    main()