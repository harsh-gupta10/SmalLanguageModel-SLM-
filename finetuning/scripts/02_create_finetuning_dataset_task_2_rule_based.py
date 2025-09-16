import spacy
import random
import json
import nltk
import os
import subprocess
import re
from nltk.tokenize import sent_tokenize

# --------------------------
# Configure custom data paths
# --------------------------
BASE_DATA_DIR = os.path.join(os.getcwd(), "finetuning")
NLTK_DATA_DIR = os.path.join(BASE_DATA_DIR, "nltk_data")
SAVE_DATA_DIR = BASE_DATA_DIR + "/data"

os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)


def download_nltk_resources():
    """Checks for and downloads all required NLTK resources if they are missing."""
    resources_to_check = {
        'punkt': 'tokenizers/punkt',
        'gutenberg': 'corpora/gutenberg'
    }
    for resource_name, resource_path in resources_to_check.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"NLTK resource '{resource_name}' not found. Downloading...")
            nltk.download(resource_name, download_dir=NLTK_DATA_DIR)
            print("Download complete.")


def load_spacy_model():
    """Loads the spaCy English model, downloading it if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Downloading...")
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Cleans up artifacts after token swapping:
    - Removes extra spaces before punctuation
    - Collapses multiple spaces
    - Trims leading/trailing whitespace
    """
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)   # no space before punctuation
    text = re.sub(r'([({\[])\s+', r'\1', text)    # no space after opening brackets
    text = re.sub(r'\s+([)}\]])', r'\1', text)    # no space before closing brackets
    text = re.sub(r'\s+', ' ', text)              # collapse multiple spaces
    return text.strip()


def generate_dataset(sentences, num_samples, nlp):
    """
    Generates dataset by swapping nouns and verbs with post-processing cleanup.
    """
    dataset = []
    processed_sentences = 0
    
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        print("Error: Input 'sentences' must be a list of strings.")
        return []

    random.shuffle(sentences)

    for sentence in sentences:
        if processed_sentences >= num_samples:
            break

        doc = nlp(sentence.strip())
        nouns = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        verbs = [token for token in doc if token.pos_ == "VERB"]

        if not nouns or not verbs:
            continue
        
        max_swaps = min(len(nouns), len(verbs))
        num_to_swap = random.randint(1, max_swaps)

        nouns_to_swap = random.sample(nouns, k=num_to_swap)
        verbs_to_swap = random.sample(verbs, k=num_to_swap)

        noun_replacements = list(nouns_to_swap)
        verb_replacements = list(verbs_to_swap)
        random.shuffle(noun_replacements)
        random.shuffle(verb_replacements)
        
        swap_map = {}
        for i in range(num_to_swap):
            original_noun = nouns_to_swap[i]
            original_verb = verbs_to_swap[i]
            swap_map[original_noun.i] = verb_replacements[i]
            swap_map[original_verb.i] = noun_replacements[i]

        modified_sentence_parts = []
        for token in doc:
            if token.i in swap_map:
                replacement_token = swap_map[token.i]
                modified_sentence_parts.append(replacement_token.text + token.whitespace_)
            else:
                modified_sentence_parts.append(token.text_with_ws)

        original_sentence = doc.text
        modified_sentence = "".join(modified_sentence_parts)

        # Clean post-processed output
        modified_sentence = clean_text(modified_sentence)

        if original_sentence != modified_sentence:
            dataset.append({
                "instruction": "Identify and swap nouns with verbs, and verbs with nouns, in the given sentence.",
                "input": clean_text(original_sentence),
                "output": modified_sentence
            })
            processed_sentences += 1

    return dataset


def main():
    """Main function to generate the dataset and save it to a file."""
    download_nltk_resources()
    nlp = load_spacy_model()

    print("Loading text corpus...")
    from nltk.corpus import gutenberg

    raw_text = gutenberg.raw('austen-sense.txt')
    
    # Replace all sequences of whitespace with a single space.
    raw_text = re.sub(r'\s+', ' ', raw_text)

    sentences = sent_tokenize(raw_text)

    num_samples_to_generate = 5000
    print(f"Generating {num_samples_to_generate} samples...")
    generated_data = generate_dataset(sentences, num_samples_to_generate, nlp)

    output_filename = os.path.join(SAVE_DATA_DIR, "noun_verb_swap_dataset.json")
    with open(output_filename, "w") as f:
        json.dump(generated_data, f, indent=4)

    print(f"Dataset generation complete. Saved {len(generated_data)} samples to {output_filename}")
    print("\n--- Sample Generated Data (Cleaned) ---")
    for i in range(min(5, len(generated_data))):
        print(json.dumps(generated_data[i], indent=2))


if __name__ == "__main__":
    main()
