import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import re # Import the regular expressions module

# --- Configuration ---
# This assumes your script is in a 'tasks' directory, and the project root is two levels up.
# Adjust the number of .parents[] if your file structure is different.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Point MODEL_DIR directly to the specific checkpoint
# I've updated this to checkpoint-8000 to match your latest test
MODEL_DIR = PROJECT_ROOT / "model" / "checkpoints" / "pretrained" / "checkpoint-12000"

# The tokenizer directory
TOKENIZER_DIR = PROJECT_ROOT / "model" / "tokenizer"


# --- 1. Load the Fine-Tuned Model and Tokenizer ---
print(f"Loading model from: {MODEL_DIR}")
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}. Please check the path.")

# Determine the torch_dtype to use
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto" # Let accelerate handle device placement
)
print("Model loaded successfully.")

print(f"Loading tokenizer from: {TOKENIZER_DIR}")
if not TOKENIZER_DIR.exists():
    raise FileNotFoundError(f"Tokenizer directory not found at {TOKENIZER_DIR}. Please check the path.")

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")


# --- 2. Post-Processing Function ---
def clean_generated_text(text: str, language: str) -> str:
    """
    Cleans the generated text by removing language tags, filtering out
    characters from incorrect scripts, and removing trailing junk punctuation.
    """
    # Step 1: Remove all language tags like <en>, <hi>, <sa>
    text = re.sub(r'<(en|hi|sa)>', '', text)

    # Step 2: Filter characters based on the target language
    if language == "English":
        # Remove characters in the Devanagari Unicode range
        text = re.sub(r'[\u0900-\u097F]+', '', text)
    elif language in ["Hindi", "Sanskrit"]:
        # Remove Roman alphabet characters
        text = re.sub(r'[a-zA-Z]+', '', text)

    # --- NEW: Step 3: Remove trailing junk punctuation and whitespace ---
    # This regex matches one or more characters from the set [\s.,'"\-?!’]
    # at the end of the string ($) and replaces them with nothing.
    text = re.sub(r'[\s.,\'"-?!’]+$', '', text)

    # Step 4: Normalize whitespace (replace multiple spaces with a single one)
    text = re.sub(r'\s+', ' ', text)

    # Step 5: Final strip to remove any leading/trailing space
    return text.strip()


# --- 3. Set up the Text Generation Pipeline ---
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# --- 4. Define Prompts for Each Language and Generate Text ---
# IMPORTANT: Added the language tags back to the prompts.
# The model NEEDS these to know what language to generate.
# prompts = {
#     "English": "<en>The future of artificial intelligence is",
#     "Hindi": "<hi>कृत्रिम बुद्धिमत्ता का भविष्य",
#     "Sanskrit": "<sa>कृत्रिमप्रज्ञायाः भविष्यम्"
# }

# prompts = {
#     "English": "<en>The sun rises in the",
#     "Hindi": "<hi>सूरज उगता है",
#     "Sanskrit": "<sa>सूर्यः उदेति"
# }


prompts = {
    "English": "<en>Once upon a time",
    "Hindi": "<hi>एक समय की बात है",
    "Sanskrit": "<sa>कदाचित्कालः आसीत्"
}


for lang, prompt in prompts.items():
    print(f"\n--- Generating for {lang} ---")
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

    # --- 5. Clean and Print the Generated Text ---
    print("\nCleaned Generated Text:")
    for output in generated_outputs:
        raw_text = output['generated_text']
        cleaned_text = clean_generated_text(raw_text, lang)
        print(cleaned_text)