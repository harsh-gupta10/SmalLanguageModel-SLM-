import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "model" / "checkpoints" / "pretrained" / "checkpoint-24000"

TOKENIZER_DIR = PROJECT_ROOT / "model" / "tokenizer"

print(f"Loading model from: {MODEL_DIR}")
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}. Please check the path.")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto"
)
print("Model loaded successfully.")

print(f"Loading tokenizer from: {TOKENIZER_DIR}")
if not TOKENIZER_DIR.exists():
    raise FileNotFoundError(f"Tokenizer directory not found at {TOKENIZER_DIR}. Please check the path.")

tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")


def clean_generated_text(text: str, language: str) -> str:
    text = re.sub(r'<(en|hi|sa)>', '', text)

    if language == "English":
        text = re.sub(r'[\u0900-\u097F]+', '', text)
    elif language in ["Hindi", "Sanskrit"]:
        text = re.sub(r'[a-zA-Z]+', '', text)

    text = re.sub(r'[\s.,\'"-?!’]+$', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

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

    print("\nCleaned Generated Text:")
    for output in generated_outputs:
        raw_text = output['generated_text']
        cleaned_text = clean_generated_text(raw_text, lang)
        print(cleaned_text)