# pip install google-genai tqdm

import os
import json
import random
from tqdm import tqdm
from google import genai
from google.genai import types

# ======================
# CONFIGURATION
# ======================
MODEL = "gemini-2.5-flash"
N_SAMPLES = 50_000
BATCH_SIZE = 20  # sentences per request
OUTPUT_EN = "dataset_english.jsonl"
OUTPUT_HI = "dataset_hindi.jsonl"

# ======================
# GEMINI CLIENT
# ======================
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def build_prompt(language: str, sentences: list[str]) -> str:
    """Create a prompt for Gemini to replace nouns with verbs."""
    task = (
        f"Task: Replace all nouns in the given {language} sentences with verbs. "
        f"If no noun exists, return the same sentence. "
        f"Return output in strict JSON with a list of objects, each object having "
        f"'input' and 'output'."
    )

    examples = "\n".join([f"- {s}" for s in sentences])
    return f"{task}\n\nSentences:\n{examples}"

def call_gemini(prompt: str):
    """Call Gemini and return text output."""
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1)
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text if chunk.text else ""

    return response_text

def generate_dataset(language: str, output_file: str):
    """Generate dataset for given language and save to JSONL."""
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in tqdm(range(N_SAMPLES // BATCH_SIZE), desc=f"Generating {language}"):
            # Make random seed sentences (you can plug in Wikipedia/news corpora here)
            # For now, using simple synthetic seed sentences
            seed_sentences = [
                f"This is a sample sentence number {random.randint(1, 100000)}"
                if language == "english"
                else f"यह एक उदाहरण वाक्य संख्या {random.randint(1, 100000)} है"
                for _ in range(BATCH_SIZE)
            ]

            prompt = build_prompt(language, seed_sentences)
            try:
                response = call_gemini(prompt)
                # Parse JSON safely
                data = json.loads(response)
                for pair in data:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            except Exception as e:
                print("Error parsing response, skipping batch:", e)

if __name__ == "__main__":
    generate_dataset("english", OUTPUT_EN)
    generate_dataset("hindi", OUTPUT_HI)
