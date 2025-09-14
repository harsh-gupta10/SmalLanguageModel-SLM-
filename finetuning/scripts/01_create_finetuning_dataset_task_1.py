import json
import os
import sys
import textwrap # For better display of text

# Use the 'google.genai' library as per your example
from google import genai
from google.genai import types # Import types for Content, Part, etc.

# --- Configuration ---
OUTPUT_FILE = "gemini_decontextualization_dataset.jsonl"
MODEL_NAME = "gemini-2.5-flash" # As in your example, but this name is often internal or future.
# Note: If you specifically want to use "gemini-pro" as your original error suggested,
# you might need to check your project's region/permissions or use 'google.generativeai'.
# However, this script is tailored to the `google.genai` client shown in your example.

# --- Gemini API Configuration ---
def get_gemini_client() -> genai.Client:
    """Configures the Gemini API client and returns it."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not found.", file=sys.stderr)
        api_key = input("Please enter your Gemini API key: ").strip()
        if not api_key:
            print("Gemini API key is required. Exiting.", file=sys.stderr)
            sys.exit(1)
    
    try:
        client = genai.Client(api_key=api_key)
        # Optional: Test a small generation to ensure API is working
        # test_content = [types.Content(role="user", parts=[types.Part.from_text(text="hello")])]
        # client.models.generate_content(model=MODEL_NAME, contents=test_content)
        return client
    except Exception as e:
        print(f"Error initializing Gemini client. Check your API key and network connection: {e}", file=sys.stderr)
        sys.exit(1)

# --- Gemini Interaction Functions ---
def generate_initial_sentences_with_gemini(client: genai.Client, topic: str, num_sentences: int = 10) -> list[str]:
    """
    Uses Gemini to generate a list of contextualized input sentences for the task.
    """
    prompt = textwrap.dedent(f"""
    Generate {num_sentences} diverse, complex, and contextualized English sentences that contain explicit or implicit facts. These sentences should be suitable for a 'fact decontextualization' task where a standalone, atomic fact needs to be extracted. Focus on the topic of "{topic}".

    Present each sentence on a new line, without numbering or introductory text. Ensure the sentences are varied in structure and contain clear factual information.

    Example:
    CEO John Smith announced record profits at the meeting.
    Despite the rain, the marathon continued, and she finished first.
    The company's new product, launched last week, has already garnered significant attention.
    The scientist discovered a new element during her experiment yesterday.
    The ancient city, founded in 300 BC, is a popular tourist destination.
    """)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    print(f"\n--- Generating {num_sentences} input sentences on '{topic}' using Gemini ({MODEL_NAME})... ---")
    try:
        # Using generate_content for a single, complete response rather than streaming
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        sentences_raw = response.text.strip()
        sentences = [s.strip() for s in sentences_raw.split('\n') if s.strip()]
        # Gemini might generate more or less, try to get closer to num_sentences
        return sentences[:num_sentences] if len(sentences) > num_sentences else sentences
    except Exception as e:
        print(f"Error generating initial sentences with Gemini: {e}", file=sys.stderr)
        return []

def generate_fact_with_gemini(client: genai.Client, sentence: str) -> str:
    """
    Uses Gemini to suggest a decontextualized fact for a given sentence.
    Includes few-shot examples for better guidance.
    """
    prompt = textwrap.dedent(f"""
    From the following sentence, extract a single, standalone, atomic fact. Resolve any pronouns and make sure the fact is understandable on its own. Ensure the fact is concise and directly stated or clearly implied.

    Sentence: "CEO John Smith announced record profits at the meeting."
    Fact: John Smith is a CEO.

    Sentence: "Despite the rain, the marathon continued, and she finished first."
    Fact: The marathon continued.

    Sentence: "The company's new product, launched last week, has already garnered significant attention."
    Fact: The company launched a new product.

    Sentence: "The ancient city, founded in 300 BC, is a popular tourist destination."
    Fact: The ancient city was founded in 300 BC.

    Sentence: "{sentence}"
    Fact:
    """)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        # Gemini might return "Fact: [fact]" or just "[fact]"
        generated_text = response.text.strip()
        if generated_text.lower().startswith("fact:"):
            return generated_text[len("fact:"):].strip()
        return generated_text.split('\n')[0].strip() # Take the first line if multiple are generated
    except Exception as e:
        print(f"Error calling Gemini for decontextualization: {e}", file=sys.stderr)
        return ""

# --- Main Script Logic ---
def generate_dataset_with_gemini_and_review(client: genai.Client, topic: str, num_sentences: int, output_file: str):
    """
    Generates the finetuning dataset using Gemini for both input generation
    and decontextualization, with manual review.
    """
    dataset = []
    print(f"\n--- Starting dataset generation for topic '{topic}' ---")

    input_sentences = generate_initial_sentences_with_gemini(client, topic, num_sentences)
    if not input_sentences:
        print("Could not generate initial sentences. Exiting.", file=sys.stderr)
        return

    print(f"\n--- Reviewing and decontextualizing {len(input_sentences)} sentences ---")

    for i, sentence in enumerate(input_sentences):
        print(f"\n--- Processing Sentence {i + 1}/{len(input_sentences)} ---")
        print(f"Original: {sentence}")

        suggested_fact = generate_fact_with_gemini(client, sentence)
        print(f"Gemini's Suggested Fact: {suggested_fact}")

        decontextualized_fact = ""
        # review_choice = input("Accept (Y/n/edit)? [Y]: ").strip().lower()
        review_choice = 'Y'

        if review_choice == 'n':
            while not decontextualized_fact:
                decontextualized_fact = input("Enter corrected fact: ").strip()
                if not decontextualized_fact:
                    print("Fact cannot be empty. Please enter a fact.")
        elif review_choice == 'edit':
             decontextualized_fact = input(f"Edit fact [{suggested_fact}]: ").strip()
             if not decontextualized_fact: # If user just pressed enter, use suggested
                 decontextualized_fact = suggested_fact
        else: # Default to 'y' or empty input
            decontextualized_fact = suggested_fact

        dataset.append({
            "input": sentence,
            "output": decontextualized_fact
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nDataset generation complete. Saved {len(dataset)} entries to '{output_file}'")

if __name__ == "__main__":
    gemini_client = get_gemini_client()

    topic = input("Enter a topic for sentence generation (e.g., 'technology', 'historical events', 'science'): ").strip()
    if not topic:
        print("Topic cannot be empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        num_sentences = int(input("How many sentences to generate for this dataset (e.g., 20)? "))
        if num_sentences <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive number for sentences. Exiting.", file=sys.stderr)
        sys.exit(1)

    generate_dataset_with_gemini_and_review(gemini_client, topic, num_sentences, OUTPUT_FILE)