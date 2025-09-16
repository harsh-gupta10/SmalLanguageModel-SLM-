import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
# Path to the original, non-fine-tuned base model
BASE_MODEL_PATH = 'model/checkpoints/pretrained/checkpoint-24000' 
# Path to the directory where your LoRA adapters are saved
PEFT_MODEL_PATH = 'model/checkpoints/finetuned'

# --- Prompt Template (MUST match the one used in training) ---
PROMPT_TEMPLATE = """### Instruction:
Extract a standalone fact from the following sentence.

### Input:
{sentence}

### Response:
"""

def extract_fact(model, tokenizer, sentence: str, device: str):
    """
    This function takes a sentence, formats it with the prompt template,
    and returns the fact extracted by the model.
    """
    # 1. Format the prompt
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    
    # 2. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    # 3. Generate a response from the model
    model.eval() 
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            do_sample=True
        )

    # 4. Decode ONLY the newly generated tokens
    # This is the key change to fix the repeated prompt issue.
    # We slice the output tensor to exclude the original input tokens.
    input_length = input_ids.shape[1]
    generated_token_ids = outputs[0][input_length:]
    
    # 5. Decode the generated tokens into text
    clean_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    
    return clean_response


def main():
    """
    Main function to load the model and run inference on example sentences.
    """
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Base Model and Tokenizer ---
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    
    # Load the tokenizer from the LoRA adapter directory for consistency
    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_PATH) 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id

    # --- Load and Apply LoRA Adapters ---
    print(f"Loading LoRA adapters from: {PEFT_MODEL_PATH}")
    # This automatically merges the LoRA weights with the base model for inference
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
    model = model.to(device)
    
    print("\nModel ready for inference!")

    # --- Example Usage ---
    
    # English example
    # english_sentence = "The sun, a star at the center of the Solar System, is a nearly perfect sphere of hot plasma."
    english_sentence = "She enjoys painting landscapes in her free time"
    extracted_english_fact = extract_fact(model, tokenizer, english_sentence, device)
    print("-" * 20)
    print(f"Original English Sentence:\n{english_sentence}")
    print(f"\nExtracted Fact:\n{extracted_english_fact}")
    print("-" * 20)
    
    # Hindi example
    hindi_sentence = "ताजमहल, जो भारत के आगरा शहर में स्थित एक हाथीदांत-सफेद संगमरमर का मकबरा है, को मुगल सम्राट शाहजहाँ ने अपनी पसंदीदा पत्नी मुमताज महल की याद में बनवाया था।"
    extracted_hindi_fact = extract_fact(model, tokenizer, hindi_sentence, device)
    print("-" * 20)
    print(f"Original Hindi Sentence:\n{hindi_sentence}")
    print(f"\nExtracted Fact:\n{extracted_hindi_fact}")
    print("-" * 20)

    # Your own sentence
    custom_sentence = "Python, a high-level, general-purpose programming language, was created by Guido van Rossum and first released in 1991."
    extracted_custom_fact = extract_fact(model, tokenizer, custom_sentence, device)
    print("-" * 20)
    print(f"Original Custom Sentence:\n{custom_sentence}")
    print(f"\nExtracted Fact:\n{extracted_custom_fact}")
    print("-" * 20)


if __name__ == '__main__':
    main()