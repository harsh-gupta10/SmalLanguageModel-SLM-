import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_PATH = 'model/checkpoints/pretrained/checkpoint-24000' 
PEFT_MODEL_PATH = 'model/checkpoints/finetuned/task_1/'

PROMPT_TEMPLATE = """### Instruction:
Extract a standalone fact from the following sentence.

### Input:
{sentence}

### Response:
"""

def extract_fact(model, tokenizer, sentence: str, device: str):
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
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

    input_length = input_ids.shape[1]
    generated_token_ids = outputs[0][input_length:]
    
    clean_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    
    return clean_response


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_PATH) 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = base_model.config.eos_token_id

    print(f"Loading LoRA adapters from: {PEFT_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
    model = model.to(device)
    
    print("\nModel ready for inference!")

    english_sentence = "She enjoys painting landscapes in her free time"
    extracted_english_fact = extract_fact(model, tokenizer, english_sentence, device)
    print("-" * 20)
    print(f"Original English Sentence:\n{english_sentence}")
    print(f"\nExtracted Fact:\n{extracted_english_fact}")
    print("-" * 20)
    
    hindi_sentence = "ताजमहल, जो भारत के आगरा शहर में स्थित एक हाथीदांत-सफेद संगमरमर का मकबरा है, को मुगल सम्राट शाहजहाँ ने अपनी पसंदीदा पत्नी मुमताज महल की याद में बनवाया था।"
    extracted_hindi_fact = extract_fact(model, tokenizer, hindi_sentence, device)
    print("-" * 20)
    print(f"Original Hindi Sentence:\n{hindi_sentence}")
    print(f"\nExtracted Fact:\n{extracted_hindi_fact}")
    print("-" * 20)

    custom_sentence = "The mall, which opened in 2005, is the largest shopping center in the city."
    extracted_custom_fact = extract_fact(model, tokenizer, custom_sentence, device)
    print("-" * 20)
    print(f"Original Custom Sentence:\n{custom_sentence}")
    print(f"\nExtracted Fact:\n{extracted_custom_fact}")
    print("-" * 20)


if __name__ == '__main__':
    main()