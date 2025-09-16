import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = 'model/checkpoints/pretrained/checkpoint-24000'

ADAPTER_PATH = 'model/checkpoints/finetuned/task_2/'

TOKENIZER_PATH = ADAPTER_PATH

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def load_model_for_inference(base_model_path, adapter_path, tokenizer_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    model.eval()
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer, device

def generate_swap_response(model, tokenizer, instruction, input_text):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start_index = decoded_output.find("### Response:") + len("### Response:")
    parsed_response = decoded_output[response_start_index:].strip()
    
    return parsed_response


if __name__ == '__main__':
    model, tokenizer, device = load_model_for_inference(
        BASE_MODEL_PATH, ADAPTER_PATH, TOKENIZER_PATH
    )

    print("\n" + "="*50)
    print("Running Inference Examples...")
    print("="*50 + "\n")

    english_instruction = "Identify and swap nouns with verbs, and verbs with nouns, in the given sentence."
    english_input = "The quick brown fox jumps over the lazy dog."
    
    print(f"Instruction: {english_instruction}")
    print(f"Input: {english_input}")
    
    english_output = generate_swap_response(model, tokenizer, english_instruction, english_input)
    print(f"Model Output: {english_output}\n")
    print("-"*50)

    hindi_instruction = "दिए गए वाक्य में संज्ञा को क्रिया से और क्रिया को संज्ञा से पहचानें और बदलें।"
    hindi_input = "राजा ने लाओ-त्ज़ु के बताये अनुसार प्रशासन में सुधार किये और धीरे-धीरे उसका राज्य आदर्श राज्य बन गया."
    
    print(f"Instruction: {hindi_instruction}")
    print(f"Input: {hindi_input}")
    
    hindi_output = generate_swap_response(model, tokenizer, hindi_instruction, hindi_input)
    print(f"Model Output: {hindi_output}\n")
    print("-"*50)
    
    english_input_2 = "His name, he replied, was Willoughby, and his present home was at Allenham."
    
    print(f"Instruction: {english_instruction}")
    print(f"Input: {english_input_2}")
    
    english_output_2 = generate_swap_response(model, tokenizer, english_instruction, english_input_2)
    print(f"Model Output: {english_output_2}\n")
    print("="*50)