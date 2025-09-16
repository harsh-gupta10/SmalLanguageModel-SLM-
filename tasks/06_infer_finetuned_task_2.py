import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
# 1. Path to the original base model
BASE_MODEL_PATH = 'model/checkpoints/pretrained/checkpoint-24000'

# 2. Path to the trained LoRA adapters for your noun-verb swap task
ADAPTER_PATH = 'model/checkpoints/finetuned/task_2/'

# The tokenizer is also loaded from the adapter path for consistency
TOKENIZER_PATH = ADAPTER_PATH

# --- Prompt Template (MUST match the one used in training) ---
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def load_model_for_inference(base_model_path, adapter_path, tokenizer_path):
    """
    Loads the base model, applies the LoRA adapters, and prepares it for inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the base model
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16, # Use float16 for faster inference if your GPU supports it
        device_map="auto"
    )

    # Load the tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the PEFT model (LoRA adapters) and merge it with the base model
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # The model is already on the device due to device_map="auto"
    model.eval() # Set the model to evaluation mode
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer, device

def generate_swap_response(model, tokenizer, instruction, input_text):
    """
    Formats the prompt, generates a response from the model, and parses the output.
    """
    # 1. Format the prompt
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)
    
    # 2. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 3. Generate the output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 4. Decode and parse the response
    # The output contains the original prompt, so we need to slice it off
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start_index = decoded_output.find("### Response:") + len("### Response:")
    parsed_response = decoded_output[response_start_index:].strip()
    
    return parsed_response


if __name__ == '__main__':
    # Load the fine-tuned model and tokenizer
    model, tokenizer, device = load_model_for_inference(
        BASE_MODEL_PATH, ADAPTER_PATH, TOKENIZER_PATH
    )

    # --- Test with some examples ---
    print("\n" + "="*50)
    print("Running Inference Examples...")
    print("="*50 + "\n")

    # Example 1: English
    english_instruction = "Identify and swap nouns with verbs, and verbs with nouns, in the given sentence."
    english_input = "The quick brown fox jumps over the lazy dog."
    
    print(f"Instruction: {english_instruction}")
    print(f"Input: {english_input}")
    
    english_output = generate_swap_response(model, tokenizer, english_instruction, english_input)
    print(f"Model Output: {english_output}\n")
    print("-"*50)

    # Example 2: Hindi
    hindi_instruction = "दिए गए वाक्य में संज्ञा को क्रिया से और क्रिया को संज्ञा से पहचानें और बदलें।"
    hindi_input = "राजा ने लाओ-त्ज़ु के बताये अनुसार प्रशासन में सुधार किये और धीरे-धीरे उसका राज्य आदर्श राज्य बन गया."
    
    print(f"Instruction: {hindi_instruction}")
    print(f"Input: {hindi_input}")
    
    hindi_output = generate_swap_response(model, tokenizer, hindi_instruction, hindi_input)
    print(f"Model Output: {hindi_output}\n")
    print("-"*50)
    
    # Example 3: Another English example from your dataset
    english_input_2 = "His name, he replied, was Willoughby, and his present home was at Allenham."
    
    print(f"Instruction: {english_instruction}")
    print(f"Input: {english_input_2}")
    
    english_output_2 = generate_swap_response(model, tokenizer, english_instruction, english_input_2)
    print(f"Model Output: {english_output_2}\n")
    print("="*50)