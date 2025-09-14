import torch
from transformers import AutoConfig, AutoModelForCausalLM
import sentencepiece as spm
import os

# --- Configuration ---
STUDENT_MODEL_PATH = 'model/student_trained/' # Path where your trained student model is saved
TOKENIZER_MODEL_PATH = 'model/tokenizer/multilingual_spm.model'
MAX_INFERENCE_SEQUENCE_LENGTH = 512 # Max length for input + generated tokens
NUM_GENERATED_TOKENS = 50 # How many new tokens to generate

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")

    # Load tokenizer
    print("Loading custom tokenizer...")
    sp = spm.SentencePieceProcessor()
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        print(f"Error: Tokenizer model not found at {TOKENIZER_MODEL_PATH}")
        return
    sp.load(TOKENIZER_MODEL_PATH)

    # Load student model
    print(f"Loading trained student model from {STUDENT_MODEL_PATH}...")
    try:
        student_config = AutoConfig.from_pretrained(STUDENT_MODEL_PATH)
        student_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_PATH,
            config=student_config,
            torch_dtype=student_config.torch_dtype # Use the dtype specified in its config
        )
        student_model.eval() # Set model to evaluation mode
        student_model.to(device)
        print(f"Student model loaded successfully. Model parameters: {student_model.num_parameters() / 1e6:.2f} M")
    except Exception as e:
        print(f"Error loading student model from {STUDENT_MODEL_PATH}: {e}")
        print("Please ensure the trained student model files (config.json, model.safetensors etc.) are in the specified path.")
        return

    print("\n--- Model Inference (Prefix your prompt with <en>, <hi>, or <sa>) ---")
    while True:
        prompt = input("Enter a prompt (e.g., '<en> hello how are you', or 'quit'): ")
        if prompt.lower() == 'quit':
            break


        # Tokenize the input prompt
        input_ids = sp.encode(prompt)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Generate text
        print("Generating response...")
        with torch.no_grad():
            output_tokens = student_model.generate(
                input_ids_tensor,
                max_length=len(input_ids) + NUM_GENERATED_TOKENS,
                num_return_sequences=1,
                pad_token_id=sp.pad_id(), # Use your tokenizer's pad_id
                do_sample=True, # Enable sampling for more creative output
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                # eos_token_id=sp.eos_id() # Uncomment if you want generation to stop at EOS token
            )

        # Decode the generated tokens
        generated_text = sp.decode(output_tokens[0].tolist())
    
        # Adjust for language tokens potentially being repeated or present in output
        # A more robust way might be to remove the first occurrence of the prompt tokens
        # and then clean up any remaining language tokens if not desired in raw output.
        # For now, a simple heuristic:
        if generated_text.startswith(prompt):
            generated_response = generated_text[len(prompt):].strip()
        else:
            generated_response = generated_text # Fallback
        
        # Clean up any residual language tokens if they appear i
        # n the *generated* part
        # (e.g. if the model hallucinates another <en> mid-sentence, which can happen)
        for lang_tok in ['<en>', '<hi>', '<sa>']:
            generated_response = generated_response.replace(lang_tok, '').strip()


        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_response}\n")


if __name__ == '__main__':
    run_inference()