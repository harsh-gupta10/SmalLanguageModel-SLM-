import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_PATH = 'model/student_trained/'
ADAPTER_PATH = 'model/lora_decontextualizer_adapters/'
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load the base model and tokenizer ---
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) # Load tokenizer from the adapter dir

# --- Load the LoRA PEFT model ---
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval() # Set to evaluation mode

# --- Inference ---
sentence = "The Eiffel Tower, located in Paris, is a famous landmark."
# sentence = "पेरिस, जिसे 'रोशनी का शहर' भी कहते हैं, फ्रांस की राजधानी है।"

# Format the input using the same prompt template, but without the response
prompt = f"""### Instruction:
Extract a standalone fact from the following sentence.

### Input:
{sentence}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating fact...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# The response will contain the original prompt, so we extract just the generated part
generated_text = response.split("### Response:")[1].strip()

print(f"\nSentence: {sentence}")
print(f"Extracted Fact: {generated_text}")