# 02_train_tokeniser.py (Optimized for Multi-core CPU)

import os
import sentencepiece as spm
from transformers import LlamaTokenizer # Import LlamaTokenizer from transformers

def train_sentencepiece_tokenizer():
    print("Starting tokenizer training...")

    num_threads = os.cpu_count()
    print(f"Detected {num_threads} CPU cores. Will use all of them for training.")

    raw_data_dir = 'data/raw'
    model_dir = 'model/tokenizer'
    
    input_files = [
        os.path.join(raw_data_dir, 'lang_english.txt'),
        os.path.join(raw_data_dir, 'lang_hindi.txt'),
        os.path.join(raw_data_dir, 'lang_sanskrit.txt')
    ]

    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: Input file not found at {f}")
            print("Please ensure your raw data files are in the 'data/raw' directory.")
            return

    os.makedirs(model_dir, exist_ok=True)

    model_prefix = os.path.join(model_dir, 'multilingual_spm')
    vocab_size = 32003
    character_coverage = 1.0
    model_type = 'bpe'
    
    input_sentence_size = 10000000  # Sample 10 million sentences
    shuffle_input_sentence = 'true' # Enable random sampling

    spm_command = (
        f'--input={",".join(input_files)} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={character_coverage} '
        f'--model_type={model_type} '
        f'--input_sentence_size={input_sentence_size} '
        f'--shuffle_input_sentence={shuffle_input_sentence} '
        f'--num_threads={num_threads} '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        '--user_defined_symbols=<en>,<hi>,<sa>'
    )

    try:
        print("\nStarting SentencePiece training process. This may take a while...")
        spm.SentencePieceTrainer.train(spm_command)

        print("\nTokenizer training completed successfully!")
        print(f"SentencePiece model saved to: {model_prefix}.model")
        print(f"SentencePiece vocabulary saved to: {model_prefix}.vocab")
        
        print("\n--- Converting SentencePiece model to Hugging Face format ---")
        # Load the trained SentencePiece model using SentencePieceProcessor for ID mapping
        sp = spm.SentencePieceProcessor()
        sp.load(f'{model_prefix}.model')

        # Initialize LlamaTokenizer directly from the SentencePiece model file
        hf_tokenizer = LlamaTokenizer(vocab_file=f'{model_prefix}.model')
        
        # Set the special tokens for the Hugging Face tokenizer based on SentencePiece's configuration
        # This ensures consistency and proper handling by Hugging Face models.
        hf_tokenizer.pad_token = sp.id_to_piece(sp.pad_id())
        hf_tokenizer.unk_token = sp.id_to_piece(sp.unk_id())
        hf_tokenizer.bos_token = sp.id_to_piece(sp.bos_id())
        hf_tokenizer.eos_token = sp.id_to_piece(sp.eos_id())
        
        # Save the Hugging Face compatible tokenizer
        # This will create tokenizer.json, special_tokens_map.json, and tokenizer_config.json
        # in the specified model_dir, alongside the .model and .vocab files.
        hf_tokenizer.save_pretrained(model_dir)
        print(f"Hugging Face tokenizer saved to: {model_dir}")
        print("This directory now contains tokenizer.json, special_tokens_map.json, etc.,")
        print("allowing direct loading with AutoTokenizer.from_pretrained.")

        print("\n--- Testing the trained tokenizer ---")
        # Use the newly saved Hugging Face tokenizer for testing
        test_tokenizer = LlamaTokenizer.from_pretrained(model_dir) 

        test_sentences = [
            "This is a test sentence in English.",
            "यह हिंदी में एक परीक्षण वाक्य है।",
            "एषः संस्कृतस्य परीक्षणवाक्यम् अस्ति।"
        ]
        
        for sentence in test_sentences:
            encoded = test_tokenizer.encode_as_pieces(sentence) # Use encode_as_pieces for SentencePiece like output
            # Alternatively, for standard HF tokenization:
            # encoded_ids = test_tokenizer.encode(sentence)
            # decoded_text = test_tokenizer.decode(encoded_ids)
            print(f"Original: {sentence}")
            print(f"Tokenized: {encoded}\n")

    except Exception as e:
        print(f"An error occurred during tokenizer training: {e}")

if __name__ == '__main__':
    train_sentencepiece_tokenizer()