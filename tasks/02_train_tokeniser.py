# 02_train_tokeniser.py (Optimized for Multi-core CPU)

import os
import sentencepiece as spm

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
    vocab_size = 320003
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
        print(f"Model saved to: {model_prefix}.model")
        print(f"Vocabulary saved to: {model_prefix}.vocab")
        
        print("\n--- Testing the trained tokenizer ---")
        sp = spm.SentencePieceProcessor()
        sp.load(f'{model_prefix}.model')

        test_sentences = [
            "This is a test sentence in English.",
            "यह हिंदी में एक परीक्षण वाक्य है।",
            "एषः संस्कृतस्य परीक्षणवाक्यम् अस्ति।"
        ]
        
        for sentence in test_sentences:
            encoded = sp.encode_as_pieces(sentence)
            print(f"Original: {sentence}")
            print(f"Tokenized: {encoded}\n")

    except Exception as e:
        print(f"An error occurred during tokenizer training: {e}")

if __name__ == '__main__':
    train_sentencepiece_tokenizer()