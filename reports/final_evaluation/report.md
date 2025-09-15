# Final Report

## Language, Model and Agents Mini-Project

`Name:` Ansh Chablani <br>
`Roll No.:` 2022111031

## Specific Information

**Languages:**
1. `English`
2. `Hindi`
3. `Sanskrit`

**Model:** `Qwen3-Dense`

**Tasks:**

1. Fact Decontextualization (Knowledge & Commonsense Reasoning)
2. Identify and replace nouns with verbs from given sentence (Sentence Structure & Syntax)

## Dataset Collection

I had to extract the data for the 3 languages English, Hindi and Sanskrit

Fortunately, I was able to get the data for these three languages because of their popularity.

**Sources:**

1. English -> [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
2. Hindi -> [Fineweb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
3. Sanskrit -> [AI4Bharat-Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/san)

These three data sources were enough to provide sufficient sources.

The approximate token counts for the above languages are:

> `!Fill This`

The file sizes of all the languages are:

1. English: `!Fill This`
2. Hindi: `!Fill This`
3. Sanskrit: `!Fill This`

The file for dataset gathering is: [tasks/00_get_data.py](tasks/00_get_data.py).


## Preprocessing

I applied several preprocessing steps to enhance the quality of my data.

The raw text data was processed using a comprehensive pipeline (`tasks/01_preprocess_data.py`) to ensure quality and consistency. 

The key steps were:

1. **Text Cleaning:** Each line of text from all language files was individually cleaned. This involved:  
   * Using the `ftfy` library to fix potential Unicode errors and inconsistencies (mojibake).  
   * Normalizing all whitespace (tabs, newlines, multiple spaces) into a single space to create uniform sentences.  

2. **Parallel Processing:** To handle the large volume of data efficiently, the cleaning process was parallelized to utilize all available CPU cores, significantly reducing the processing time.  

3. **Global Deduplication:** To prevent the model from overfitting on repeated sentences, a two-step deduplication was performed. First, duplicates were removed within each language file. Then, all unique lines were combined into a single master set, which removed any duplicates that existed *between* the language files.  

4. **Data Splitting:** The final, clean, and globally unique dataset was shuffled thoroughly to ensure a random distribution of languages. It was then split into three sets for the model lifecycle:  
   * **Training Set:** 98%  
   * **Validation Set:** 1%  
   * **Test Set:** 1%  

5. **Script and Language Filtering:** Each line underwent an additional layer of filtering to ensure that the text matched the intended script and language. This involved:  
   * **Regex-based script detection:** Lines were quickly pre-filtered using Unicode ranges — Latin characters for English, Devanagari for Hindi and Sanskrit.  
   * **Language identification:** The `langdetect` library was applied to verify that the cleaned line belonged to the expected language (e.g., `en` for English, `hi` for Hindi, `sa` or `hi` for Sanskrit, accounting for common misclassifications).  
   * **Noise reduction:** Lines with too many symbols or mixed scripts were discarded.  

This process resulted in clean, non-redundant, and well-structured datasets saved in the `data/processed/` directory, ready for model training.

## Tokenisation

I trained a **SentencePiece** tokenizer optimized for handling English, Hindi, and Sanskrit text.

The key steps were:

1. **Multilingual Training Data:** The raw text files for English, Hindi, and Sanskrit were used together to train a shared tokenizer. This ensured that all three languages could be encoded consistently with a single vocabulary.

2. **SentencePiece with BPE:** I used the [SentencePiece](https://github.com/google/sentencepiece) library with the **Byte Pair Encoding (BPE)** algorithm.  
   * Vocabulary size: **320,003 tokens**  
   * Character coverage: **100%**, ensuring that all Unicode characters from the training set were represented.  
   * Special tokens were explicitly reserved:  
     * `<pad>` → ID 0  
     * `<unk>` → ID 1  
     * `<bos>` → ID 2  
     * `<eos>` → ID 3  
   * **Custom user-defined symbols were added for each language tag: `<en>`, `<hi>`, `<sa>`.**

3. **Efficient Training:**  
   * Leveraged all available CPU cores (`num_threads = os.cpu_count()`) for parallel training.  
   * Sampled up to **10 million sentences** with shuffling to ensure balanced representation of each language.  

4. **Integration with Hugging Face:**  
   * After training, the `.model` and `.vocab` files from SentencePiece were converted into a **Hugging Face–compatible tokenizer** (`LlamaTokenizer`).  
   * The tokenizer was saved in `model/tokenizer/`, producing the standard Hugging Face files:  
     * `tokenizer.json`  
     * `tokenizer_config.json`  
     * `special_tokens_map.json`  

5. **Testing:**  
   The trained tokenizer was tested on sample sentences in all three languages to verify correct tokenization:  
   * English: `"This is a test sentence in English."`  
   * Hindi: `"यह हिंदी में एक परीक्षण वाक्य है।"`  
   * Sanskrit: `"एषः संस्कृतस्य परीक्षणवाक्यम् अस्ति।"`  

---

### Why `<en>`, `<hi>`, `<sa>`?  

Instead of simply mixing English, Hindi, and Sanskrit lines into the dataset, I introduced **explicit language tags** (`<en>`, `<hi>`, `<sa>`) at the tokenizer level.  

* **Disambiguation:** These tags give the model a clear signal about the language of the current input. Without tags, the model might confuse closely related scripts (e.g., Hindi vs Sanskrit, which are often misclassified).  
* **Better Control:** During inference, I can *force* the model to generate text in a specific language by prefixing the input with `<en>`, `<hi>`, or `<sa>`.  
* **Multilingual Generalization:** Language tags have been shown in multilingual models (e.g., mBART, XLM-R) to improve cross-lingual alignment and reduce code-switching errors.  
* **Innovation vs Naive Mixing:** A naive approach would just concatenate all multilingual lines and let the model figure out the language implicitly. By explicitly marking the language at the tokenizer level, I give the model structured guidance, which improves both training efficiency and generation quality.  

This design choice is an **innovation** in my pipeline: instead of treating multilingual data as an unstructured mixture, I explicitly encode language identity as a token. This not only improves training stability but also provides controllability during downstream tasks (translation, classification, or generation).

## Pretraining

For pretraining my multilingual model, I explored **two different strategies**:

---

### 1. Simple Training (Baseline)

The first approach was a **straightforward pretraining pipeline** using the Hugging Face `Trainer` API:  

* **Architecture:** A custom Qwen3-based model (~125M parameters) configured from scratch.  
* **Tokenizer:** The multilingual SentencePiece tokenizer (`<en>`, `<hi>`, `<sa>` support) trained earlier.  
* **Dataset:** The cleaned and deduplicated text corpus across English, Hindi, and Sanskrit.  
* **Training setup:**  
  * Maximum sequence length: 2048 tokens  
  * Training epochs: 3  
  * Cosine learning rate schedule with warmup  
  * Gradient checkpointing for memory efficiency  
  * Mixed precision (`fp16` or `bf16`) to speed up training  

This method directly optimized the model on the raw dataset using cross-entropy loss. It produced a **baseline multilingual model** that was consistent but relatively large and resource-intensive.

---

### 2. Knowledge Distillation (Student–Teacher Training)

To make the model lighter while retaining performance, I used **knowledge distillation**:  

* **Teacher Model:** A larger pretrained checkpoint (producing logits for each token).  
* **Student Model:** A smaller model initialized with the same architecture family, but with fewer parameters.  
* **Distillation Loss:** A weighted combination of:  
  * **Hard targets** → Cross-entropy with ground-truth labels.  
  * **Soft targets** → KL-divergence between teacher and student probability distributions, scaled by a temperature parameter.  
* **Chunked Training:** Teacher logits were precomputed and stored in chunks (`distillation_teacher_logits_chunk_*.pt`). The student model iterated over these chunks efficiently without re-running the teacher forward pass.  
* **Checkpointing:** Student checkpoints were saved regularly to support resumption and avoid loss of progress.  

---

### Results from Distillation

The **distilled student model** achieved:  

* **Smaller size:** Significantly fewer parameters compared to the baseline model, reducing memory footprint and inference cost.  
* **Faster inference:** Improved latency, making the model more practical for deployment on resource-constrained devices.  
* **Comparable quality:** Retained most of the language understanding and generation capabilities of the baseline model, thanks to the teacher’s soft supervision.  
* **Better stability:** The combination of hard labels and softened teacher distributions improved convergence and reduced overfitting.  

---

**In summary:**  
I first pretrained a baseline multilingual Qwen3 model from scratch for performance benchmarking. Then, I trained a distilled student model that was smaller and faster but still competitive in quality. This two-step process provided both a **high-quality reference model** (teacher) and a **lightweight deployable model** (student).

## Finetuning

### Dataset Creation - Task 1

## Fine-tuning Dataset Creation

To fine-tune the model on a **fact decontextualization** task, I created a custom dataset using a large language model (LLM).  
The goal was to train the model to convert a **contextualized sentence** into a **standalone atomic fact**.

---

### Dataset Generation Pipeline

1. **Sentence Generation:**  
   Using Gemini (`gemini-2.5-flash`), I prompted the model to generate diverse, contextualized sentences containing explicit or implicit factual content.  
   * Example prompt: *"Generate 10 diverse, complex, and contextualized English sentences that contain explicit or implicit facts..."*  
   * This was repeated separately for **English** and **Hindi** datasets.

2. **Fact Extraction:**  
   For each sentence, I asked Gemini to extract a single **decontextualized fact** that:  
   * Resolves pronouns.  
   * Can stand alone without additional context.  
   * Is concise and factual.  

   Few-shot examples were included in the prompt to guide Gemini’s outputs.

3. **Human-in-the-loop Review:**  
   Each Gemini-suggested fact was reviewed. The script allowed three options:  
   * **Accept** → keep Gemini’s suggestion.  
   * **Edit** → modify Gemini’s fact.  
   * **Reject** → manually enter a corrected fact.  
   For the initial dataset, I auto-accepted all suggestions (`review_choice = 'Y'`), but the script supports manual review for higher quality.

4. **JSONL Output:**  
   The final dataset was saved in `.jsonl` format, where each entry consists of:  
   ```json
   {
     "input": "CEO John Smith announced record profits at the meeting.",
     "output": "John Smith is a CEO."
   }

#### Why This Approach?

**Synthetic but diverse:** Leveraging an LLM allowed me to generate diverse, domain-rich sentences without needing a large annotated dataset.

**Decontextualization Task-Specific:** Instead of generic QA pairs, the dataset directly targeted the skill I wanted my model to learn: rewriting context-rich sentences into standalone factual statements.

**Bilingual Extension:** The same pipeline was applied to Hindi, ensuring that the fine-tuned model could handle decontextualization across multiple scripts and languages.