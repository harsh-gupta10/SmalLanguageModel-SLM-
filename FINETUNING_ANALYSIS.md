# Finetuning Pipeline - Loss Function & Training Details

## Summary
This project uses **Cross-Entropy Loss (implicit in causal language modeling)**, NOT cosine similarity. The model is trained as a **causal language model** where weights are updated via **backpropagation and gradient descent using AdamW optimizer**.

---

## 1. LOSS FUNCTION

### **Cross-Entropy Loss (via Hugging Face `transformers`)**

The loss is **NOT computed manually**. Instead, it's automatically calculated by the model:

```python
outputs = model(**batch)  # batch contains input_ids, attention_mask, labels
loss = outputs.loss       # This is cross-entropy loss from the model
```

**Why Cross-Entropy?**
- The model is a **causal language model** (predicts next token given previous tokens)
- Each position in the sequence must predict the next token from a vocabulary of ~50K tokens
- Cross-entropy loss compares the predicted probability distribution with the ground truth (which token should actually come next)

### **Loss Calculation in Pretraining**
From [03_pretrain_model.py](tasks/03_pretrain_model.py):
```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# mlm=False means CAUSAL language modeling (not masked)
# The model internally uses cross-entropy loss on next-token prediction
```

---

## 2. HOW INSTRUCTION + INPUT ARE COMBINED AND FED TO MODEL

### **Prompt Template Structure**

#### **Task 1 (Fact Decontextualization)** - [05_finetune_model_task_1.py](tasks/05_finetune_model_task_1.py)

```python
PROMPT_TEMPLATE = """### Instruction:
Extract a standalone fact from the following sentence.

### Input:
{sentence}

### Response:
{fact}"""
```

**Example:**
```
### Instruction:
Extract a standalone fact from the following sentence.

### Input:
The Eiffel Tower is located in Paris, France.

### Response:
Eiffel Tower is a landmark<EOS>
```

#### **Task 2 (Noun-Verb Swap)** - [05_finetune_model_task_2.py](tasks/05_finetune_model_task_2.py)

```python
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

**Example:**
```
### Instruction:
Identify and swap nouns with verbs, and verbs with nouns, in the given sentence.

### Input:
The dog ran quickly.

### Response:
The run dogged quickly.<EOS>
```

---

## 3. HOW INSTRUCTION+INPUT+RESPONSE ARE TOKENIZED

### **Key Process: Masking the Prompt Part**

The critical step is to **only compute loss on the Response part**, not on Instruction+Input:

```python
def __getitem__(self, idx):
    item = self.data[idx]
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    # Full prompt with response
    full_prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output_text
    ) + self.tokenizer.eos_token
    
    tokenized_full = self.tokenizer(full_prompt, truncation=True, max_length=512, padding=False)

    # Prompt WITHOUT response (just instruction + input)
    prompt_without_response = PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=""  # Empty response section
    )
    tokenized_prompt_only = self.tokenizer(prompt_without_response, truncation=True, max_length=512, padding=False)
    
    # Get number of tokens in prompt-only
    prompt_len = len(tokenized_prompt_only['input_ids'])

    # Create labels: mask prompt tokens with -100 (will be ignored in loss)
    labels = list(tokenized_full['input_ids'])
    for i in range(prompt_len):
        labels[i] = -100  # Loss won't be computed on these positions
    
    return {
        "input_ids": torch.tensor(tokenized_full['input_ids']),
        "attention_mask": torch.tensor(tokenized_full['attention_mask']),
        "labels": torch.tensor(labels)  # Only response part contributes to loss
    }
```

### **Visual Representation:**

```
Tokenized Sequence:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
 └────────────────────────────┘  └──────────────┘
      Instruction + Input          Response (fact/output)
      (Prompt part)

Labels for Loss:
[-100, -100, -100, -100, -100, -100, -100, -100, 9, 10, 11, 12, 13, 14]
 └─────────────────────────────────────────────────────┘  └──────────────┘
                  Ignored (loss=0)                        Contributes to loss
```

**Why Mask the Prompt?**
- The model should learn to **generate the output** given instruction+input
- We don't want to waste training on predicting the instruction/input (which is already given)
- This focuses the training gradient on the actual task

---

## 4. HOW WEIGHTS ARE UPDATED

### **Training Loop** (both Task 1 & Task 2 use same approach):

```python
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass: model computes loss on masked positions
        outputs = model(**batch)
        loss = outputs.loss  # Cross-entropy loss (masked prompt positions ignored)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights via AdamW optimizer
        optimizer.step()
        
        # Update learning rate
        lr_scheduler.step()
        
        # Reset gradients for next iteration
        optimizer.zero_grad()
```

### **Optimizer: AdamW**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

**Parameters:**
- **Learning Rate:** `2e-4` (0.0002)
- **Scheduler:** Linear warmup (Task 1 & 2) or Cosine (Pretraining)
- **LoRA Configuration:**
  - **Rank (r):** 16
  - **Alpha:** 32
  - **Dropout:** 0.05
  - **Target Modules:** `["q_proj", "v_proj"]` (only attention layers updated)

---

## 5. EXPECTED OUTPUT vs REAL OUTPUT COMPARISON

### **The Comparison Happens Implicitly in Cross-Entropy Loss**

The model doesn't have separate "expected" and "real" outputs. Instead:

1. **Model Output:** Logits (unnormalized probabilities) for each token position
   - Shape: `[batch_size, sequence_length, vocab_size]`
   - Contains scores for all ~50K possible tokens

2. **Ground Truth:** The `labels` tensor
   - Contains actual token IDs that should come next
   - At masked positions: `-100` (ignored)
   - At response positions: actual token IDs to predict

3. **Cross-Entropy Loss Calculation:**
   ```
   loss = Cross-Entropy(model_logits, labels)
   
   For each unmasked position:
       loss += -log(P(correct_token))
   
   Where P(correct_token) = softmax(logits)[correct_token]
   ```

### **Example:**

```
Position in Response: "Eiffel"
Model predicts logits: [0.1, 0.9, 0.2, ..., 0.05]  (vocab_size scores)
Ground truth label: 245 (token ID for "Eiffel")

Softmax probabilities: [0.02, 0.60, 0.05, ..., 0.01]
P(correct_token) = 0.60  ← Model predicted "Eiffel" with 60% confidence

Cross-entropy loss at this position:
loss = -log(0.60) ≈ 0.51

Backprop: Gradients flow back to update weights to increase P(Eiffel) next time
```

---

## 6. TRAINING STATISTICS

### **Task 2 (Noun-Verb Swap) Results:**

From [finetuning/ft_logs/task2.txt](finetuning/ft_logs/task2.txt):

```
Epoch 1:  Loss: 6.3313
Epoch 2:  Loss: 6.0679
Epoch 3:  Loss: 5.9106
...
Epoch 13: Loss: 4.4802
Epoch 14: Loss: 4.4407  ← Converged
```

**Observations:**
- Loss **decreased from 6.33 → 4.44** over 14 epochs
- Model learned the task progressively
- Plateau around epoch 13-14 suggests convergence
- **No cosine similarity used** - pure cross-entropy

---

## 7. KEY DIFFERENCES FROM OTHER APPROACHES

| Aspect | This Project | Other Approaches |
|--------|-------------|-----------------|
| **Loss Function** | Cross-Entropy | Contrastive, Cosine, MSE |
| **Training Type** | Causal LM | Classification, Similarity, Regression |
| **Prompt Masking** | ✅ Yes (only response counted) | Varies |
| **Weight Update** | AdamW + Backprop | Can vary |
| **LoRA** | ✅ Used (efficient) | Full fine-tune or none |

---

## Summary

- **Loss:** Cross-entropy loss (predicting next token)
- **Input Pipeline:** Instruction + Input concatenated → tokenized → response part unmasked, prompt part masked
- **Weight Updates:** Via backpropagation with AdamW optimizer
- **Comparison:** Expected (ground truth label) vs Real (model prediction) happens in cross-entropy loss calculation
- **No Cosine Similarity:** This is a pure causal language modeling task, not a contrastive/similarity task
