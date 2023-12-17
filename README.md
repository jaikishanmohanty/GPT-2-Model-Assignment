# GPT-2 Model Implementation and Optimization

This repository contains the implementation and optimization of the GPT-2 model by Jaikishan Mohanty (Roll No: 2K22/SWE/07). The tasks covered in this repository are:

## Task 1: Model Implementation and Checkpoints

### MultiHeadAttention Class

The `MultiHeadAttention` class implements the multi-head attention mechanism. It is a crucial component of the GPT-2 model.

### GPT2Small Class

The `GPT2Small` class represents the GPT-2 model with a smaller configuration. It includes the word embedding layer, positional embedding layer, and multiple layers of the transformer architecture.

#### Usage Example:
```python
vocab_size = 10000
embed_size = 128
heads = 4
depth = 3
model = GPT2Small(vocab_size, embed_size, heads, depth)

input_sequence = torch.randint(0, vocab_size, (32, 20))
mask = (input_sequence != 0).unsqueeze(1).unsqueeze(2)
output = model(input_sequence, mask)
print(output.shape)
```

## Task 2: Transformer Architectural Changes

In this task, the GPT-2 model is loaded using the Hugging Face `transformers` library. A sample text is provided as input, and the model generates text based on the input.

### Usage Example:

```python
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_text = "Once upon a time, in a"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

output_ids = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=torch.ones_like(input_ids)
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
```

## Task 3: Training Loop Implementation

This task includes a training loop for the GPT-2 model using the provided `test_gpt2_small` function. It generates text based on different input prompts.

### Usage Example:

```python
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

input_prompt = "Once upon a time, in a"
output_text = test_gpt2_small(gpt2_model, tokenizer, input_prompt)
print("Test Case 1:\n", "Input Prompt:", input_prompt, "\nGenerated Text:", output_text)
print("="*50)

input_prompt = "In a galaxy far, far away,"
output_text = test_gpt2_small(gpt2_model, tokenizer, input_prompt)
print("Test Case 2:\n", "Input Prompt:", input_prompt, "\nGenerated Text:", output_text)
print("="*50)

input_prompt = "To be or not to be, that is"
output_text = test_gpt2_small(gpt2_model, tokenizer, input_prompt)
print("Test Case 3:\n", "Input Prompt:", input_prompt, "\nGenerated Text:", output_text)
```
## Note

Make sure to install the required dependencies by running the following commands:

```bash
pip install einops
pip install rotary-embedding-torch
pip install torch
pip install axial-positional-embedding
pip install transformers
```
These dependencies are essential for running the provided code. If you encounter any issues during execution, ensure that you have the necessary libraries installed in your Python environment.

---

**Author:** Jaikishan Mohanty  
**Roll No:** 2K22/SWE/07


