# LLM From Scratch

This project provides the building blocks for creating large language models from scratch. It is based off the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

The codebase includes:
- **Tokenization**: SimpleTokenizerV1 class for text preprocessing and vocabulary building
- **Dataset Creation**: GPTDatasetV1 class for generating training data with sliding window approach
- **Self-Attention**: Multiple attention mechanisms including basic (SelfAttention_v1, v2), causal, and multi-head attention
- **Utilities**: Helper functions for creating tokenizers and dataloaders
- **Chapter Examples**: Complete implementations from the book in the `book/` directory (ch2 and ch3)

## Setup

1. Install the package and its dependencies (via `pyproject.toml`):

   ```bash
   uv sync
   ```

2. For development with additional tools (pytest, black, ruff, pre-commit):

   ```bash
   uv sync --extra dev
   ```

## Usage

Once installed, you can import and use the tokenizer and dataset classes:

```python
from llm_from_scratch.tokenizer import SimpleTokenizerV1
from llm_from_scratch.dataset import GPTDatasetV1
from llm_from_scratch.attention import SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention
from llm_from_scratch.utils import create_tokenizer, create_dataloader_v1
import torch

# Tokenization
tokenizer = SimpleTokenizerV1(dataset="Hello, world!")
encoded = tokenizer.encode("Hello, world!")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)

# Dataset for training
dataset = GPTDatasetV1(txt="Hello, world!", tokenizer=tokenizer, max_length=4, stride=1)
print(len(dataset))

# Self-Attention
d_in, d_out = 3, 2
inputs = torch.rand(6, d_in)  # 6 tokens, 3-dimensional embeddings

# Basic implementation
attn_v1 = SelfAttention_v1(d_in, d_out)
context_v1 = attn_v1(inputs)

# Production-ready implementation with linear layers
attn_v2 = SelfAttention_v2(d_in, d_out, qkv_bias=False)
context_v2 = attn_v2(inputs)

# Causal attention for autoregressive models
batch_size = 2
inputs_batched = torch.rand(batch_size, 6, d_in)
causal_attn = CausalAttention(d_in, d_out, context_length=6, dropout=0.1)
context_causal = causal_attn(inputs_batched)

# Multi-head attention
multi_head = MultiHeadAttention(d_in, d_out*4, context_length=6, dropout=0.1, num_heads=4)
context_multi = multi_head(inputs_batched)

# Using utility functions
dataloader = create_dataloader_v1(
    txt="Your text here",
    batch_size=4,
    max_length=8,
    stride=4,
    use_bpe=True  # Uses tiktoken GPT-2 tokenizer
)
```

## Running Tests

This project includes pytest tests under `tests/`. To run them:

```bash
uv run pytest
```

## Development Tools

The project includes several development tools:

- **Testing**: `uv run pytest` or `uv run pytest -v` for verbose output
- **Code formatting**: `uv run black .` to format code
- **Linting**: `uv run ruff check .` to check for issues
- **Pre-commit hooks**: `uv run pre-commit install` to set up automatic checks

## Requirements

- Python 3.12 (3.13 not supported due to TensorFlow compatibility)
