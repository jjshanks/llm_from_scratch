# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides building blocks for creating large language models from scratch, based on the book "Build a Large Language Model (From Scratch)" by Manning. The codebase is structured as a Python package with a focus on tokenization functionality.

## Development Setup

```bash
# Install package in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e '.[dev]'
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tokenizer.py

# Run with verbose output
uv run pytest -v
```

## Architecture

The project follows a standard Python package structure:
- `src/llm_from_scratch/` - Main package containing core implementations
- `src/llm_from_scratch/tokenizer.py` - SimpleTokenizerV1 class with vocabulary building and encoding/decoding
- `src/llm_from_scratch/dataset.py` - GPTDatasetV1 class for creating training data with sliding window approach
- `src/llm_from_scratch/attention.py` - Self-attention implementations (SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention)
- `src/llm_from_scratch/model.py` - GPT model components (LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel)
- `src/llm_from_scratch/utils.py` - Utility functions for creating tokenizers and dataloaders
- `book/ch2/` - Chapter 2 implementation examples and utilities
- `book/ch3/` - Chapter 3 self-attention mechanisms and trainable components (simple, masked, multi-head)
- `book/ch4/` - Chapter 4 GPT model building blocks and assembly scripts
- `tests/` - pytest test suite
- `pyproject.toml` - Project configuration and dependencies

## Key Components

### SimpleTokenizerV1 Class
The main `SimpleTokenizerV1` class handles:
- Vocabulary building from text datasets
- Text preprocessing and tokenization
- Encoding text to integer sequences
- Decoding integer sequences back to text
- Special token handling (`<|endoftext|>` and `<|unk|>`)

### GPTDatasetV1 Class
The `GPTDatasetV1` PyTorch Dataset class provides:
- Sliding window approach for creating input-target pairs
- Configurable stride and context length
- Integration with PyTorch DataLoader for batch processing

### Self-Attention Classes
**SelfAttention_v1**: Basic implementation using nn.Parameter
- Randomly initialized weight matrices for Q, K, V transformations
- Scaled dot-product attention computation
- Educational implementation for understanding core concepts

**SelfAttention_v2**: Production-ready implementation using nn.Linear
- Proper weight initialization (Xavier/Glorot)
- Optional bias terms for flexibility
- More stable training dynamics

**CausalAttention**: Masked self-attention for autoregressive models
- Causal masking to prevent attending to future tokens
- Dropout regularization for training
- Efficient batched computation

**MultiHeadAttention**: Parallel attention with multiple heads
- Splits attention across multiple representation subspaces
- Includes output projection layer
- Supports causal masking

### Model Components
**LayerNorm**: Layer normalization for training stability
- Normalizes activations to zero mean and unit variance
- Learnable scale and shift parameters
- Essential for deep transformer training

**GELU**: Gaussian Error Linear Unit activation
- Smooth, differentiable activation function
- Better performance than ReLU in transformers
- Standard activation for modern language models

**FeedForward**: Position-wise feed-forward network
- Two linear layers with GELU activation
- 4x hidden dimension expansion
- Dropout for regularization

**TransformerBlock**: Complete transformer layer
- Multi-head self-attention
- Feed-forward network
- Layer normalization (pre-norm architecture)
- Residual connections

**GPTModel**: Full GPT architecture
- Token and position embeddings
- Stack of transformer blocks
- Output projection layer
- Configurable architecture (layers, heads, dimensions)

### Utility Functions
The `utils.py` module provides convenience functions:
- `create_tokenizer()`: Creates SimpleTokenizerV1 or tiktoken BPE tokenizer
- `create_dataloader_v1()`: Creates DataLoader with GPTDatasetV1 for training

## Task Completion

Before committing changes:
1. Run `uv run pytest` to ensure all tests pass
2. Verify package imports work:
   - `from llm_from_scratch.tokenizer import SimpleTokenizerV1`
   - `from llm_from_scratch.dataset import GPTDatasetV1`
   - `from llm_from_scratch.attention import SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention`
   - `from llm_from_scratch.model import LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, GPTConfig`
   - `from llm_from_scratch.utils import create_tokenizer, create_dataloader_v1`
3. Ensure new dependencies are added to pyproject.toml

## Tech Stack

- Python 3.12 (3.13 not supported due to TensorFlow compatibility)
- PyTorch for deep learning components
- TikToken for tokenization utilities
- pytest for testing
- Additional ML libraries: matplotlib, tensorflow, numpy, pandas
