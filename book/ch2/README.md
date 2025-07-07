# Chapter 2: Working with Text Data

This directory contains the implementation scripts for Chapter 2 of "Build a Large Language Model (From Scratch)" by Manning, focusing on preparing text data for large language model training.

## Overview

Chapter 2 covers the essential preprocessing steps required before training an LLM:

- Text tokenization and vocabulary creation
- Converting tokens to numerical representations
- Implementing byte pair encoding (BPE)
- Creating data loaders with sliding window sampling
- Token embeddings and positional encoding

## Scripts

### Core Implementation Files

- **`create_vocab.py`** - Creates vocabulary from tokenized text, mapping unique tokens to integer IDs
- **`tokenizer_example.py`** - Demonstrates basic tokenization using regular expressions to split text into words and punctuation
- **`tokenize_story.py`** - Tokenizes "The Verdict" short story using the custom tokenizer implementation
- **`tiktoken_example.py`** - Shows usage of the tiktoken library for BPE tokenization (GPT-2 style)
- **`dataloader.py`** - Implements PyTorch dataset and dataloader for creating input-target pairs using sliding window approach
- **`sliding_window.py`** - Demonstrates the sliding window technique for generating training samples
- **`token_embedding.py`** - Creates token embeddings and adds positional encodings
- **`read_story.py`** - Utility script to read and preprocess "The Verdict" text file

## Key Concepts Implemented

### 1. Tokenization
- Basic regex-based tokenization splitting on whitespace and punctuation
- Handling special characters and tokens like `<|endoftext|>` and `<|unk|>`
- Byte Pair Encoding (BPE) using tiktoken library

### 2. Vocabulary Building
- Creating mappings from tokens to integer IDs
- Handling unknown words with special tokens
- Vocabulary size considerations (GPT-2 uses 50,257 tokens)

### 3. Data Loading
- Sliding window approach for creating input-target pairs
- Configurable stride and context length
- Batch processing with PyTorch DataLoader

### 4. Embeddings
- Token embeddings converting IDs to dense vectors
- Positional embeddings for sequence position information
- Combining token and positional embeddings for LLM input

## Usage Example

```python
# Basic tokenization
from tokenizer_example import SimpleTokenizerV2
tokenizer = SimpleTokenizerV2(vocab)
tokens = tokenizer.encode("Hello, world!")

# BPE tokenization
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode("Hello, world!")

# Data loading
from dataloader import create_dataloader_v1
dataloader = create_dataloader_v1(
    text, batch_size=8, max_length=256, stride=128
)

# Token embeddings
import torch
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
token_embeddings = embedding_layer(token_ids)
```

## Dependencies

- PyTorch - for embeddings and data loading
- tiktoken - for BPE tokenization
- Standard Python libraries (re, urllib)

## Data

The scripts work with "The Verdict" by Edith Wharton, a public domain short story used as the training text example. The story contains 20,479 characters and demonstrates the complete text preprocessing pipeline.

## Next Steps

The preprocessed text data and embeddings created in this chapter serve as input for the attention mechanisms and transformer architecture implemented in Chapter 3.
