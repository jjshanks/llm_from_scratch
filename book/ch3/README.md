# Chapter 3: Understanding Attention Mechanisms

This directory contains the implementation scripts for Chapter 3 of "Build a Large Language Model (From Scratch)" by Manning, focusing on attention mechanisms that form the core of transformer architectures.

## Overview

Chapter 3 covers the fundamental attention mechanisms used in modern LLMs:

- Basic self-attention computation
- Trainable self-attention with learned parameters
- Masked self-attention for autoregressive models
- Multi-head attention for parallel representation learning

## Scripts

### Core Implementation Files

- **`simple_self_attention.py`** - Basic self-attention using fixed weights to demonstrate the attention computation concept
- **`trainable_self_attention.py`** - Self-attention with learnable weight matrices (W_q, W_k, W_v) for query, key, and value transformations
- **`masked_attention.py`** - Causal self-attention with masking to prevent attending to future tokens (essential for GPT-style models)
- **`multi_head_attention.py`** - Multi-head attention mechanism that runs multiple attention operations in parallel

## Key Concepts Implemented

### 1. Basic Self-Attention
- Dot-product attention computation
- Attention weights calculation using softmax
- Context vector aggregation
- Visualization of attention patterns

### 2. Trainable Self-Attention
- Learnable query (Q), key (K), and value (V) transformations
- Scaled dot-product attention to prevent gradient issues
- Parameter initialization strategies

### 3. Causal Masking
- Upper triangular mask creation
- Preventing information leakage from future tokens
- Essential for autoregressive language modeling

### 4. Multi-Head Attention
- Splitting representation into multiple heads
- Parallel attention computations
- Concatenation and output projection
- Improved model expressiveness

## Usage Example

```python
# Simple self-attention
from simple_self_attention import simple_self_attention
attention_scores, context_vectors = simple_self_attention(inputs)

# Trainable self-attention
from trainable_self_attention import SelfAttention
self_attn = SelfAttention(d_in, d_out)
output = self_attn(input_embeddings)

# Masked attention
from masked_attention import CausalAttention
causal_attn = CausalAttention(d_in, d_out, context_length)
output = causal_attn(input_embeddings)

# Multi-head attention
from multi_head_attention import MultiHeadAttentionWrapper
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, num_heads)
output = mha(input_embeddings)
```

## Key Parameters

- **d_in**: Input dimension (embedding size)
- **d_out**: Output dimension per attention head
- **context_length**: Maximum sequence length
- **num_heads**: Number of parallel attention heads
- **dropout**: Dropout rate for regularization

## Dependencies

- PyTorch - for tensor operations and neural network modules
- NumPy - for numerical computations
- Matplotlib - for visualizing attention weights

## Connection to Transformer Architecture

The attention mechanisms implemented here are building blocks for:
- The self-attention layers in transformer encoder blocks
- The masked self-attention in transformer decoder blocks
- The foundation for GPT-style language models

## Next Steps

These attention mechanisms will be integrated into complete transformer blocks in Chapter 4, including:
- Layer normalization
- Feed-forward networks
- Residual connections
- Complete GPT architecture
