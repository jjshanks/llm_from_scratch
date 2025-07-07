"""
Chapter 2 - Creating Token Embeddings and Positional Embeddings

This script demonstrates the final step in text preprocessing for LLM training:
converting token IDs into embedding vectors and adding positional information.

Key concepts covered:
1. Token embeddings: Converting discrete token IDs to continuous vector representations
2. Embedding layers as lookup operations
3. Positional embeddings: Adding position information to token embeddings
4. Combining token and positional embeddings for LLM input

The embedding process is crucial because:
- Neural networks require continuous numerical inputs
- Embeddings capture semantic relationships between tokens
- Positional embeddings help LLMs understand token order and context

Usage: uv run python book/ch2/token_embedding.py
"""

import torch
from dataloader import create_dataloader_v1

# Demonstrate basic embedding concepts with a simple example
print("=== Basic Token Embedding Example ===")

# Create sample token IDs for demonstration
input_ids = torch.tensor([2, 3, 5, 1])
print(f"Sample token IDs: {input_ids}")

# Define a small vocabulary and embedding dimension for illustration
vocab_size = 6  # Small vocabulary with 6 possible tokens
output_dim = 3  # 3-dimensional embedding vectors

# Set random seed for reproducible results
torch.manual_seed(123)

# Create an embedding layer - this is essentially a lookup table
# Each token ID maps to a unique embedding vector
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("Embedding layer weight matrix:")
print(embedding_layer.weight)

# Demonstrate single token embedding lookup
print("\nEmbedding for token ID 3:")
print(embedding_layer(torch.tensor([3])))

# Demonstrate batch token embedding
print(f"\nEmbeddings for all input token IDs {input_ids}:")
print(embedding_layer(input_ids))

print("\n" + "=" * 50)

# Realistic example with GPT-2 scale embeddings
print("=== Realistic LLM Token Embeddings ===")

# Use realistic vocabulary and embedding sizes similar to GPT-2
vocab_size = 50257  # GPT-2 BPE tokenizer vocabulary size
output_dim = 256  # Embedding dimension (smaller than GPT-2's 768 for demonstration)

# Create token embedding layer for the full vocabulary
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Set context length (sequence length)
max_length = 4

# Load the training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create data loader to get batched token sequences
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

# Get a batch of token sequences
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Convert token IDs to embeddings
# Shape: (batch_size, sequence_length, embedding_dim) = (8, 4, 256)
token_embeddings = token_embedding_layer(inputs)
print(f"Token embeddings shape: {token_embeddings.shape}")

print("\n=== Adding Positional Embeddings ===")

# Create positional embeddings to encode token positions
# GPT models use learnable absolute positional embeddings
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Create position indices: [0, 1, 2, 3] for sequence length 4
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"Positional embeddings shape: {pos_embeddings.shape}")

# Combine token embeddings with positional embeddings
# PyTorch broadcasts the positional embeddings across all batch samples
input_embeddings = token_embeddings + pos_embeddings
print(f"Final input embeddings shape: {input_embeddings.shape}")

print("\nInput embeddings are now ready for the LLM!")
"""expected output:
=== Basic Token Embedding Example ===
Sample token IDs: tensor([2, 3, 5, 1])
Embedding layer weight matrix:
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)

Embedding for token ID 3:
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

Embeddings for all input token IDs tensor([2, 3, 5, 1]):
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)

==================================================
=== Realistic LLM Token Embeddings ===
Token IDs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])
Token embeddings shape: torch.Size([8, 4, 256])

=== Adding Positional Embeddings ===
Positional embeddings shape: torch.Size([4, 256])
Final input embeddings shape: torch.Size([8, 4, 256])

Input embeddings are now ready for the LLM!
"""
