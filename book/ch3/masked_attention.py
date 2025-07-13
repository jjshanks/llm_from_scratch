"""
Chapter 3 - Causal Attention Mechanism with Masking

This script demonstrates implementing causal (masked) attention to prevent
the model from accessing future tokens when computing attention scores.

Key concepts demonstrated:
- Applying causal attention masks to hide future tokens
- Two methods of masking: post-softmax normalization vs pre-softmax -inf masking
- Adding dropout to attention weights for regularization
- Batch processing with the CausalAttention class

The implementation follows Section 3.5 of the book:
1. First shows manual masking with renormalization (less efficient)
2. Then demonstrates the -inf masking trick (more efficient)
3. Adds dropout for reducing overfitting during training
4. Finally uses the CausalAttention class for batch processing

Usage: uv run python book/ch3/masked_attention.py
"""

from llm_from_scratch.causal_attention import CausalAttention
from llm_from_scratch.self_attention import SelfAttention_v2
import torch

# === Section 3.5.1: Applying a causal attention mask ===

# Input embeddings representing the sentence "Your journey starts with one step"
# Each row is a 3-dimensional embedding vector for one token
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)

# Set up dimensions
d_in = inputs.shape[1]  # Input embedding dimension = 3
d_out = 2  # Output embedding dimension = 2 (for demonstration)

# Initialize self-attention module with trainable weights
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# Step 1: Compute attention weights using standard self-attention
# This gives us the baseline attention weights before masking
print("=== Computing attention weights before masking ===")
queries = sa_v2.W_query(inputs)  # Shape: [6, 2]
keys = sa_v2.W_key(inputs)  # Shape: [6, 2]
attn_scores = queries @ keys.T  # Shape: [6, 6]

# Apply scaled dot-product attention
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print("\nAttention weights (no masking):")
print(attn_weights)
print("\nNote: Each row sums to 1.0, but future tokens are still visible")

# === Method 1: Post-softmax masking and renormalization (less efficient) ===

print("\n=== Method 1: Apply mask after softmax ===")

# Step 2: Create a lower triangular mask to hide future tokens
# tril creates a lower triangular matrix (1s on/below diagonal, 0s above)
context_length = attn_scores.shape[0]  # Number of tokens = 6
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\nCausal mask (1 = keep, 0 = hide):")
print(mask_simple)

# Step 3: Apply mask by element-wise multiplication
# This zeros out attention weights for future tokens
masked_simple = attn_weights * mask_simple
print("\nMasked attention weights (future tokens zeroed):")
print(masked_simple)
print("\nNote: Rows no longer sum to 1.0!")

# Step 4: Renormalize so each row sums to 1 again
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("\nRenormalized masked attention weights:")
print(masked_simple_norm)
print("\nVerify rows sum to 1.0:", masked_simple_norm.sum(dim=-1))
# === Method 2: Pre-softmax masking with -inf (more efficient) ===

print("\n=== Method 2: Apply mask before softmax (more efficient) ===")

# Create upper triangular mask (1s above diagonal)
# We'll use this to identify positions to mask out
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("\nUpper triangular mask (1 = positions to hide):")
print(mask)

# Replace masked positions with -inf BEFORE applying softmax
# When softmax sees -inf, it outputs 0 for those positions
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("\nAttention scores with -inf for future positions:")
print(masked)

# Apply softmax - the -inf values become 0 automatically
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("\nFinal masked attention weights (automatic normalization):")
print(attn_weights)
print("\nNote: Same result as Method 1, but more efficient!")

# === Section 3.5.2: Masking additional attention weights with dropout ===

print("\n=== Adding dropout for regularization ===")

# Dropout randomly zeros out values during training to prevent overfitting
# Here we use 50% dropout for demonstration (typical values are 10-20%)
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # 50% dropout rate

# First, let's see how dropout works on a simple matrix of ones
example = torch.ones(6, 6)
print("\nDropout applied to matrix of ones:")
print(dropout(example))
print("Note: ~50% of values are zeroed, remaining values are scaled by 2x")

# Apply dropout to our masked attention weights
torch.manual_seed(123)
print("\nDropout applied to masked attention weights:")
print(dropout(attn_weights))
print("\nNote: Additional weights are zeroed out for regularization")
print("During inference, dropout is disabled and all weights are used")

# === Section 3.5.3: Using the CausalAttention class ===

print("\n=== Batch processing with CausalAttention class ===")

# Create a batch of inputs by duplicating our sentence
# This simulates processing multiple sequences at once
batch = torch.stack((inputs, inputs), dim=0)
print("\nBatch shape:", batch.shape)
print("Dimensions: [batch_size=2, num_tokens=6, embedding_dim=3]")

# Initialize CausalAttention module
# This encapsulates all the masking logic we implemented above
torch.manual_seed(123)
context_length = batch.shape[1]  # Maximum sequence length
ca = CausalAttention(
    d_in,  # Input embedding dimension
    d_out,  # Output embedding dimension
    context_length,  # Maximum sequence length
    0.0,  # Dropout rate (0 = no dropout for now)
)

# Process the batch
context_vecs = ca(batch)
print("\nOutput context vectors shape:", context_vecs.shape)
print("Dimensions: [batch_size=2, num_tokens=6, output_dim=2]")

print("\nContext vectors for first sequence:")
print(context_vecs[0])

print("\nKey features of CausalAttention:")
print("- Automatically applies causal masking")
print("- Handles batches of sequences")
print("- Includes optional dropout for regularization")
print("- Registers mask as buffer for efficient GPU usage")
"""
expected output:
=== Computing attention weights before masking ===

Attention weights (no masking):
tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)

Note: Each row sums to 1.0, but future tokens are still visible

=== Method 1: Apply mask after softmax ===

Causal mask (1 = keep, 0 = hide):
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])

Masked attention weights (future tokens zeroed):
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>)

Note: Rows no longer sum to 1.0!

Renormalized masked attention weights:
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)

Verify rows sum to 1.0: tensor([1., 1., 1., 1., 1., 1.], grad_fn=<SumBackward1>)

=== Method 2: Apply mask before softmax (more efficient) ===

Upper triangular mask (1 = positions to hide):
tensor([[0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]])

Attention scores with -inf for future positions:
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)

Final masked attention weights (automatic normalization):
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)

Note: Same result as Method 1, but more efficient!

=== Adding dropout for regularization ===

Dropout applied to matrix of ones:
tensor([[2., 2., 0., 2., 2., 0.],
        [0., 0., 0., 2., 0., 2.],
        [2., 2., 2., 2., 0., 2.],
        [0., 2., 2., 0., 0., 2.],
        [0., 2., 0., 2., 0., 2.],
        [0., 2., 2., 2., 2., 0.]])
Note: ~50% of values are zeroed, remaining values are scaled by 2x

Dropout applied to masked attention weights:
tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
        [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
       grad_fn=<MulBackward0>)

Note: Additional weights are zeroed out for regularization
During inference, dropout is disabled and all weights are used

=== Batch processing with CausalAttention class ===

Batch shape: torch.Size([2, 6, 3])
Dimensions: [batch_size=2, num_tokens=6, embedding_dim=3]

Output context vectors shape: torch.Size([2, 6, 2])
Dimensions: [batch_size=2, num_tokens=6, output_dim=2]

Context vectors for first sequence:
tensor([[-0.4519,  0.2216],
        [-0.5874,  0.0058],
        [-0.6300, -0.0632],
        [-0.5675, -0.0843],
        [-0.5526, -0.0981],
        [-0.5299, -0.1081]], grad_fn=<SelectBackward0>)

Key features of CausalAttention:
- Automatically applies causal masking
- Handles batches of sequences
- Includes optional dropout for regularization
- Registers mask as buffer for efficient GPU usage
"""
