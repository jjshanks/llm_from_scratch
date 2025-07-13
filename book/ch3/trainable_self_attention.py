"""
Chapter 3 - Trainable Self-Attention Implementation

This script demonstrates the implementation of self-attention with trainable weights,
showing the step-by-step computation of attention scores, weights, and context vectors.

The example walks through:
1. Manual computation of self-attention for a single query token
2. Computing attention scores using dot products
3. Applying scaled softmax to get attention weights
4. Using attention weights to compute context vectors
5. Comparing manual computation with the SelfAttention classes

This implementation forms the foundation for transformer-based models like GPT.

Usage: uv run python book/ch3/trainable_self_attention.py
"""

from llm_from_scratch.self_attention import SelfAttention_v1, SelfAttention_v2
import torch


# Input embeddings representing the sentence "Your journey starts with one step"
# Each row is a 3-dimensional embedding vector for one token
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)

# Select the second token "journey" as our query example
x_2 = inputs[1]  # Shape: (3,)
d_in = inputs.shape[1]  # Input dimension = 3
d_out = 2  # Output dimension = 2 (for demonstration)

# Initialize weight matrices with random values
# In practice, these would be learned during training
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Step 1: Transform the query token (x_2) using weight matrices
query_2 = x_2 @ W_query  # Shape: (2,)
key_2 = x_2 @ W_key  # Shape: (2,)
value_2 = x_2 @ W_value  # Shape: (2,)
print("Query vector for token 2:", query_2)

# Step 2: Compute keys and values for ALL tokens
# This allows the query to attend to all positions in the sequence
keys = inputs @ W_key  # Shape: (6, 2)
values = inputs @ W_value  # Shape: (6, 2)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Step 3: Compute attention score between query_2 and key_2 (self-attention)
keys_2 = keys[1]  # Extract key for position 2
attn_score_22 = query_2.dot(keys_2)  # Dot product measures similarity
print("Self-attention score (2,2):", attn_score_22)

# Step 4: Compute attention scores between query_2 and ALL keys
# Higher scores indicate tokens that are more relevant to the query
attn_scores_2 = query_2 @ keys.T  # Shape: (6,)
print("All attention scores for query 2:", attn_scores_2)

# Step 5: Apply scaled softmax to get normalized attention weights
# Scaling by sqrt(d_k) prevents softmax from becoming too peaked
d_k = keys.shape[-1]  # Key dimension
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights (sum to 1):", attn_weights_2)

# Step 6: Compute context vector as weighted sum of values
# The context vector contains information from all tokens,
# weighted by their relevance to the query token
context_vec_2 = attn_weights_2 @ values  # Shape: (2,)
print("Context vector for token 2:", context_vec_2)

# === Using the SelfAttention_v1 class ===
print("\n=== Comparing with SelfAttention_v1 class ===")
torch.manual_seed(123)
# Create an instance of the SelfAttention_v1 class
sa_v1 = SelfAttention_v1(d_in, d_out)
# Calling sa_v1(inputs) automatically invokes the forward() method
# This performs the same self-attention computation as done manually above,
# but for all input tokens at once instead of just token 2
print("All context vectors from SelfAttention_v1:")
print(sa_v1(inputs))

# === Demonstrating SelfAttention_v2 with nn.Linear ===
print("\n=== SelfAttention_v2 with nn.Linear layers ===")
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# === Weight transfer demonstration ===
print("\n=== Transferring weights from v2 to v1 ===")
# Copy weights from nn.Linear (d_out x d_in) to nn.Parameter (d_in x d_out)
# Need to transpose because nn.Linear stores weights as (out_features, in_features)
# Also need to wrap in nn.Parameter since .T returns a regular tensor
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

# After weight transfer, both versions should produce identical outputs
print("SelfAttention_v1 output (with v2 weights):")
print(sa_v1(inputs))
print("SelfAttention_v2 output (original):")
print(sa_v2(inputs))

"""
Expected output:
Query vector for token 2: tensor([0.4306, 1.4551])
keys.shape: torch.Size([6, 2])
values.shape: torch.Size([6, 2])
Self-attention score (2,2): tensor(1.8524)
All attention scores for query 2: tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
Attention weights (sum to 1): tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
Context vector for token 2: tensor([0.3061, 0.8210])

=== Comparing with SelfAttention_v1 class ===
All context vectors from SelfAttention_v1:
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)

=== SelfAttention_v2 with nn.Linear layers ===

=== Transferring weights from v2 to v1 ===
SelfAttention_v1 output (with v2 weights):
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
SelfAttention_v2 output (original):
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
"""
