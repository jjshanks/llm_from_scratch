"""
Chapter 2 - Token and Positional Embeddings

Demonstrates converting token IDs to continuous embeddings and adding positional information.

Key concepts:
- Token embeddings: lookup table mapping IDs to vectors
- Positional embeddings: learnable position encodings
- Combining embeddings for LLM input

Usage: uv run python book/ch2/token_embedding.py
"""

import torch
import tiktoken
from dataloader import create_dataloader_v1


def display_embedding_slice(tensor, name="Embedding", num_dims=4):
    """Display first and last few dimensions of embedding tensor."""
    if len(tensor.shape) == 1:
        # Single embedding vector
        first_dims = tensor[:num_dims].tolist()
        last_dims = tensor[-num_dims:].tolist()
        first_str = ", ".join(f"{x:.4f}" for x in first_dims)
        last_str = ", ".join(f"{x:.4f}" for x in last_dims)
        print(f"{name}: [{first_str}, ..., {last_str}]")
    elif len(tensor.shape) == 2:
        # Batch of embeddings
        print(f"{name} (showing first {num_dims} and last {num_dims} dims):")
        for i, emb in enumerate(tensor):
            first_dims = emb[:num_dims].tolist()
            last_dims = emb[-num_dims:].tolist()
            first_str = ", ".join(f"{x:.4f}" for x in first_dims)
            last_str = ", ".join(f"{x:.4f}" for x in last_dims)
            print(f"  [{i}]: [{first_str}, ..., {last_str}]")
    else:
        # 3D tensor (batch, seq_len, embed_dim)
        print(f"{name} (showing first {num_dims} and last {num_dims} dims):")
        for batch_idx in range(tensor.shape[0]):
            print(f"  Batch {batch_idx}:")
            for seq_idx in range(tensor.shape[1]):
                emb = tensor[batch_idx, seq_idx]
                first_dims = emb[:num_dims].tolist()
                last_dims = emb[-num_dims:].tolist()
                first_str = ", ".join(f"{x:.4f}" for x in first_dims)
                last_str = ", ".join(f"{x:.4f}" for x in last_dims)
                print(f"    Pos {seq_idx}: [{first_str}, ..., {last_str}]")


def decode_token_ids(token_ids, tokenizer):
    """Decode token IDs to text."""
    if isinstance(token_ids, torch.Tensor):
        if len(token_ids.shape) == 1:
            # Single sequence
            return [tokenizer.decode([token_id.item()]) for token_id in token_ids]
        else:
            # Batch of sequences
            return [
                [tokenizer.decode([token_id.item()]) for token_id in seq]
                for seq in token_ids
            ]
    return token_ids


# Basic embedding example
print("=== Basic Token Embedding Example ===")

# Create sample token IDs
input_ids = torch.tensor([2, 3, 5, 1])
print(f"Sample token IDs: {input_ids}")

# Define small vocabulary and embedding dimension
vocab_size = 6  # 6 possible tokens
output_dim = 3  # 3-dimensional embeddings

# Set random seed for reproducible results
torch.manual_seed(123)

# Create embedding layer - lookup table mapping token IDs to vectors
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("Embedding layer weight matrix:")
print(embedding_layer.weight)

# Single token embedding lookup
print("\nEmbedding for token ID 3:")
print(embedding_layer(torch.tensor([3])))

# Batch token embedding
print(f"\nEmbeddings for all input token IDs {input_ids}:")
print(embedding_layer(input_ids))

print("\n" + "=" * 50)

# Realistic example with GPT-2 scale embeddings
print("=== Realistic LLM Token Embeddings ===")

# Use realistic vocabulary and embedding sizes
vocab_size = 50257  # GPT-2 BPE tokenizer vocabulary size
output_dim = 256  # Embedding dimension (smaller than GPT-2's 768)

# Create token embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Set context length
max_length = 4

# Load training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create data loader
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

# Get a batch of token sequences
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Create tiktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
print("\nToken IDs to Text Mapping:")
for i, token_seq in enumerate(inputs):
    tokens_text = decode_token_ids(token_seq, tokenizer)
    print(f"  Sample {i}: {token_seq.tolist()} -> {tokens_text}")

# Show first sample for detailed analysis
print("\nDetailed view of first sample:")
sample_tokens = inputs[0]
sample_text = decode_token_ids(sample_tokens, tokenizer)
print(f"Token IDs: {sample_tokens.tolist()}")
print(f"Text: {sample_text}")
print(f"Reconstructed: {''.join(sample_text)}")

# Convert token IDs to embeddings
# Shape: (batch_size, sequence_length, embedding_dim) = (8, 4, 256)
token_embeddings = token_embedding_layer(inputs)
print(f"Token embeddings shape: {token_embeddings.shape}")

# Display token embeddings for first sample
print(f"\nToken embeddings for first sample (first 4 + last 4 of {output_dim} dims):")
sample_token_embeddings = token_embeddings[0]  # Shape: (4, 256)
for i, (token_id, text, embedding) in enumerate(
    zip(sample_tokens, sample_text, sample_token_embeddings)
):
    print(f"  Position {i} - Token '{text}' (ID: {token_id.item()}):")
    display_embedding_slice(embedding, "    Token embedding", num_dims=4)

print("\n=== Adding Positional Embeddings ===")

# Create positional embeddings
# GPT models use learnable absolute positional embeddings
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Create position indices: [0, 1, 2, 3] for sequence length 4
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"Positional embeddings shape: {pos_embeddings.shape}")

# Display positional embeddings
print(f"\nPositional embeddings (first 4 + last 4 of {output_dim} dims):")
for i, pos_embedding in enumerate(pos_embeddings):
    print(f"  Position {i}:")
    display_embedding_slice(pos_embedding, "    Positional embedding", num_dims=4)

print("\n=== Step-by-Step Embedding Combination ===")

# Combine token embeddings with positional embeddings
# PyTorch broadcasts positional embeddings across all batch samples
input_embeddings = token_embeddings + pos_embeddings
print(f"Final input embeddings shape: {input_embeddings.shape}")

# Show step-by-step combination for first sample
print("\nStep-by-step combination for first sample:")
sample_input_embeddings = input_embeddings[0]  # Shape: (4, 256)
for i, (token_id, text) in enumerate(zip(sample_tokens, sample_text)):
    print(f"\n  Position {i} - Token '{text}' (ID: {token_id.item()}):")

    # Token embedding
    token_emb = sample_token_embeddings[i]
    display_embedding_slice(token_emb, "    Token embedding", num_dims=4)

    # Positional embedding
    pos_emb = pos_embeddings[i]
    display_embedding_slice(pos_emb, "    Positional embedding", num_dims=4)

    # Combined embedding
    combined_emb = sample_input_embeddings[i]
    display_embedding_slice(
        combined_emb, "    Final embedding (token + pos)", num_dims=4
    )

print("\n=== Embedding Analysis ===")

# Calculate embedding magnitudes
print("Embedding magnitudes (L2 norm):")
token_magnitudes = torch.norm(sample_token_embeddings, dim=1)
pos_magnitudes = torch.norm(pos_embeddings, dim=1)
final_magnitudes = torch.norm(sample_input_embeddings, dim=1)

for i, (token_id, text) in enumerate(zip(sample_tokens, sample_text)):
    print(
        f"  Position {i} - Token '{text}': "
        f"Token={token_magnitudes[i]:.4f}, "
        f"Pos={pos_magnitudes[i]:.4f}, "
        f"Final={final_magnitudes[i]:.4f}"
    )

print("\nInput embeddings are now ready for the LLM!")
"""expected output:
First batch (input_ids, target_ids):
[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

Second batch (shifted by stride=1):
[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

Batch with 8 samples:
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
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

Token IDs to Text Mapping:
  Sample 0: [40, 367, 2885, 1464] -> ['I', ' H', 'AD', ' always']
  Sample 1: [1807, 3619, 402, 271] -> [' thought', ' Jack', ' G', 'is']
  Sample 2: [10899, 2138, 257, 7026] -> ['burn', ' rather', ' a', ' cheap']
  Sample 3: [15632, 438, 2016, 257] -> [' genius', '--', 'though', ' a']
  Sample 4: [922, 5891, 1576, 438] -> [' good', ' fellow', ' enough', '--']
  Sample 5: [568, 340, 373, 645] -> ['so', ' it', ' was', ' no']
  Sample 6: [1049, 5975, 284, 502] -> [' great', ' surprise', ' to', ' me']
  Sample 7: [284, 3285, 326, 11] -> [' to', ' hear', ' that', ',']

Detailed view of first sample:
Token IDs: [40, 367, 2885, 1464]
Text: ['I', ' H', 'AD', ' always']
Reconstructed: I HAD always
Token embeddings shape: torch.Size([8, 4, 256])

Token embeddings for first sample (first 4 + last 4 of 256 dims):
  Position 0 - Token 'I' (ID: 40):
    Token embedding: [0.4913, 1.1239, 1.4588, -0.3653, ..., -1.1052, -0.3995, -1.8735, -0.1445]
  Position 1 - Token ' H' (ID: 367):
    Token embedding: [0.4481, 0.2536, -0.2655, -0.2798, ..., -0.0542, 0.4997, -1.1991, -1.1844]
  Position 2 - Token 'AD' (ID: 2885):
    Token embedding: [-0.2507, -0.0546, 0.6687, 0.7299, ..., 0.2570, 0.9618, 2.3737, -0.0528]
  Position 3 - Token ' always' (ID: 1464):
    Token embedding: [0.9457, 0.8657, 1.6191, 0.9051, ..., 0.4417, -0.4544, -0.7460, 0.3483]

=== Adding Positional Embeddings ===
Positional embeddings shape: torch.Size([4, 256])

Positional embeddings (first 4 + last 4 of 256 dims):
  Position 0:
    Positional embedding: [1.7375, -0.5620, -0.6303, -0.4848, ..., -0.7666, -0.2277, 1.5748, 1.0345]
  Position 1:
    Positional embedding: [1.6423, -0.7201, 0.2062, 0.6078, ..., 1.3007, 0.4118, 0.1498, -0.4628]
  Position 2:
    Positional embedding: [-0.4651, -0.7757, 0.5806, -1.3846, ..., -0.0383, 1.4335, -0.4963, 0.8579]
  Position 3:
    Positional embedding: [-0.6754, -0.4628, 1.4323, 0.2217, ..., -0.3207, 0.8139, -0.7088, 0.4827]

=== Step-by-Step Embedding Combination ===
Final input embeddings shape: torch.Size([8, 4, 256])

Step-by-step combination for first sample:

  Position 0 - Token 'I' (ID: 40):
    Token embedding: [0.4913, 1.1239, 1.4588, -0.3653, ..., -1.1052, -0.3995, -1.8735, -0.1445]
    Positional embedding: [1.7375, -0.5620, -0.6303, -0.4848, ..., -0.7666, -0.2277, 1.5748, 1.0345]
    Final embedding (token + pos): [2.2288, 0.5619, 0.8286, -0.8501, ..., -1.8717, -0.6272, -0.2987, 0.8900]

  Position 1 - Token ' H' (ID: 367):
    Token embedding: [0.4481, 0.2536, -0.2655, -0.2798, ..., -0.0542, 0.4997, -1.1991, -1.1844]
    Positional embedding: [1.6423, -0.7201, 0.2062, 0.6078, ..., 1.3007, 0.4118, 0.1498, -0.4628]
    Final embedding (token + pos): [2.0903, -0.4664, -0.0593, 0.3280, ..., 1.2465, 0.9115, -1.0493, -1.6473]

  Position 2 - Token 'AD' (ID: 2885):
    Token embedding: [-0.2507, -0.0546, 0.6687, 0.7299, ..., 0.2570, 0.9618, 2.3737, -0.0528]
    Positional embedding: [-0.4651, -0.7757, 0.5806, -1.3846, ..., -0.0383, 1.4335, -0.4963, 0.8579]
    Final embedding (token + pos): [-0.7158, -0.8304, 1.2494, -0.6547, ..., 0.2186, 2.3952, 1.8773, 0.8051]

  Position 3 - Token ' always' (ID: 1464):
    Token embedding: [0.9457, 0.8657, 1.6191, 0.9051, ..., 0.4417, -0.4544, -0.7460, 0.3483]
    Positional embedding: [-0.6754, -0.4628, 1.4323, 0.2217, ..., -0.3207, 0.8139, -0.7088, 0.4827]
    Final embedding (token + pos): [0.2703, 0.4029, 3.0514, 1.1268, ..., 0.1210, 0.3595, -1.4548, 0.8310]

=== Embedding Analysis ===
Embedding magnitudes (L2 norm):
  Position 0 - Token 'I': Token=15.7231, Pos=15.9539, Final=23.5554
  Position 1 - Token ' H': Token=16.2072, Pos=16.2630, Final=23.1529
  Position 2 - Token 'AD': Token=15.7313, Pos=16.4955, Final=22.5571
  Position 3 - Token ' always': Token=16.9367, Pos=16.1315, Final=23.1143

Input embeddings are now ready for the LLM!
"""
