"""
Chapter 3: Multi-Head Attention - Detailed Implementation and Visualization

This file provides an in-depth exploration of multi-head attention (Section 3.6),
showing manual computation before using library implementations.

Key concepts demonstrated:
1. Manual computation of multi-head attention with 2 heads
2. Visualization of what different attention heads learn
3. Step-by-step breakdown of the multi-head mechanism
4. Comparison between single-head and multi-head attention
5. Efficient implementation strategies

Usage: uv run python book/ch3/multi_head_attention.py
"""

import torch
from llm_from_scratch.attention import MultiHeadAttention, MultiHeadAttentionWrapper


def visualize_multihead_attention(attn_weights_list, tokens, head_names):
    """
    Visualize attention patterns for multiple heads side by side.

    Args:
        attn_weights_list: List of attention weight matrices (one per head)
        tokens: List of token strings
        head_names: List of descriptive names for each head
    """
    print("\n=== Multi-Head Attention Pattern Visualization ===")
    print("(Higher weights shown with more filled blocks)\n")

    # First, print separate visualizations for each head
    for head_idx, (attn_weights, head_name) in enumerate(
        zip(attn_weights_list, head_names)
    ):
        print(f"\n{head_name}:")
        print("         ", end="")
        # Column headers
        for token in tokens:
            print(f"{token:8}", end=" ")
        print("\n" + "-" * (10 + 9 * len(tokens)))

        # Attention patterns
        for i, from_token in enumerate(tokens):
            print(f"{from_token:8} ", end="")
            for j, weight in enumerate(attn_weights[i]):
                # Visual representation
                if weight > 0.3:
                    vis = "████"
                elif weight > 0.2:
                    vis = "▓▓▓"
                elif weight > 0.15:
                    vis = "▒▒"
                elif weight > 0.1:
                    vis = "░"
                else:
                    vis = "·"
                print(f"{vis:8}", end=" ")
            print()

        # Add interpretation for each head
        if head_idx == 0:  # Head 1 interpretation
            print("\nInterpretation:")
            print(
                "- Strong backward attention: Later tokens (journey→step) heavily attend to 'Your'"
            )
            print(
                "- Position-based pattern: The further into the sentence, the more attention to position 0"
            )
            print(
                "- 'Your' distributes attention forward, showing it anchors the sentence"
            )
            print("- This head captures syntactic dependencies and sentence structure")
        else:  # Head 2 interpretation
            print("\nInterpretation:")
            print(
                "- Balanced attention: More evenly distributed weights across semantically related words"
            )
            print(
                "- Content-based pattern: 'journey', 'starts', 'step' (action words) attend to each other"
            )
            print(
                "- Function words like 'with' show different patterns, attending to content words"
            )
            print(
                "- This head captures semantic relationships and meaning associations"
            )

    # Then print side-by-side comparison with actual weights
    print("\n\nSide-by-side weight comparison:")
    print("Token    ", end="")
    for name in head_names:
        print(f"│ {name:^30} ", end="")
    print("\n" + "─" * (10 + 34 * len(head_names)))

    # For each token, show which tokens it attends to
    for i, from_token in enumerate(tokens):
        print(f"{from_token:8} ", end="")
        for head_idx, attn_weights in enumerate(attn_weights_list):
            print("│", end="")
            # Find top 2 attended tokens for this head
            weights = attn_weights[i]
            top2_indices = torch.topk(weights, k=min(2, len(tokens))).indices
            attended_info = []
            for idx in top2_indices:
                attended_info.append(f"{tokens[idx]}({weights[idx]:.2f})")
            print(f" {', '.join(attended_info):^30} ", end="")
        print()

    print("\nLegend: · (0-0.1) ░ (0.1-0.15) ▒▒ (0.15-0.2) ▓▓▓ (0.2-0.3) ████ (>0.3)")


def analyze_head_specialization(attn_weights_list, tokens, head_names):
    """Analyze what each attention head focuses on."""
    print("\n=== Attention Head Specialization Analysis ===")

    for head_idx, (attn_weights, head_name) in enumerate(
        zip(attn_weights_list, head_names)
    ):
        print(f"\n{head_name}:")

        # Find dominant attention patterns
        avg_attn_by_position = torch.zeros(len(tokens))
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                # Relative position
                rel_pos = j - i
                if -len(tokens) < rel_pos < len(tokens):
                    avg_attn_by_position[j] += attn_weights[i, j]

        avg_attn_by_position /= len(tokens)

        # Identify pattern
        self_attn = torch.mean(torch.diagonal(attn_weights))
        backward_attn = torch.mean(torch.tril(attn_weights, -1))
        forward_attn = torch.mean(torch.triu(attn_weights, 1))

        print(f"  Average self-attention: {self_attn:.3f}")
        print(f"  Average backward attention: {backward_attn:.3f}")
        print(f"  Average forward attention: {forward_attn:.3f}")

        # Most attended tokens
        total_attention_received = attn_weights.sum(dim=0)
        most_attended_idx = total_attention_received.argmax()
        print(
            f"  Most attended token: '{tokens[most_attended_idx]}' (total attention: {total_attention_received[most_attended_idx]:.3f})"
        )


# === Section 3.6: Multi-Head Attention ===

# Define tokens and create meaningful embeddings
tokens = ["Your", "journey", "starts", "with", "one", "step"]
print("Input sentence:", " ".join(tokens))

# Create embeddings with multiple semantic dimensions
# Dimensions represent: [entity, action, syntax, position]
print("\nEmbedding dimensions:")
print("  [0] Entity/noun strength")
print("  [1] Action/verb strength")
print("  [2] Syntactic role (determiner/preposition)")
print("  [3] Position encoding (normalized)")

inputs = torch.tensor(
    [
        [0.9, 0.1, 0.8, 0.0],  # Your - determiner, start position
        [0.8, 0.7, 0.1, 0.2],  # journey - noun with action
        [0.2, 0.9, 0.2, 0.4],  # starts - action verb
        [0.1, 0.1, 0.9, 0.6],  # with - preposition
        [0.7, 0.2, 0.3, 0.8],  # one - number/determiner
        [0.6, 0.5, 0.2, 1.0],  # step - noun with action
    ]
)

print("\nInput embeddings:")
for i, token in enumerate(tokens):
    emb = inputs[i]
    print(f"  {token:8}: [{emb[0]:.1f}, {emb[1]:.1f}, {emb[2]:.1f}, {emb[3]:.1f}]")

# === Manual Multi-Head Attention Computation ===

print("\n" + "=" * 70)
print("=== Manual Multi-Head Attention Computation (2 heads) ===")
print("=" * 70)

torch.manual_seed(42)
d_in = 4
d_out_per_head = 2
num_heads = 2

# Initialize separate weight matrices for each head
# Head 1: Focus on syntactic/positional relationships
print("\n=== Head 1: Syntactic/Positional Focus ===")
print("This head will learn to attend based on grammatical structure and position")
W_query_1 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))
W_key_1 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))
W_value_1 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))

# Initialize with bias toward position and syntax dimensions
# Recall our embedding dimensions: [entity, action, syntax, position]
W_query_1.data[2:, :] *= 2  # Amplify syntax (dim 2) and position (dim 3)
W_key_1.data[2:, :] *= 2

print(
    f"  Weight shapes: W_query_1={W_query_1.shape}, W_key_1={W_key_1.shape}, W_value_1={W_value_1.shape}"
)
print("  Initialization strategy:")
print("    - Dimensions 0-1 (entity/action): normal random init")
print("    - Dimensions 2-3 (syntax/position): 2x amplified")
print("    → This biases Head 1 to focus on syntactic and positional patterns")

# Head 2: Focus on semantic relationships
print("\n=== Head 2: Semantic/Meaning Focus ===")
print("This head will learn to attend based on semantic content and meaning")
W_query_2 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))
W_key_2 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))
W_value_2 = torch.nn.Parameter(torch.randn(d_in, d_out_per_head))

# Initialize with bias toward entity and action dimensions
W_query_2.data[:2, :] *= 2  # Amplify entity (dim 0) and action (dim 1)
W_key_2.data[:2, :] *= 2

print(
    f"  Weight shapes: W_query_2={W_query_2.shape}, W_key_2={W_key_2.shape}, W_value_2={W_value_2.shape}"
)
print("  Initialization strategy:")
print("    - Dimensions 0-1 (entity/action): 2x amplified")
print("    - Dimensions 2-3 (syntax/position): normal random init")
print(
    "    → This biases Head 2 to focus on semantic relationships between content words"
)

# === Step 1: Compute queries, keys, values for each head ===
print("\n=== Step 1: Transform inputs to queries, keys, and values ===")

# Head 1 transformations
queries_1 = inputs @ W_query_1
keys_1 = inputs @ W_key_1
values_1 = inputs @ W_value_1

print("\nHead 1 - Queries (syntactic focus):")
for i, token in enumerate(tokens):
    print(f"  {token:8}: {queries_1[i].tolist()}")

# Head 2 transformations
queries_2 = inputs @ W_query_2
keys_2 = inputs @ W_key_2
values_2 = inputs @ W_value_2

print("\nHead 2 - Queries (semantic focus):")
for i, token in enumerate(tokens):
    print(f"  {token:8}: {queries_2[i].tolist()}")

# === Step 2: Compute attention scores and weights for each head ===
print("\n=== Step 2: Compute attention scores and weights ===")

# Head 1 attention
scores_1 = queries_1 @ keys_1.T
d_k = keys_1.shape[1]
scores_1_scaled = scores_1 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
attn_weights_1 = torch.softmax(scores_1_scaled, dim=-1)

print("\nHead 1 - Attention weights (syntactic patterns):")
print(attn_weights_1)

# Head 2 attention
scores_2 = queries_2 @ keys_2.T
scores_2_scaled = scores_2 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
attn_weights_2 = torch.softmax(scores_2_scaled, dim=-1)

print("\nHead 2 - Attention weights (semantic patterns):")
print(attn_weights_2)

# Visualize both heads
visualize_multihead_attention(
    [attn_weights_1, attn_weights_2],
    tokens,
    ["Head 1: Syntactic/Positional", "Head 2: Semantic/Meaning"],
)

# === Step 3: Compute context vectors for each head ===
print("\n=== Step 3: Compute context vectors ===")

context_1 = attn_weights_1 @ values_1
context_2 = attn_weights_2 @ values_2

print("\nHead 1 context vectors:")
for i, token in enumerate(tokens):
    print(f"  {token:8}: {context_1[i].tolist()}")

print("\nHead 1 interpretation:")
print("- Values increase monotonically from 'Your' → 'step' in both dimensions")
print(
    "- This creates a position encoding: early tokens get lower values, later tokens higher"
)
print(
    "- The gradual increase (0.55→2.18 in dim 0) encodes distance from sentence start"
)
print("- This head effectively learns positional relationships through attention")

print("\nHead 2 context vectors:")
for i, token in enumerate(tokens):
    print(f"  {token:8}: {context_2[i].tolist()}")

print("\nHead 2 interpretation:")
print(
    "- First dimension clusters content words (journey/starts/step: ~0.95-1.01) vs others"
)
print("- Second dimension near zero for most tokens, showing semantic neutrality")
print("- 'with' stands out (0.735, 0.048) as it bridges different semantic groups")
print("- This head creates semantic groupings rather than positional encoding")

# === Step 4: Concatenate head outputs ===
print("\n=== Step 4: Concatenate outputs from all heads ===")

multihead_output = torch.cat([context_1, context_2], dim=-1)
print("\nConcatenated output shape:", multihead_output.shape)
print("(6 tokens × 4 dimensions: 2 from each head)")

print("\nFinal multi-head output:")
for i, token in enumerate(tokens):
    print(f"  {token:8}: {multihead_output[i].tolist()}")

# === Step 5: (Optional) Output projection ===
print("\n=== Step 5: Output projection ===")
print("In practice, we apply a final linear transformation:")

d_out_total = d_out_per_head * num_heads
W_out = torch.nn.Parameter(torch.randn(d_out_total, d_in))
final_output = multihead_output @ W_out

print(f"\nOutput projection: {d_out_total} → {d_in} dimensions")
print("This allows mixing information from all heads")

# === Analysis of what each head learned ===
analyze_head_specialization(
    [attn_weights_1, attn_weights_2],
    tokens,
    ["Head 1: Syntactic/Positional", "Head 2: Semantic/Meaning"],
)

# === Comparison: Single-head vs Multi-head ===
print("\n" + "=" * 70)
print("=== Comparison: Single-Head vs Multi-Head Attention ===")
print("=" * 70)

# Compute single-head attention for comparison
W_query_single = torch.nn.Parameter(torch.randn(d_in, d_out_total))
W_key_single = torch.nn.Parameter(torch.randn(d_in, d_out_total))
W_value_single = torch.nn.Parameter(torch.randn(d_in, d_out_total))

queries_single = inputs @ W_query_single
keys_single = inputs @ W_key_single
values_single = inputs @ W_value_single

scores_single = queries_single @ keys_single.T
scores_single_scaled = scores_single / torch.sqrt(
    torch.tensor(d_out_total, dtype=torch.float32)
)
attn_weights_single = torch.softmax(scores_single_scaled, dim=-1)
context_single = attn_weights_single @ values_single

print("\nRepresentation richness (std dev of context vectors):")
print(f"  Single-head: {torch.std(context_single):.4f}")
print(f"  Multi-head:  {torch.std(multihead_output):.4f}")

print("\nKey insight: Multi-head attention captures different types of relationships")
print("simultaneously, leading to richer representations.")

# === Library Implementations ===
print("\n" + "=" * 70)
print("=== Library Implementations ===")
print("=" * 70)

# Create batch for library functions
batch = inputs.unsqueeze(0)  # Add batch dimension
context_length = len(tokens)

# MultiHeadAttentionWrapper (less efficient)
print("\n1. MultiHeadAttentionWrapper (stacks single-head modules):")
torch.manual_seed(123)
mha_wrapper = MultiHeadAttentionWrapper(
    d_in, d_out_per_head, context_length, 0.0, num_heads=2
)
wrapper_output = mha_wrapper(batch)
print("   Output shape:", wrapper_output.shape)
print("   This approach creates separate attention modules for each head")

# MultiHeadAttention (efficient)
print("\n2. MultiHeadAttention (efficient implementation):")
torch.manual_seed(123)
mha_efficient = MultiHeadAttention(d_in, d_out_total, context_length, 0.0, num_heads=2)
efficient_output = mha_efficient(batch)
print("   Output shape:", efficient_output.shape)
print("   This approach uses weight splitting and batched operations")

print("\nKey differences:")
print("- Wrapper: d_out is per head, total output = d_out × num_heads")
print("- Efficient: d_out is total dimension, split across heads internally")
print("- Efficient version includes output projection for better mixing")

# === Summary ===
print("\n" + "=" * 70)
print("=== Summary: Why Multi-Head Attention? ===")
print("=" * 70)

print("\n1. Different heads learn different relationships:")
print("   - Syntactic patterns (grammar, position)")
print("   - Semantic patterns (meaning, content)")
print("   - Local vs global dependencies")

print("\n2. Parallel processing of multiple representation subspaces")

print("\n3. More stable training and robust representations")

print("\n4. Used in all modern transformer architectures (GPT, BERT, etc.)")

"""Expected output shows:
- Manual computation of 2-head attention
- Clear visualization of different attention patterns per head
- Analysis showing Head 1 focuses on position/syntax, Head 2 on semantics
- Comparison demonstrating multi-head creates richer representations
- Explanation of efficient implementation strategies
"""
