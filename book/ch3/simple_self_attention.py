"""
Chapter 3 - Simple Self-Attention Mechanism

This script implements a simplified self-attention mechanism without trainable weights
to illustrate the fundamental concepts before adding complexity.

Key concepts demonstrated:
- Computing attention scores using dot products
- Normalizing scores to get attention weights
- Creating context vectors as weighted sums of inputs
- Generalizing from single token to all tokens
- Visualizing attention patterns
- Comparing attention-based vs simple averaging

The implementation follows Section 3.3 of the book:
1. First computes attention for a single query token
2. Then generalizes to compute attention for all tokens
3. Shows both manual computation and efficient matrix operations

Usage: uv run python book/ch3/simple_self_attention.py
"""

import torch


def visualize_attention(attn_weights, tokens, query_idx=None):
    """
    Display attention weights as a text-based heatmap.

    Args:
        attn_weights: Attention weight matrix (num_tokens x num_tokens)
        tokens: List of token strings
        query_idx: If provided, highlights a specific query row
    """
    print("\nAttention Weight Visualization:")
    print("(Higher weights shown with more filled blocks)")
    print("\n         ", end="")
    # Column headers
    for token in tokens:
        print(f"{token:8}", end=" ")
    print("\n" + "-" * (10 + 9 * len(tokens)))

    for i, token_from in enumerate(tokens):
        # Row label with optional highlighting
        if query_idx is not None and i == query_idx:
            print(f"*{token_from:7}*", end=" ")
        else:
            print(f"{token_from:8}", end=" ")

        # Attention weights visualization
        for j, weight in enumerate(attn_weights[i]):
            # Use different characters for different weight ranges
            if weight > 0.3:
                print("████", end="     ")
            elif weight > 0.2:
                print("▓▓▓", end="      ")
            elif weight > 0.15:
                print("▒▒", end="       ")
            elif weight > 0.1:
                print("░", end="        ")
            else:
                print("·", end="        ")

        # Show actual values for the query row
        if query_idx is not None and i == query_idx:
            print("  <- query")
        else:
            print()

    # Add legend
    print("\nLegend: · (0-0.1) ░ (0.1-0.15) ▒▒ (0.15-0.2) ▓▓▓ (0.2-0.3) ████ (>0.3)")


# === Section 3.3.1: A simple self-attention mechanism without trainable weights ===

# Define tokens for our example sentence
tokens = ["Your", "journey", "starts", "with", "one", "step"]
print("Input sentence:", " ".join(tokens))

# Create more meaningful embeddings where similar concepts have similar vectors
# Dimension 0: entity/concept (high for nouns/pronouns)
# Dimension 1: action/movement (high for verbs/action nouns)
# Dimension 2: grammatical/functional (high for function words)
inputs = torch.tensor(
    [
        [0.9, 0.1, 0.2],  # Your     - pronoun (entity)
        [0.8, 0.7, 0.1],  # journey  - noun (entity + action)
        [0.2, 0.9, 0.2],  # starts   - verb (action)
        [0.1, 0.1, 0.9],  # with     - preposition (functional)
        [0.7, 0.2, 0.3],  # one      - number (entity-like)
        [0.6, 0.5, 0.2],
    ]  # step     - noun (entity + some action)
)

print("\nEmbedding vectors (rows = tokens, columns = embedding dimensions):")
print("         entity  action  function")
print("         ------  ------  --------")
for i, token in enumerate(tokens):
    embedding = inputs[i].tolist()
    print(
        f"{token:8}   {embedding[0]:.1f}     {embedding[1]:.1f}       {embedding[2]:.1f}"
    )

# === Computing attention scores for a single query (Figure 3.8) ===

print("\n" + "=" * 60)
print("=== Computing attention for query token 'journey' ===")

# Use second input element x^(2) ("journey") as query
query_idx = 1
query = inputs[query_idx]
query_formatted = [f"{val:.1f}" for val in query.tolist()]
print(
    f"\nQuery token: '{tokens[query_idx]}' with embedding [{', '.join(query_formatted)}]"
)
print("  Dimensions: [entity=0.8, action=0.7, function=0.1]")
print("  → 'journey' has high entity and action values (it's a noun about movement)")

# Compute attention scores between query and all input tokens via dot product
# The dot product measures similarity between vectors
print("\nComputing similarity scores (dot products):")
print("Higher scores = more similar to 'journey' in the embedding space")
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
    # Add interpretation of similarity
    interpretation = ""
    if attn_scores_2[i] > 1.0:
        interpretation = " ← highest (self-similarity)"
    elif attn_scores_2[i] > 0.8:
        interpretation = " ← high (related concept)"
    elif attn_scores_2[i] < 0.3:
        interpretation = " ← low (different type)"
    print(f"  {tokens[i]:8} · journey = {attn_scores_2[i]:.4f}{interpretation}")

scores_formatted = [f"{val:.4f}" for val in attn_scores_2.tolist()]
print(f"\nRaw attention scores: [{', '.join(scores_formatted)}]")
print("(These will be normalized to sum to 1.0 to create attention weights)")

# === Understanding dot products ===
# (Dot product = sum of element-wise multiplication, computed internally by torch.dot)

# === Normalizing attention scores to get weights (Figure 3.9) ===

# Simple normalization by dividing by sum
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
weights_formatted = [f"{val:.4f}" for val in attn_weights_2_tmp.tolist()]
print(f"\nAttention weights (simple normalization): [{', '.join(weights_formatted)}]")
print(f"Sum: {attn_weights_2_tmp.sum():.4f}")


# Define naive softmax function for illustration
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# Apply softmax normalization (preferred method)
attn_weights_2_naive = softmax_naive(attn_scores_2)
weights_formatted = [f"{val:.4f}" for val in attn_weights_2_naive.tolist()]
print(f"\nAttention weights (softmax): [{', '.join(weights_formatted)}]")
print(f"Sum: {attn_weights_2_naive.sum():.4f}")

print("\nInterpreting attention weights for 'journey':")
for i, (token, weight) in enumerate(zip(tokens, attn_weights_2_naive)):
    bar = "█" * int(weight * 20)  # Visual bar
    percentage = weight * 100
    print(f"  {token:8} {weight:.4f} ({percentage:4.1f}%) {bar}")

# Use PyTorch's optimized softmax implementation
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# === Computing context vector (Figure 3.10) ===

# Context vector z^(2) is weighted sum of all input vectors
print("\n=== Computing context vector ===")
query = inputs[query_idx]
context_vec_2 = torch.zeros(query.shape)

print("\nWeighted contributions to context vector:")
print("Each contribution = attention_weight × token_embedding")
print("")
for i, x_i in enumerate(inputs):
    contribution = attn_weights_2[i] * x_i
    context_vec_2 += contribution

    # Show the multiplication explicitly
    weight = attn_weights_2[i].item()
    embedding = inputs[i].tolist()
    contrib = contribution.tolist()

    print(
        f"  {tokens[i]:8}: {weight:.4f} × [{embedding[0]:.1f}, {embedding[1]:.1f}, {embedding[2]:.1f}] = [{contrib[0]:.4f}, {contrib[1]:.4f}, {contrib[2]:.4f}]"
    )

# Show the final sum
context_formatted = [f"{val:.4f}" for val in context_vec_2.tolist()]
print("  " + "─" * 70)
print(f"  Sum = [{', '.join(context_formatted)}]  ← context vector z^({query_idx+1})")
print("\nThis context vector is an enriched representation of 'journey'")
print("that incorporates information from all tokens in the sentence.")

# === Compare with simple averaging (no attention) ===

print("\n" + "=" * 60)
print("=== Comparing Attention vs Simple Averaging ===")

# Simple averaging gives equal weight to all tokens
simple_avg = inputs.mean(dim=0)
simple_formatted = [f"{val:.4f}" for val in simple_avg.tolist()]
print(f"\nSimple average (equal weights): [{', '.join(simple_formatted)}]")
print(f"Attention-based context:        [{', '.join(context_formatted)}]")

# Compare the difference
difference = (context_vec_2 - simple_avg).abs()
diff_formatted = [f"{val:.4f}" for val in difference.tolist()]
print(f"\nAbsolute difference:            [{', '.join(diff_formatted)}]")
print(f"Total difference (L1 norm):     {difference.sum():.4f}")

# Show what attention emphasizes
print("\nWhat attention emphasizes for 'journey':")
orig_formatted = [f"{val:.1f}" for val in inputs[query_idx].tolist()]
print(f"- Original 'journey' embedding: [{', '.join(orig_formatted)}]")
print("- Context vector enriched with related tokens")
print("- Note how 'starts' and 'step' (action-related) get higher weights")

# === Section 3.3.2: Computing attention weights for all input tokens ===

print("\n" + "=" * 60)
print("=== Generalizing to all tokens ===")

# Step 1: Compute all attention scores via matrix multiplication
# (More efficient than nested loops)
attn_scores = inputs @ inputs.T
print("\nAttention scores matrix (via inputs @ inputs.T):")
print(attn_scores)

# Step 2: Normalize each row to get attention weights
# dim=-1 means normalize along last dimension (columns)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nAttention weights matrix:")
print(attn_weights)

# Visualize the attention pattern
visualize_attention(attn_weights, tokens)

# Verify that each row sums to 1
print("\nVerifying row sums:")
print("All row sums:", attn_weights.sum(dim=-1))

# Step 3: Compute all context vectors via matrix multiplication
all_context_vecs = attn_weights @ inputs
print("\nAll context vectors:")
for i, (token, ctx_vec) in enumerate(zip(tokens, all_context_vecs)):
    ctx_formatted = [f"{val:.4f}" for val in ctx_vec.tolist()]
    print(f"  {token:8}: [{', '.join(ctx_formatted)}]")

# Note: Row 1 matches our detailed computation for 'journey' above

# === Analyzing attention patterns ===

print("\n" + "=" * 60)
print("=== Analyzing Attention Patterns ===")

# Show which tokens each word pays most attention to
print("\nHighest attention weights for each token:")
for i, token in enumerate(tokens):
    max_idx = attn_weights[i].argmax().item()
    max_weight = attn_weights[i].max().item()
    self_weight = attn_weights[i][i].item()

    if max_idx == i:
        print(f"  {token:8} pays most attention to itself (weight={max_weight:.4f})")
    else:
        print(
            f"  {token:8} pays most attention to '{tokens[int(max_idx)]}' (weight={max_weight:.4f})"
        )
        print(f"           (self-attention weight: {self_weight:.4f})")

print("\nWhy do 'one' and 'step' attend more to 'journey' than themselves?")
print("Let's examine the self-similarity scores (diagonal of attention score matrix):")
for i, token in enumerate(tokens):
    self_score = attn_scores[i][i].item()
    print(f"  {token:8} self-similarity: {self_score:.4f}")

print("\nKey insights:")
print(
    "- 'journey' has the highest self-similarity (1.14) due to strong entity+action values"
)
print("- 'one' (0.62) and 'step' (0.65) have weaker self-similarity")
print("- After softmax normalization, 'journey' becomes an attention magnet")
print(
    "- This demonstrates how attention helps tokens 'borrow' context from semantically rich words"
)

# Compare with simple average to show benefit of attention
print("\n=== Benefits of Attention ===")
simple_context = inputs.mean(dim=0, keepdim=True).expand(len(tokens), -1)
attention_benefit = (
    (all_context_vecs - simple_context).abs().sum() / len(tokens)
).item()
print(f"\nAverage change from simple averaging: {attention_benefit:.4f}")
print("This shows how attention creates more diverse, context-aware representations")
print("compared to treating all tokens equally.")

"""expected output:
Input sentence: Your journey starts with one step

Embedding vectors (rows = tokens, columns = embedding dimensions):
         entity  action  function
         ------  ------  --------
Your       0.9     0.1       0.2
journey    0.8     0.7       0.1
starts     0.2     0.9       0.2
with       0.1     0.1       0.9
one        0.7     0.2       0.3
step       0.6     0.5       0.2

============================================================
=== Computing attention for query token 'journey' ===

Query token: 'journey' with embedding [0.8, 0.7, 0.1]
  Dimensions: [entity=0.8, action=0.7, function=0.1]
  ΓåÆ 'journey' has high entity and action values (it's a noun about movement)

Computing similarity scores (dot products):
Higher scores = more similar to 'journey' in the embedding space
  Your     ┬╖ journey = 0.8100 ΓåÉ high (related concept)
  journey  ┬╖ journey = 1.1400 ΓåÉ highest (self-similarity)
  starts   ┬╖ journey = 0.8100 ΓåÉ high (related concept)
  with     ┬╖ journey = 0.2400 ΓåÉ low (different type)
  one      ┬╖ journey = 0.7300
  step     ┬╖ journey = 0.8500 ΓåÉ high (related concept)

Raw attention scores: [0.8100, 1.1400, 0.8100, 0.2400, 0.7300, 0.8500]
(These will be normalized to sum to 1.0 to create attention weights)

Attention weights (simple normalization): [0.1769, 0.2489, 0.1769, 0.0524, 0.1594, 0.1856]
Sum: 1.0000

Attention weights (softmax): [0.1689, 0.2349, 0.1689, 0.0955, 0.1559, 0.1758]
Sum: 1.0000

Interpreting attention weights for 'journey':
  Your     0.1689 (16.9%) ΓûêΓûêΓûê
  journey  0.2349 (23.5%) ΓûêΓûêΓûêΓûê
  starts   0.1689 (16.9%) ΓûêΓûêΓûê
  with     0.0955 ( 9.6%) Γûê
  one      0.1559 (15.6%) ΓûêΓûêΓûê
  step     0.1758 (17.6%) ΓûêΓûêΓûê

=== Computing context vector ===

Weighted contributions to context vector:
Each contribution = attention_weight ├ù token_embedding

  Your    : 0.1689 ├ù [0.9, 0.1, 0.2] = [0.1520, 0.0169, 0.0338]
  journey : 0.2349 ├ù [0.8, 0.7, 0.1] = [0.1880, 0.1645, 0.0235]
  starts  : 0.1689 ├ù [0.2, 0.9, 0.2] = [0.0338, 0.1520, 0.0338]
  with    : 0.0955 ├ù [0.1, 0.1, 0.9] = [0.0096, 0.0096, 0.0860]
  one     : 0.1559 ├ù [0.7, 0.2, 0.3] = [0.1091, 0.0312, 0.0468]
  step    : 0.1758 ├ù [0.6, 0.5, 0.2] = [0.1055, 0.0879, 0.0352]
  ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  Sum = [0.5979, 0.4620, 0.2590]  ΓåÉ context vector z^(2)

This context vector is an enriched representation of 'journey'
that incorporates information from all tokens in the sentence.

============================================================
=== Comparing Attention vs Simple Averaging ===

Simple average (equal weights): [0.5500, 0.4167, 0.3167]
Attention-based context:        [0.5979, 0.4620, 0.2590]

Absolute difference:            [0.0479, 0.0453, 0.0577]
Total difference (L1 norm):     0.1510

What attention emphasizes for 'journey':
- Original 'journey' embedding: [0.8, 0.7, 0.1]
- Context vector enriched with related tokens
- Note how 'starts' and 'step' (action-related) get higher weights

============================================================
=== Generalizing to all tokens ===

Attention scores matrix (via inputs @ inputs.T):
tensor([[0.8600, 0.8100, 0.3100, 0.2800, 0.7100, 0.6300],
        [0.8100, 1.1400, 0.8100, 0.2400, 0.7300, 0.8500],
        [0.3100, 0.8100, 0.8900, 0.2900, 0.3800, 0.6100],
        [0.2800, 0.2400, 0.2900, 0.8300, 0.3600, 0.2900],
        [0.7100, 0.7300, 0.3800, 0.3600, 0.6200, 0.5800],
        [0.6300, 0.8500, 0.6100, 0.2900, 0.5800, 0.6500]])

Attention weights matrix:
tensor([[0.2108, 0.2005, 0.1216, 0.1180, 0.1815, 0.1675],
        [0.1689, 0.2349, 0.1689, 0.0955, 0.1559, 0.1758],
        [0.1276, 0.2104, 0.2279, 0.1251, 0.1368, 0.1722],
        [0.1471, 0.1413, 0.1486, 0.2550, 0.1594, 0.1486],
        [0.1910, 0.1948, 0.1373, 0.1346, 0.1746, 0.1677],
        [0.1692, 0.2109, 0.1659, 0.1204, 0.1610, 0.1726]])

Attention Weight Visualization:
(Higher weights shown with more filled blocks)

         Your     journey  starts   with     one      step
----------------------------------------------------------------
Your     ΓûôΓûôΓûô      ΓûôΓûôΓûô      Γûæ        Γûæ        ΓûÆΓûÆ       ΓûÆΓûÆ
journey  ΓûÆΓûÆ       ΓûôΓûôΓûô      ΓûÆΓûÆ       ┬╖        ΓûÆΓûÆ       ΓûÆΓûÆ
starts   Γûæ        ΓûôΓûôΓûô      ΓûôΓûôΓûô      Γûæ        Γûæ        ΓûÆΓûÆ
with     Γûæ        Γûæ        Γûæ        ΓûôΓûôΓûô      ΓûÆΓûÆ       Γûæ
one      ΓûÆΓûÆ       ΓûÆΓûÆ       Γûæ        Γûæ        ΓûÆΓûÆ       ΓûÆΓûÆ
step     ΓûÆΓûÆ       ΓûôΓûôΓûô      ΓûÆΓûÆ       Γûæ        ΓûÆΓûÆ       ΓûÆΓûÆ

Legend: ┬╖ (0-0.1) Γûæ (0.1-0.15) ΓûÆΓûÆ (0.15-0.2) ΓûôΓûôΓûô (0.2-0.3) ΓûêΓûêΓûêΓûê (>0.3)

Verifying row sums:
All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

All context vectors:
  Your    : [0.6138, 0.4028, 0.2807]
  journey : [0.5979, 0.4620, 0.2590]
  starts  : [0.5403, 0.4911, 0.2802]
  with    : [0.5014, 0.3791, 0.3803]
  one     : [0.5915, 0.4113, 0.2922]
  step    : [0.5825, 0.4444, 0.2793]

============================================================
=== Analyzing Attention Patterns ===

Highest attention weights for each token:
  Your     pays most attention to itself (weight=0.2108)
  journey  pays most attention to itself (weight=0.2349)
  starts   pays most attention to itself (weight=0.2279)
  with     pays most attention to itself (weight=0.2550)
  one      pays most attention to 'journey' (weight=0.1948)
           (self-attention weight: 0.1746)
  step     pays most attention to 'journey' (weight=0.2109)
           (self-attention weight: 0.1726)

Why do 'one' and 'step' attend more to 'journey' than themselves?
Let's examine the self-similarity scores (diagonal of attention score matrix):
  Your     self-similarity: 0.8600
  journey  self-similarity: 1.1400
  starts   self-similarity: 0.8900
  with     self-similarity: 0.8300
  one      self-similarity: 0.6200
  step     self-similarity: 0.6500

Key insights:
- 'journey' has the highest self-similarity (1.14) due to strong entity+action values
- 'one' (0.62) and 'step' (0.65) have weaker self-similarity
- After softmax normalization, 'journey' becomes an attention magnet
- This demonstrates how attention helps tokens 'borrow' context from semantically rich words

=== Benefits of Attention ===

Average change from simple averaging: 0.1173
This shows how attention creates more diverse, context-aware representations
compared to treating all tokens equally.
"""
