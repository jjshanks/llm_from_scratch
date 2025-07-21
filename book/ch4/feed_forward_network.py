"""
Chapter 4 - Feed Forward Network in Transformers

This script explains the Feed Forward Network (FFN) component that appears
in every transformer block. The FFN processes each token independently,
applying a series of learned transformations.

Key concepts demonstrated:
- FFN architecture: Linear → GELU → Linear
- Why we expand then contract dimensions (bottleneck)
- How linear layers transform inputs
- What patterns FFN might learn
- Position-wise processing (no cross-token interaction)

The FFN is like a "thinking space" where each token's representation
is expanded, processed, and refined before moving to the next layer.

Usage: uv run python book/ch4/feed_forward_network.py
"""

import torch
from llm_from_scratch.model import FeedForward
from configuration import GPT_CONFIG_124M


def explain_ffn_architecture():
    """Explain the Feed Forward Network architecture."""
    print("=== Feed Forward Network in Transformers ===\n")

    print("Structure: Linear → GELU → Linear")
    print("Expands dimension by 4x internally (768 → 3072 → 768)")
    print("This expansion allows learning complex transformations")

    print("\n--- What Does the FFN Actually Do? ---")
    print("\nThink of it like a 'thinking space' for each token:")
    print("1. First Linear (768 → 3072): Projects to higher dimension")
    print("   - Like spreading data across more neurons to analyze")
    print("   - Each of 3072 neurons looks for different patterns")
    print("   - More dimensions = more capacity to detect features")

    print("\n2. GELU Activation: Adds non-linearity")
    print("   - Without this, two Linear layers = one Linear layer!")
    print("   - Allows learning complex, non-linear patterns")
    print("   - Smooth activation helps with gradient flow")

    print("\n3. Second Linear (3072 → 768): Projects back down")
    print("   - Combines the 3072 features into final representation")
    print("   - Forces model to compress/summarize what it learned")


def explain_linear_transformation():
    """Explain how linear layers transform inputs."""
    print("\n\n=== How Does Linear Layer Transform x → h? ===")

    print("\nIt's NOT dividing or copying! Each h is a weighted combination:")
    print("\nFor input x = [x₁, x₂, ..., x₇₆₈], each output neuron h_i:")
    print("  h₁ = w₁,₁×x₁ + w₂,₁×x₂ + ... + w₇₆₈,₁×x₇₆₈ + b₁")
    print("  h₂ = w₁,₂×x₁ + w₂,₂×x₂ + ... + w₇₆₈,₂×x₇₆₈ + b₂")
    print("  ...")
    print("  h₃₀₇₂ = w₁,₃₀₇₂×x₁ + w₂,₃₀₇₂×x₂ + ... + w₇₆₈,₃₀₇₂×x₇₆₈ + b₃₀₇₂")

    print("\nEach h_i uses DIFFERENT weights - it's looking for different patterns!")


def concrete_example():
    """Show a concrete example with small numbers."""
    print("\n\n=== Concrete Example with Small Numbers ===")

    print("\nSay x = [1, 2, 3] and we expand to 4 neurons:")
    print("  h₁ = 0.1×1 + 0.5×2 + (-0.2)×3 = 0.5   (detects pattern A)")
    print("  h₂ = -0.3×1 + 0.2×2 + 0.4×3 = 1.3    (detects pattern B)")
    print("  h₃ = 0.8×1 + (-0.1)×2 + 0.1×3 = 0.9  (detects pattern C)")
    print("  h₄ = 0.0×1 + 0.7×2 + 0.3×3 = 2.3     (detects pattern D)")

    print("\nNotice: Each output is a UNIQUE combination of ALL inputs!")
    print("The weights determine what patterns each neuron detects.")


def explain_neuron_specialization():
    """Explain how neurons specialize during training."""
    print("\n\n=== What Makes Each Neuron Different? ===")

    print("\nThe WEIGHTS! Each of the 3072 neurons has its own set of 768 weights.")
    print("That's 768 × 3072 = 2,359,296 trainable parameters in W1 alone!")

    print("\nDuring training, each neuron learns to detect different features:")
    print("• Neuron 1 might learn weights that activate for 'animal-like' patterns")
    print("• Neuron 2 might learn weights that activate for 'verb-like' patterns")
    print("• Neuron 3 might learn weights that activate for 'past-tense' patterns")
    print("• ... and so on for all 3072 neurons")

    print("\nIt's like having 3072 different 'detectors', each looking for")
    print("something different in the 768-dimensional input vector!")


def visualize_ffn():
    """ASCII visualization of FFN architecture."""
    print("\n\n=== ASCII Visualization of FFN ===")

    print("\nToken Input (768 dims)")
    print("    │")
    print("    │  [x₁, x₂, ..., x₇₆₈]")
    print("    ↓")
    print("Linear Layer (W1 @ x + b1)")
    print("    │")
    print("    │  Each h_i = weighted sum of ALL x values:")
    print("    │  h₁ = Σ(w₁ᵢ × xᵢ) + b₁  ←─ unique weights")
    print("    │  h₂ = Σ(w₂ᵢ × xᵢ) + b₂  ←─ different weights")
    print("    │  h₃ = Σ(w₃ᵢ × xᵢ) + b₃  ←─ different weights")
    print("    │  ...")
    print("    │  h₃₀₇₂ = Σ(w₃₀₇₂ᵢ × xᵢ) + b₃₀₇₂")
    print("    ↓")
    print("GELU Activation")
    print("    │")
    print("    │  [GELU(h₁), GELU(h₂), ..., GELU(h₃₀₇₂)]")
    print("    ↓")
    print("Linear Layer (W2 @ h + b2)")
    print("    │")
    print("    │  [y₁, y₂, ..., y₇₆₈]")
    print("    ↓")
    print("Token Output (768 dims)")

    print("\nKey insight: Even though input and output have same size,")
    print("the intermediate expansion allows complex transformations!")


def demonstrate_ffn():
    """Demonstrate FFN with actual computation."""
    print("\n\n=== Demonstrating FFN Operation ===")

    # Create FFN module
    ffn = FeedForward(GPT_CONFIG_124M)

    # Test with random input: [batch_size=2, seq_len=3, embed_dim=768]
    torch.manual_seed(42)
    x = torch.rand(2, 3, 768)

    print(f"\nInput shape:  {x.shape}")
    print("[batch_size=2, seq_len=3, embed_dim=768]")

    # Forward pass
    out = ffn(x)

    print(f"\nOutput shape: {out.shape}")
    print("[batch_size=2, seq_len=3, embed_dim=768]")

    print("\nNote: Input and output shapes match!")
    print("This allows stacking many transformer blocks")

    # Look at intermediate dimensions
    print("\n--- Internal Dimensions ---")
    print(f"Input dimension:      {ffn.layers[0].in_features}")
    print(f"Hidden dimension:     {ffn.layers[0].out_features}")
    print(f"Output dimension:     {ffn.layers[2].out_features}")
    print(
        f"Expansion factor:     {ffn.layers[0].out_features / ffn.layers[0].in_features:.1f}x"
    )

    # Count parameters
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\nTotal FFN parameters: {total_params:,}")
    print(
        f"First linear:  {ffn.layers[0].weight.numel() + ffn.layers[0].bias.numel():,}"
    )
    print(
        f"Second linear: {ffn.layers[2].weight.numel() + ffn.layers[2].bias.numel():,}"
    )


def explain_bottleneck_design():
    """Explain the bottleneck architecture design."""
    print("\n\n=== Why This Bottleneck Design? ===")

    print("\n• Bottleneck architecture: expand then compress")
    print("• Expansion gives model room to compute complex features")
    print("• Compression forces it to extract only what's important")
    print("• Applied to EACH token position independently")

    print("\n--- Real World Analogy ---")
    print("Like analyzing an image:")
    print("1. Expand: Extract many features (edges, colors, textures...)")
    print("2. Process: Identify patterns in those features")
    print("3. Compress: Summarize into useful representation")

    print("\nThe FFN does this for each token's representation!")


def show_learned_patterns():
    """Explain what patterns FFN might learn."""
    print("\n\n=== Example: What Patterns Might FFN Learn? ===")

    print("\nImagine the token 'cat' with its 768-dim representation:")
    print("• Some of the 3072 neurons might activate for:")
    print("  - 'animal' patterns")
    print("  - 'pet' patterns")
    print("  - 'mammal' patterns")
    print("  - 'noun' patterns")
    print("  - 'living thing' patterns")
    print("  - And many more subtle features...")

    print("\nThe second Linear layer then combines these to create")
    print("a refined representation that captures the essence of 'cat'")
    print("in the context of the current sentence.")

    print("\nIMPORTANT: FFN operates on EACH token independently!")
    print("It doesn't look at other tokens - that's the attention's job.")


def explain_position_wise():
    """Explain position-wise feed forward processing."""
    print("\n\n=== Position-wise Processing ===")

    print("\n'Position-wise' means the FFN is applied to each position independently:")

    print("\nFor sequence [The, cat, sat]:")
    print("  - FFN processes 'The' → refined 'The'")
    print("  - FFN processes 'cat' → refined 'cat'")
    print("  - FFN processes 'sat' → refined 'sat'")

    print("\nNo cross-position interaction! Each token is processed alone.")
    print("This is different from attention, which looks across all positions.")

    print("\nWhy separate?")
    print("1. Attention: Gathers information from other tokens")
    print("2. FFN: Processes the gathered information for each token")
    print("3. Together: Complete transformer block functionality")


def compare_ffn_sizes():
    """Compare FFN sizes in different models."""
    print("\n\n=== FFN Sizes in Different Models ===")

    models = [
        ("GPT-2 Small", 768, 3072),
        ("GPT-2 Medium", 1024, 4096),
        ("GPT-2 Large", 1280, 5120),
        ("GPT-3 175B", 12288, 49152),
        ("PaLM 540B", 18432, 73728),
    ]

    print(
        f"{'Model':<15} {'Embed Dim':<10} {'FFN Dim':<10} {'Expansion':<10} {'Parameters':<15}"
    )
    print("-" * 60)

    for name, embed_dim, ffn_dim in models:
        expansion = ffn_dim / embed_dim
        params = 2 * (embed_dim * ffn_dim + ffn_dim)  # Approximate
        print(
            f"{name:<15} {embed_dim:<10} {ffn_dim:<10} {expansion:<10.1f}x {params:>14,}"
        )

    print("\nObservations:")
    print("- All models use 4x expansion (standard ratio)")
    print("- Larger models have wider FFNs")
    print("- FFN parameters scale quadratically with width")


def implementation_tips():
    """Provide implementation tips for FFN."""
    print("\n\n=== Implementation Tips ===")

    print("\n1. **Initialization:**")
    print("   - Use proper weight initialization (Xavier/He)")
    print("   - Biases usually initialized to zero")

    print("\n2. **Dropout:**")
    print("   - Often applied after activation")
    print("   - Sometimes after each linear layer")
    print("   - Typical rates: 0.1-0.3")

    print("\n3. **Alternatives to 4x expansion:**")
    print("   - Some models use 2x or 8x")
    print("   - Depends on compute budget")
    print("   - Can be a hyperparameter to tune")

    print("\n4. **Activation choices:**")
    print("   - GELU is standard for transformers")
    print("   - Some models use SwiGLU or GeGLU")
    print("   - Affects convergence and performance")

    print("\n5. **Memory considerations:**")
    print("   - FFN uses significant memory during training")
    print("   - Gradient checkpointing can help")
    print("   - Consider mixed precision training")


def mathematical_view():
    """Show the mathematical formulation."""
    print("\n\n=== Mathematical View ===")

    print("\nFor input x ∈ ℝ^d where d is embedding dimension:")

    print("\nFFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂")

    print("\nWhere:")
    print("• W₁ ∈ ℝ^(4d×d) projects from d to 4d dimensions")
    print("• b₁ ∈ ℝ^4d is the first bias vector")
    print("• GELU is the activation function")
    print("• W₂ ∈ ℝ^(d×4d) projects from 4d back to d dimensions")
    print("• b₂ ∈ ℝ^d is the second bias vector")

    print("\nIn our GPT-2 config:")
    print("• d = 768")
    print("• 4d = 3072")
    print("• Total parameters = 768×3072 + 3072 + 3072×768 + 768 = 4,719,360")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Feed Forward Network in Transformers")
    print("=" * 60)

    # Architecture explanation
    explain_ffn_architecture()

    # Linear transformation details
    explain_linear_transformation()

    # Concrete example
    concrete_example()

    # Neuron specialization
    explain_neuron_specialization()

    # Visualization
    visualize_ffn()

    # Demonstration
    demonstrate_ffn()

    # Bottleneck design
    explain_bottleneck_design()

    # Learned patterns
    show_learned_patterns()

    # Position-wise processing
    explain_position_wise()

    # Model comparisons
    compare_ffn_sizes()

    # Implementation tips
    implementation_tips()

    # Mathematical view
    mathematical_view()

    print("\n" + "=" * 60)
    print("FFN: Where each token gets its own 'thinking space'!")
    print("=" * 60)
