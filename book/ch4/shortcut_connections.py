"""
Chapter 4 - Shortcut (Residual) Connections

This script demonstrates shortcut connections (also called residual connections),
which are crucial for training deep neural networks. They provide direct paths
for gradients to flow backward, preventing the vanishing gradient problem.

Key concepts demonstrated:
- Vanishing gradient problem in deep networks
- How shortcuts provide gradient highways
- Comparing networks with and without shortcuts
- Implementation in transformer blocks
- Why this enables very deep models

Shortcut connections were introduced in ResNet and are now standard
in all modern deep architectures, including transformers.

Usage: uv run python book/ch4/shortcut_connections.py
"""

import torch
import torch.nn as nn
from llm_from_scratch.model import ExampleDeepNeuralNetwork


def explain_gradient_problem():
    """Explain the vanishing gradient problem in deep networks."""
    print("=== Understanding Shortcut Connections ===\n")

    print("Problem: In deep networks, gradients can vanish")
    print("Solution: Shortcuts provide direct paths for gradients")

    print("\n--- Why Do Gradients Vanish? ---")
    print("\nDuring backpropagation, gradients are multiplied through layers:")
    print("• If gradient at each layer < 1, the product approaches 0")
    print("• Example: 0.8 × 0.8 × 0.8 × 0.8 = 0.4096")
    print("• After 10 layers: 0.8^10 = 0.107")
    print("• After 20 layers: 0.8^20 = 0.012")
    print("• The gradient signal disappears!")

    print("\nThis means:")
    print("• Early layers stop learning (no gradient)")
    print("• Network acts like it's shallow")
    print("• Deeper doesn't mean better!")


def visualize_architectures():
    """ASCII visualization of regular vs shortcut architectures."""
    print("\n\n=== Architecture Comparison ===")

    print("\n--- ASCII Diagram: Regular Deep Network ---")
    print("Input → Layer1 → Layer2 → Layer3 → Layer4 → Output")
    print("         ↓         ↓         ↓         ↓")
    print("      (grad₄)   (grad₃)   (grad₂)   (grad₁)")
    print("\nGradient flow: grad₄ × grad₃ × grad₂ × grad₁")
    print("If each gradient < 1, product → 0 (vanishing!)")

    print("\n--- ASCII Diagram: Network with Shortcut Connections ---")
    print("Input ─┬→ Layer1 ─┬→ Layer2 ─┬→ Layer3 ─┬→ Layer4 → Output")
    print("       │     ↓     │     ↓     │     ↓     │     ↓")
    print("       │    (+)    │    (+)    │    (+)    │    (+)")
    print("       └─────┴─────┴─────┴─────┴─────┴─────┴─────┘")
    print("         (shortcut connections bypass layers)")
    print("\nGradient flow: Direct paths prevent vanishing!")
    print("Even if layer gradients are small, shortcuts carry signal")


def explain_shortcut_math():
    """Explain the mathematics of shortcut connections."""
    print("\n\n=== Shortcut Connection Math ===")

    print("\n--- ASCII Diagram: Shortcut Connection Math ---")
    print("     Input (x)")
    print("        │")
    print("        ├─────────┐")
    print("        │         │")
    print("        ↓         │")
    print("    Layer(x)      │ (identity)")
    print("        │         │")
    print("        ↓         │")
    print("       (+)←───────┘")
    print("        │")
    print("        ↓")
    print("  Output = Layer(x) + x")

    print("\nThis '+' operation is the key!")
    print("It ensures gradients have multiple paths to flow back")

    print("\nMathematically:")
    print("• Forward: y = F(x) + x")
    print("• Backward: ∂L/∂x = ∂L/∂y · (∂F/∂x + 1)")
    print("• The '+1' ensures gradient doesn't vanish!")

    print("\n--- Detailed Gradient Flow Mathematics ---")
    print("\nConsider a shortcut connection: y = F(x) + x")
    print("\nDuring backpropagation:")
    print("1. We have gradient ∂L/∂y from the layer above")
    print("2. Need to compute ∂L/∂x to pass gradient down")
    print("\nUsing chain rule:")
    print("∂L/∂x = ∂L/∂y · ∂y/∂x")
    print("      = ∂L/∂y · ∂(F(x) + x)/∂x")
    print("      = ∂L/∂y · (∂F(x)/∂x + ∂x/∂x)")
    print("      = ∂L/∂y · (∂F(x)/∂x + 1)")
    print("                              ↑")
    print("                    This '+1' is crucial!")

    print("\nWhy is the '+1' important?")
    print("• Even if ∂F(x)/∂x ≈ 0 (vanishing gradient in layer)")
    print("• We still have: ∂L/∂x = ∂L/∂y · (0 + 1) = ∂L/∂y")
    print("• The gradient passes through unchanged!")
    print("• This 'gradient highway' enables training very deep networks")


def demonstrate_gradient_flow():
    """Demonstrate gradient flow with and without shortcuts."""
    print("\n\n=== Gradient Flow Demonstration ===")

    # Create a 5-layer network to demonstrate gradient flow
    layer_sizes = [3, 3, 3, 3, 3, 1]  # 5 layers: 3→3→3→3→3→1
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    # First, without shortcuts
    print("\nCreating 5-layer network WITHOUT shortcuts...")
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

    # Helper function to print gradients
    def print_gradients(model, x):
        """Helper function to visualize gradient magnitudes in each layer."""
        # Forward pass
        output = model(x)
        target = torch.tensor([[0.0]])  # Dummy target for loss calculation

        # Calculate loss (how wrong the prediction is)
        loss = nn.MSELoss()(output, target)

        # Backward pass: compute gradients
        loss.backward()

        # Print gradient magnitudes for each layer
        for name, param in model.named_parameters():
            if "weight" in name:
                # Mean absolute gradient shows how much the layer is learning
                grad_mean = param.grad.abs().mean().item()
                print(f"{name:<20} gradient magnitude: {grad_mean:.6f}")

    print("\nGradients WITHOUT shortcut connections:")
    print_gradients(model_without_shortcut, sample_input)
    print("\nNotice: Gradients get smaller in earlier layers (vanishing gradient!)")

    # Now with shortcuts
    print("\n\nCreating 5-layer network WITH shortcuts...")
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

    print("\nGradients WITH shortcut connections:")
    print_gradients(model_with_shortcut, sample_input)
    print("\nNotice: Gradients remain stable across all layers!")
    print("This enables training much deeper networks (like 12+ layer GPT)")


def visualize_transformer_shortcuts():
    """Show how shortcuts are used in transformer blocks."""
    print("\n\n=== Shortcuts in Transformer Blocks ===")

    print("\n--- ASCII Diagram: Shortcut in Transformer Block ---")
    print("     Input")
    print("       │")
    print("       ├─────────────┐")
    print("       ↓             │")
    print("  LayerNorm1         │")
    print("       ↓             │")
    print(" Multi-Head          │")
    print(" Attention           │")
    print("       ↓             │")
    print("    Dropout          │")
    print("       ↓             │")
    print("      (+)←───────────┘ (1st shortcut)")
    print("       │")
    print("       ├─────────────┐")
    print("       ↓             │")
    print("  LayerNorm2         │")
    print("       ↓             │")
    print("  FeedForward        │")
    print("   Network           │")
    print("       ↓             │")
    print("    Dropout          │")
    print("       ↓             │")
    print("      (+)←───────────┘ (2nd shortcut)")
    print("       │")
    print("       ↓")
    print("    Output")

    print("\nEach transformer block has TWO shortcut connections!")
    print("This double-shortcut pattern is key to training deep transformers")


def explain_benefits():
    """Explain the benefits of shortcut connections."""
    print("\n\n=== Benefits of Shortcut Connections ===")

    print("\n1. **Gradient Flow**")
    print("   - Direct path for gradients")
    print("   - Prevents vanishing/exploding gradients")
    print("   - Enables very deep networks (100+ layers)")

    print("\n2. **Learning Dynamics**")
    print("   - Network can learn residual functions")
    print("   - Easier to learn small changes than entire transformation")
    print("   - F(x) = desired output - x (just the difference)")

    print("\n3. **Training Stability**")
    print("   - More stable optimization")
    print("   - Faster convergence")
    print("   - Less sensitive to initialization")

    print("\n4. **Feature Reuse**")
    print("   - Later layers can access earlier features directly")
    print("   - No information bottleneck")
    print("   - Efficient feature propagation")


def implementation_details():
    """Show implementation details and variations."""
    print("\n\n=== Implementation Details ===")

    print("\n**Basic Implementation:**")
    print("```python")
    print("class ResidualBlock(nn.Module):")
    print("    def __init__(self, layer):")
    print("        super().__init__()")
    print("        self.layer = layer")
    print("    ")
    print("    def forward(self, x):")
    print("        return self.layer(x) + x  # The magic line!")
    print("```")

    print("\n**Pre-norm vs Post-norm:**")
    print("\nPost-norm (original transformer):")
    print("  x → Attention → (+) → LayerNorm → Output")
    print("\nPre-norm (modern transformers like GPT):")
    print("  x → LayerNorm → Attention → (+) → Output")
    print("\nPre-norm is more stable for very deep models")

    print("\n**Variations:**")
    print("1. Weighted shortcuts: α·F(x) + β·x")
    print("2. Highway networks: gated shortcuts")
    print("3. DenseNet: connect to all previous layers")
    print("4. Stochastic depth: randomly drop layers during training")


def practical_considerations():
    """Discuss practical considerations for using shortcuts."""
    print("\n\n=== Practical Considerations ===")

    print("\n**Dimension Matching:**")
    print("- Input and output dimensions must match for x + F(x)")
    print("- If dimensions change, use projection:")
    print("  y = F(x) + Wx  (W projects x to match F(x) size)")

    print("\n**When to Use Shortcuts:**")
    print("✓ Networks deeper than 4-6 layers")
    print("✓ When training is unstable")
    print("✓ Standard for all transformers")
    print("✓ CNNs with many layers")

    print("\n**Memory Overhead:**")
    print("- Need to store input for addition")
    print("- Slight increase in memory usage")
    print("- Worth it for training stability")

    print("\n**Initialization:**")
    print("- With shortcuts, can use larger initial weights")
    print("- Network is more robust to initialization")
    print("- Still use proper schemes (Xavier/He)")


def historical_context():
    """Provide historical context for residual connections."""
    print("\n\n=== Historical Context ===")

    print("\n**Timeline:**")
    print("• 2015: ResNet introduces shortcuts for CNNs")
    print("• 2016: Highway networks for RNNs")
    print("• 2017: Transformers adopt shortcuts as standard")
    print("• Today: Every deep model uses shortcuts")

    print("\n**Impact:**")
    print("• Enabled training of 1000+ layer networks")
    print("• Made deep learning truly 'deep'")
    print("• Standard component in modern architectures")
    print("• Nobel-worthy contribution to the field")

    print("\n**Key Insight:**")
    print("'It's easier to learn a small change than the whole function'")
    print("Instead of learning H(x), learn F(x) where H(x) = F(x) + x")
    print("If H(x) ≈ x, then F(x) ≈ 0 (easier to learn!)")


def create_gradient_comparison_plot():
    """Create a visual comparison of gradient flow."""
    print("\n\n=== Gradient Flow Comparison ===")

    # Create deeper networks for better visualization
    layer_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 1]  # 8 hidden layers

    torch.manual_seed(42)
    model_no_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

    # Forward and backward pass
    x = torch.randn(32, 10)  # Batch of 32 samples
    target = torch.randn(32, 1)

    # Collect gradients for both models
    gradients_no_shortcut = []
    gradients_with_shortcut = []

    # Without shortcuts
    out = model_no_shortcut(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    for name, param in model_no_shortcut.named_parameters():
        if "weight" in name and param.grad is not None:
            gradients_no_shortcut.append(param.grad.abs().mean().item())

    # With shortcuts
    out = model_with_shortcut(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    for name, param in model_with_shortcut.named_parameters():
        if "weight" in name and param.grad is not None:
            gradients_with_shortcut.append(param.grad.abs().mean().item())

    # Display results
    print("\nGradient magnitudes by layer:")
    print(
        f"{'Layer':<10} {'Without Shortcuts':<20} {'With Shortcuts':<20} {'Ratio':<10}"
    )
    print("-" * 60)

    for i, (g_no, g_with) in enumerate(
        zip(gradients_no_shortcut, gradients_with_shortcut)
    ):
        ratio = g_with / g_no if g_no > 0 else float("inf")
        print(f"Layer {i+1:<4} {g_no:<20.6f} {g_with:<20.6f} {ratio:<10.1f}x")

    print("\nObservation: Shortcuts maintain gradient magnitude!")
    print("Early layers have much stronger gradients with shortcuts")


def exercise_4_3_separate_dropout():
    """
    Exercise 4.3 Solution: Using separate dropout parameters.

    From Chapter 4: "Change the code to specify a separate dropout value for the
    various dropout layers throughout the model architecture."
    """
    print("\n\n=== Exercise 4.3: Separate Dropout Parameters ===")

    print("\nIn the GPT model, dropout is used in three distinct places:")
    print("1. Embedding layer dropout")
    print("2. Attention dropout (in MultiHeadAttention)")
    print("3. Residual/shortcut dropout (in TransformerBlock)")

    print("\nInstead of a single 'drop_rate', we can use:")

    # Example configuration with separate dropout rates
    print("\n```python")
    print("GPT_CONFIG_SEPARATE_DROPOUT = {")
    print("    'vocab_size': 50257,")
    print("    'context_length': 1024,")
    print("    'emb_dim': 768,")
    print("    'n_heads': 12,")
    print("    'n_layers': 12,")
    print("    # Separate dropout rates instead of single 'drop_rate'")
    print("    'emb_dropout': 0.1,      # Embedding layer dropout")
    print("    'attn_dropout': 0.1,     # Attention mechanism dropout")
    print("    'resid_dropout': 0.1,    # Residual connection dropout")
    print("    'qkv_bias': False")
    print("}")
    print("```")

    print("\nModified GPTModel initialization:")
    print("```python")
    print("class GPTModel(nn.Module):")
    print("    def __init__(self, cfg):")
    print("        super().__init__()")
    print("        # Use different dropout rates")
    print("        self.drop_emb = nn.Dropout(cfg['emb_dropout'])")
    print("        # Pass attn_dropout to attention modules")
    print("        # Pass resid_dropout to transformer blocks")
    print("```")

    print("\nWhy use different dropout rates?")
    print(
        "1. **Fine-grained control**: Different parts may need different regularization"
    )
    print(
        "2. **Embedding dropout**: Often lower (0.0-0.1) to preserve input information"
    )
    print("3. **Attention dropout**: Medium (0.1-0.2) to prevent attention overfitting")
    print("4. **Residual dropout**: Can be higher (0.1-0.3) for strong regularization")

    print("\nTypical patterns:")
    print("• Embedding dropout: 0.0 - 0.1")
    print("• Attention dropout: 0.1 - 0.2")
    print("• Residual dropout: 0.1 - 0.3")
    print("• Often: emb_dropout < attn_dropout ≤ resid_dropout")

    print("\nBenefits:")
    print("• More flexible regularization")
    print("• Can tune each component separately")
    print("• Better control over model capacity")
    print("• Useful for preventing specific types of overfitting")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Shortcut (Residual) Connections")
    print("=" * 60)

    # Explain the problem
    explain_gradient_problem()

    # Visualize architectures
    visualize_architectures()

    # Explain the math
    explain_shortcut_math()

    # Demonstrate gradient flow
    demonstrate_gradient_flow()

    # Transformer shortcuts
    visualize_transformer_shortcuts()

    # Benefits
    explain_benefits()

    # Implementation
    implementation_details()

    # Practical considerations
    practical_considerations()

    # Historical context
    historical_context()

    # Gradient comparison
    create_gradient_comparison_plot()

    # Exercise 4.3
    exercise_4_3_separate_dropout()

    print("\n" + "=" * 60)
    print("Shortcuts: The highway system for gradient flow!")
    print("=" * 60)
