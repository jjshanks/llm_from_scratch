"""
Chapter 4 - Layer Normalization

This script demonstrates layer normalization, a crucial technique for training
deep neural networks. It helps prevent vanishing/exploding gradients by
normalizing activations to have zero mean and unit variance.

Key concepts demonstrated:
- Why normalization is needed in deep networks
- Step-by-step normalization process
- Comparison before and after normalization
- Using PyTorch's LayerNorm module
- Learnable scale and shift parameters

Layer normalization is applied to each sample independently, making it
well-suited for transformer architectures where batch sizes can vary.

Usage: uv run python book/ch4/layer_normalization.py
"""

import torch
import torch.nn as nn
from llm_from_scratch.model import LayerNorm


def explain_normalization_need():
    """Explain why we need normalization in deep networks."""
    print("=== Understanding Layer Normalization ===\n")

    print("What is Layer Normalization?")
    print("- A technique that standardizes the inputs to each layer")
    print("- For each sample, it makes the activations have mean=0 and variance=1")
    print("- Like adjusting grades on a curve - everyone's scores get redistributed")

    print("\nWhy normalize? Deep networks can suffer from:")
    print("- Vanishing gradients: When values get very small (0.001 × 0.001 × ... → 0)")
    print("- Exploding gradients: When values get very large (100 × 100 × ... → ∞)")

    print("\nHow does normalization help?")
    print("- Keeps values centered around 0 with variance 1")
    print("- Prevents values from drifting too high or too low")
    print("- Like a thermostat keeping temperature stable")


def demonstrate_relu_problem():
    """Show how ReLU can create normalization challenges."""
    print("\n\n=== Example: Why We Need Normalization ===")

    print("\nFirst, let's understand ReLU:")
    print("- ReLU = Rectified Linear Unit")
    print("- Simple rule: if input < 0, output = 0; otherwise output = input")
    print("- Like a gate that only lets positive signals through")

    # Create example data
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)  # 2 samples, 5 features each

    # Apply a layer with ReLU
    layer = nn.Sequential(
        nn.Linear(5, 6),  # Transform 5 inputs to 6 outputs
        nn.ReLU(),  # ReLU activation (sets negative values to 0)
    )
    out = layer(batch_example)

    print("\n--- Example: Why We Need Normalization ---")
    print("\nStep 1: Layer output (after Linear + ReLU):")
    print(out)
    print("\nNotice: ReLU set all negative values to 0.0000")
    print(
        f"The remaining values range from {out[out > 0].min().item():.4f} to {out.max().item():.4f}"
    )

    # Calculate statistics for each sample
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("\nStep 2: Check the current statistics")
    print(f"Sample 1 - Mean: {mean[0].item():.4f}, Variance: {var[0].item():.4f}")
    print(f"Sample 2 - Mean: {mean[1].item():.4f}, Variance: {var[1].item():.4f}")

    print("\nWhat is variance?")
    print("- Variance measures how spread out numbers are")
    print("- Low variance (0.02) = numbers are close together")
    print("- High variance (1.0) = numbers are spread apart")
    print("- Like measuring if students scored 88-92 (low) or 50-100 (high)")

    print("\nProblem: These values are all over the place!")
    print("We want mean=0 and variance=1 for stable training")

    return out, mean, var


def manual_normalization(out, mean, var):
    """Demonstrate manual normalization process."""
    print("\n\n=== Manual Normalization Process ===")

    print("\nStep 3: Apply normalization formula: (x - mean) / sqrt(variance)")
    print("\nThis formula:")
    print("1. Subtracts mean: Centers data around 0")
    print("2. Divides by std: Scales variance to 1")

    # Apply normalization
    out_norm = (out - mean) / torch.sqrt(var)

    print("\nNormalized values:")
    for i in range(out_norm.shape[0]):
        values_str = ", ".join([f"{x:.3f}" for x in out_norm[i].tolist()])
        print(f"Sample {i+1}: [{values_str}]")

    # Verify the normalization worked
    mean_norm = out_norm.mean(dim=-1, keepdim=True)
    var_norm = out_norm.var(dim=-1, keepdim=True)

    print("\nStep 4: Verify normalization worked")
    print(
        f"Sample 1 - Mean: {mean_norm[0].item():.1e}, Variance: {var_norm[0].item():.4f}"
    )
    print(
        f"Sample 2 - Mean: {mean_norm[1].item():.1e}, Variance: {var_norm[1].item():.4f}"
    )
    print("\n✓ Success! Mean ≈ 0 and variance ≈ 1")
    print("(The tiny mean values like 9.9e-09 are just computer rounding errors)")

    return out_norm


def explain_layernorm_improvements():
    """Explain improvements in the LayerNorm module."""
    print("\n\n=== LayerNorm Module Features ===")

    print("\nThe LayerNorm module adds two important features:")
    print("\n1. **Learnable Scale (γ)**")
    print("   - Multiplies normalized values")
    print("   - Initialized to 1.0")
    print("   - Allows model to adjust variance if needed")

    print("\n2. **Learnable Shift (β)**")
    print("   - Adds to normalized values")
    print("   - Initialized to 0.0")
    print("   - Allows model to adjust mean if needed")

    print("\nFinal formula: y = γ * (x - mean) / sqrt(variance) + β")
    print("\nWhy learnable parameters?")
    print("- Sometimes the model needs non-zero mean or non-unit variance")
    print("- These parameters let the model decide")
    print("- If not needed, model keeps γ=1 and β=0")


def use_layernorm_module():
    """Demonstrate using the LayerNorm module."""
    print("\n\n=== Using LayerNorm Module ===")

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    # Create and apply LayerNorm
    ln = LayerNorm(emb_dim=5)  # 5 = number of features to normalize
    out_ln = ln(batch_example)

    # Verify normalization
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("\nAfter LayerNorm module:")
    print("Mean:")
    print(mean)
    print("Variance:")
    print(var)

    print("\nLayerNorm parameters:")
    print(f"Scale (γ): {ln.scale.data}")
    print(f"Shift (β): {ln.shift.data}")

    print("\nNote: Initially γ=1 and β=0 (no change to normalized values)")
    print("During training, these will be adjusted by backpropagation")


def explain_biased_variance():
    """
    Explain biased vs unbiased variance calculation.

    From Chapter 4: "In our variance calculation method, we use an implementation
    detail by setting unbiased=False. For those curious about what this means..."
    """
    print("\n\n=== Understanding Biased vs Unbiased Variance ===")

    print("\nWhat is Bessel's correction?")
    print("- Statistical adjustment for sample variance")
    print("- Divides by (n-1) instead of n")
    print("- Corrects for bias when estimating population variance from a sample")

    print("\n**Unbiased variance (with Bessel's correction):**")
    print("Var = Σ(x - mean)² / (n - 1)")
    print("- Used when estimating population variance from a sample")
    print("- Provides unbiased estimate of true population variance")

    print("\n**Biased variance (without Bessel's correction):**")
    print("Var = Σ(x - mean)² / n")
    print("- Used when calculating variance of the data itself")
    print("- What we use in LayerNorm (unbiased=False)")

    print("\nWhy use biased variance in LayerNorm?")
    print("1. **Large n**: With embedding dimension n=768, difference is tiny")
    print("   - Biased: divide by 768")
    print("   - Unbiased: divide by 767")
    print("   - Difference: 0.13%")

    print("\n2. **TensorFlow compatibility**: Original GPT-2 used TensorFlow")
    print("   - TensorFlow's default is biased variance")
    print("   - Using same setting ensures weight compatibility")

    print("\n3. **Not estimating population**: We're normalizing the actual values")
    print("   - Not trying to estimate variance of unseen data")
    print("   - Just standardizing current activations")

    # Demonstrate the difference
    print("\n--- Numerical Example ---")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = x.mean()

    # Biased variance
    var_biased = ((x - mean) ** 2).sum() / len(x)

    # Unbiased variance
    var_unbiased = ((x - mean) ** 2).sum() / (len(x) - 1)

    # PyTorch functions
    var_torch_biased = x.var(unbiased=False)
    var_torch_unbiased = x.var(unbiased=True)

    print(f"\nData: {x.tolist()}")
    print(f"Mean: {mean:.2f}")
    print("\nManual calculation:")
    print(f"  Biased variance (n=5):     {var_biased:.4f}")
    print(f"  Unbiased variance (n-1=4): {var_unbiased:.4f}")
    print("\nPyTorch var() function:")
    print(f"  var(unbiased=False): {var_torch_biased:.4f}")
    print(f"  var(unbiased=True):  {var_torch_unbiased:.4f}")
    print(
        f"\nDifference: {abs(var_biased - var_unbiased):.4f} ({abs(var_biased - var_unbiased)/var_biased*100:.1f}%)"
    )

    print("\nConclusion: For LayerNorm with large dimensions,")
    print("the difference is negligible, but we use unbiased=False")
    print("for compatibility with the original GPT-2 implementation.")


def compare_normalization_types():
    """Compare different normalization techniques."""
    print("\n\n=== Comparison: LayerNorm vs BatchNorm ===")

    print("\n**Layer Normalization:**")
    print("- Normalizes across features for each sample")
    print("- Independent of batch size")
    print("- Works well for variable batch sizes")
    print("- Standard in transformers")

    print("\n**Batch Normalization:**")
    print("- Normalizes across batch for each feature")
    print("- Depends on batch statistics")
    print("- Requires different behavior for train/eval")
    print("- More common in CNNs")

    print("\nWhy LayerNorm for Transformers?")
    print("1. Works with any batch size (even 1)")
    print("2. No train/eval mode difference")
    print("3. Better for sequence data")
    print("4. More stable for attention mechanisms")


def visualize_normalization_effect():
    """Visualize the effect of normalization on distributions."""
    print("\n\n=== Visualizing Normalization Effect ===")

    # Create data with different scales
    torch.manual_seed(42)
    data_small = torch.randn(100) * 0.1  # Small scale
    data_large = torch.randn(100) * 10  # Large scale

    print("\nOriginal data statistics:")
    print(f"Small scale - Mean: {data_small.mean():.3f}, Std: {data_small.std():.3f}")
    print(f"Large scale - Mean: {data_large.mean():.3f}, Std: {data_large.std():.3f}")

    # Normalize both
    def normalize(x):
        return (x - x.mean()) / x.std()

    norm_small = normalize(data_small)
    norm_large = normalize(data_large)

    print("\nAfter normalization:")
    print(f"Small scale - Mean: {norm_small.mean():.3f}, Std: {norm_small.std():.3f}")
    print(f"Large scale - Mean: {norm_large.mean():.3f}, Std: {norm_large.std():.3f}")

    print("\nKey insight: Normalization brings different scales to same range!")
    print("This prevents any single feature from dominating due to scale")


def practical_tips():
    """Provide practical tips for using layer normalization."""
    print("\n\n=== Practical Tips for Layer Normalization ===")

    print("\n1. **Where to apply:**")
    print("   - Before attention layers")
    print("   - Before feed-forward networks")
    print("   - Sometimes after (post-norm vs pre-norm)")

    print("\n2. **Epsilon value:**")
    print("   - Small constant (1e-5) added to variance")
    print("   - Prevents division by zero")
    print("   - Usually don't need to change")

    print("\n3. **Performance impact:**")
    print("   - Small computational cost")
    print("   - Big training stability benefit")
    print("   - Essential for deep transformers")

    print("\n4. **Common mistakes:**")
    print("   - Forgetting to normalize")
    print("   - Wrong dimension for normalization")
    print("   - Not using learnable parameters")


if __name__ == "__main__":
    # Turn off scientific notation for cleaner output
    torch.set_printoptions(sci_mode=False)

    print("=" * 60)
    print("Chapter 4: Layer Normalization")
    print("=" * 60)

    # Explain the need
    explain_normalization_need()

    # Show the problem
    out, mean, var = demonstrate_relu_problem()

    # Manual normalization
    manual_normalization(out, mean, var)

    # Explain improvements
    explain_layernorm_improvements()

    # Use the module
    use_layernorm_module()

    # Explain biased variance
    explain_biased_variance()

    # Compare types
    compare_normalization_types()

    # Visualize effect
    visualize_normalization_effect()

    # Practical tips
    practical_tips()

    print("\n" + "=" * 60)
    print("Layer normalization: Essential for stable deep learning!")
    print("=" * 60)
