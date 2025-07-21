"""
Chapter 4 - Activation Functions: GELU vs ReLU

This script compares GELU (Gaussian Error Linear Unit) and ReLU (Rectified
Linear Unit) activation functions, showing why modern transformers prefer GELU.

Key concepts demonstrated:
- What activation functions do and why they're needed
- ReLU: Simple thresholding at zero
- GELU: Smooth probabilistic activation
- Visual comparison of the functions
- Impact on gradient flow
- When to use each activation

Activation functions add non-linearity to neural networks. Without them,
stacking layers would just create one big linear transformation.

Usage: uv run python book/ch4/activation_functions.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from llm_from_scratch.model import GELU


def explain_activation_functions():
    """Explain the role of activation functions in neural networks."""
    print("=== Understanding Activation Functions ===\n")

    print("Why do we need activation functions?")
    print("- Neural networks without activation functions are just linear models")
    print("- No matter how many layers: Linear(Linear(x)) = Linear(x)")
    print("- Activation functions add non-linearity")
    print("- This allows networks to learn complex patterns")

    print("\nWhere are they used?")
    print("- After linear transformations")
    print("- Between layers in deep networks")
    print("- In transformer FFN: Linear → Activation → Linear")

    print("\nKey properties of good activation functions:")
    print("1. Non-linear (otherwise useless)")
    print("2. Differentiable (for backpropagation)")
    print("3. Computationally efficient")
    print("4. Don't kill gradients")


def explain_relu():
    """Explain ReLU activation function."""
    print("\n\n=== ReLU (Rectified Linear Unit) ===\n")

    print("Definition: ReLU(x) = max(0, x)")
    print("- If input < 0: output = 0")
    print("- If input ≥ 0: output = input")

    print("\nAdvantages:")
    print("✓ Very simple and fast")
    print("✓ Sparse activation (many zeros)")
    print("✓ No vanishing gradient for positive inputs")
    print("✓ Biological inspiration (neurons fire or don't)")

    print("\nDisadvantages:")
    print("✗ 'Dead ReLU' problem: neurons that always output 0")
    print("✗ Not differentiable at x=0")
    print("✗ No gradient for negative inputs")
    print("✗ Sharp transition can cause instability")

    # Demonstrate ReLU
    relu = nn.ReLU()
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    y = relu(x)

    print("\nExample:")
    print(f"Input:  {x.tolist()}")
    print(f"Output: {y.tolist()}")
    print("Notice: All negative values become 0")


def explain_gelu():
    """Explain GELU activation function."""
    print("\n\n=== GELU (Gaussian Error Linear Unit) ===\n")

    print("Definition: GELU(x) = x * Φ(x)")
    print("where Φ(x) is the cumulative distribution function of standard normal")

    print("\nIntuition:")
    print("- Smooth approximation of ReLU")
    print("- Weights inputs by their probability of being positive")
    print("- Small negative values can still pass through")

    print("\nAdvantages:")
    print("✓ Smooth and differentiable everywhere")
    print("✓ Better gradient flow")
    print("✓ Allows small negative values")
    print("✓ State-of-the-art for transformers")

    print("\nDisadvantages:")
    print("✗ More computationally expensive than ReLU")
    print("✗ Less interpretable")
    print("✗ No exact sparsity")

    # Demonstrate GELU
    gelu = GELU()
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    y = gelu(x)

    print("\nExample:")
    print(f"Input:  {[f'{v:.1f}' for v in x.tolist()]}")
    print(f"Output: {[f'{v:.3f}' for v in y.tolist()]}")
    print("Notice: Small negative values aren't completely zeroed")


def compare_functions_numerically():
    """Compare ReLU and GELU outputs numerically."""
    print("\n\n=== Numerical Comparison ===\n")

    relu = nn.ReLU()
    gelu = GELU()

    # Test points
    test_points = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    print(f"{'Input':>8} | {'ReLU':>8} | {'GELU':>8} | {'Difference':>10}")
    print("-" * 45)

    for x_val in test_points:
        x = torch.tensor(float(x_val))
        relu_out = relu(x).item()
        gelu_out = gelu(x).item()
        diff = gelu_out - relu_out

        print(f"{x_val:8.1f} | {relu_out:8.3f} | {gelu_out:8.3f} | {diff:10.3f}")

    print("\nKey observations:")
    print("- For x >> 0: GELU ≈ ReLU (both ≈ x)")
    print("- For x << 0: GELU ≈ 0 (like ReLU)")
    print("- For x ≈ 0: GELU is smoother, allows small negative values")


def visualize_activations(save_plot=False):
    """Create visualizations of ReLU and GELU."""
    print("\n\n=== Visualizing Activation Functions ===")

    # Create activation functions
    gelu, relu = GELU(), nn.ReLU()

    # Generate points to plot
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    # Create comparison plots
    plt.figure(figsize=(12, 4))

    # Plot 1: Side by side
    plt.subplot(1, 3, 1)
    plt.plot(x, y_relu, label="ReLU", linewidth=2)
    plt.plot(x, y_gelu, label="GELU", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("Output")
    plt.title("ReLU vs GELU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    # Plot 2: Difference
    plt.subplot(1, 3, 2)
    difference = y_gelu - y_relu
    plt.plot(x, difference, "g-", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("GELU(x) - ReLU(x)")
    plt.title("Difference between GELU and ReLU")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    # Plot 3: Derivatives (gradients)
    plt.subplot(1, 3, 3)
    # Compute derivatives numerically
    x_grad = x.requires_grad_(True)

    # ReLU derivative
    y_relu_grad = relu(x_grad)
    relu_grad = torch.autograd.grad(y_relu_grad.sum(), x_grad, retain_graph=True)[0]

    # GELU derivative
    x_grad = x.clone().requires_grad_(True)
    y_gelu_grad = gelu(x_grad)
    gelu_grad = torch.autograd.grad(y_gelu_grad.sum(), x_grad)[0]

    plt.plot(x.detach(), relu_grad.detach(), label="ReLU'", linewidth=2)
    plt.plot(x.detach(), gelu_grad.detach(), label="GELU'", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("Derivative")
    plt.title("Derivatives (Gradients)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()

    if save_plot:
        plt.savefig("activation_comparison.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved as 'activation_comparison.png'")
    else:
        print("\nPlots created (not displayed in terminal mode)")
        print("To see plots, uncomment plt.show() in the code")

    # Uncomment to display:
    # plt.show()

    print("\nWhat the plots show:")
    print("1. Left: GELU is a smooth version of ReLU")
    print("2. Middle: Maximum difference is around x = -0.5")
    print("3. Right: GELU has continuous gradients (better for optimization)")


def demonstrate_gradient_flow():
    """Demonstrate gradient flow through activation functions."""
    print("\n\n=== Gradient Flow Comparison ===")

    # Create a simple network with different activations
    torch.manual_seed(42)

    # Network with ReLU
    net_relu = nn.Sequential(
        nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1)
    )

    # Network with GELU
    net_gelu = nn.Sequential(
        nn.Linear(10, 50), GELU(), nn.Linear(50, 50), GELU(), nn.Linear(50, 1)
    )

    # Copy weights to make them identical
    with torch.no_grad():
        for i in [0, 2, 4]:  # Linear layer indices
            net_gelu[i].weight.copy_(net_relu[i].weight)
            net_gelu[i].bias.copy_(net_relu[i].bias)

    # Test gradient flow
    x = torch.randn(100, 10)  # 100 samples, 10 features
    target = torch.randn(100, 1)

    # Forward and backward for ReLU
    out_relu = net_relu(x)
    loss_relu = nn.MSELoss()(out_relu, target)
    loss_relu.backward()

    # Forward and backward for GELU
    out_gelu = net_gelu(x)
    loss_gelu = nn.MSELoss()(out_gelu, target)
    loss_gelu.backward()

    print("\nGradient magnitudes (first layer):")
    print(f"ReLU network: {net_relu[0].weight.grad.abs().mean().item():.6f}")
    print(f"GELU network: {net_gelu[0].weight.grad.abs().mean().item():.6f}")

    # Count dead neurons (zero gradients)
    relu_dead = (net_relu[0].weight.grad == 0).sum().item()
    gelu_dead = (net_gelu[0].weight.grad == 0).sum().item()

    print("\nDead gradients:")
    print(f"ReLU: {relu_dead} zero gradients")
    print(f"GELU: {gelu_dead} zero gradients")

    print("\nConclusion: GELU provides more consistent gradient flow!")


def practical_recommendations():
    """Provide practical recommendations for using activations."""
    print("\n\n=== Practical Recommendations ===")

    print("\n**When to use ReLU:**")
    print("- Simple feed-forward networks")
    print("- When computation speed is critical")
    print("- Computer vision tasks (CNNs)")
    print("- When you want sparse representations")

    print("\n**When to use GELU:**")
    print("- Transformer models (standard choice)")
    print("- Natural language processing")
    print("- When smooth gradients matter")
    print("- State-of-the-art performance needed")

    print("\n**Other activation options:**")
    print("- Leaky ReLU: Allows small negative slope")
    print("- ELU: Exponential linear unit")
    print("- Swish/SiLU: x * sigmoid(x)")
    print("- Mish: x * tanh(softplus(x))")

    print("\n**Tips:**")
    print("1. GELU is now standard for transformers")
    print("2. Don't mix activation types within a model")
    print("3. Consider computational cost for deployment")
    print("4. Some activations need careful initialization")


def implementation_details():
    """Show implementation details of GELU."""
    print("\n\n=== GELU Implementation Details ===")

    print("\nExact GELU formula:")
    print("GELU(x) = x * Φ(x)")
    print("where Φ(x) = 0.5 * (1 + erf(x / √2))")

    print("\nApproximate GELU (faster):")
    print("GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")

    print("\n--- The Story Behind the Approximation ---")
    print("\nFrom Chapter 4: 'The exact version is defined as GELU(x) = x · Φ(x)...")
    print("In practice, however, it's common to implement a computationally")
    print("cheaper approximation (the original GPT-2 model was also trained")
    print("with this approximation, which was found via curve fitting).'")

    print("\nHow was this approximation found?")
    print("1. **Curve Fitting**: Researchers wanted a faster alternative to erf()")
    print("2. **Target**: Match exact GELU as closely as possible")
    print("3. **Method**: Try different functions and optimize coefficients")
    print("4. **Result**: tanh-based approximation with magic constants")

    print("\nWhy these specific constants?")
    print("• √(2/π) ≈ 0.7978845608...")
    print("• 0.044715 was found through optimization")
    print("• Minimizes maximum deviation from exact GELU")
    print("• Error typically < 0.0003 across entire range")

    print("\nWhy use tanh instead of erf?")
    print("• tanh is faster to compute on most hardware")
    print("• Already optimized in most deep learning frameworks")
    print("• Similar shape to error function")
    print("• Easier to implement in custom hardware (TPUs, etc.)")

    print("\nPyTorch provides both:")
    print("- nn.GELU() - exact version")
    print("- nn.GELU(approximate='tanh') - fast approximation")

    # Compare exact vs approximate
    x = torch.linspace(-3, 3, 7)
    gelu_exact = nn.GELU()(x)
    gelu_approx = nn.GELU(approximate="tanh")(x)

    print("\nExact vs Approximate comparison:")
    print(f"{'x':>6} | {'Exact':>8} | {'Approx':>8} | {'Diff':>8}")
    print("-" * 35)
    for i in range(len(x)):
        diff = abs(gelu_exact[i] - gelu_approx[i]).item()
        print(
            f"{x[i]:6.1f} | {gelu_exact[i]:8.4f} | {gelu_approx[i]:8.4f} | {diff:8.4f}"
        )

    print("\nThe approximation is very accurate and faster!")

    print("\n--- Mathematical Breakdown of Approximation ---")
    print("\nLet's trace through the approximation step by step:")
    print("For x = 1.0:")
    print(f"  x³ = {1.0**3}")
    print(f"  0.044715 * x³ = {0.044715 * 1.0**3:.6f}")
    print(f"  x + 0.044715 * x³ = {1.0 + 0.044715 * 1.0**3:.6f}")
    sqrt_2_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))
    print(f"  √(2/π) = {sqrt_2_pi:.6f}")
    x_transformed = sqrt_2_pi * (1.0 + 0.044715)
    print(f"  √(2/π) * (x + 0.044715 * x³) = {x_transformed:.6f}")
    tanh_result = torch.tanh(x_transformed)
    print(f"  tanh(...) = {tanh_result:.6f}")
    print(f"  1 + tanh(...) = {1 + tanh_result:.6f}")
    gelu_approx = 0.5 * 1.0 * (1 + tanh_result)
    print(f"  0.5 * x * (1 + tanh(...)) = {gelu_approx:.6f}")
    print(f"\nExact GELU(1.0) = {nn.GELU()(torch.tensor(1.0)):.6f}")
    print("The approximation is remarkably close!")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Activation Functions - GELU vs ReLU")
    print("=" * 60)

    # Explain activation functions
    explain_activation_functions()

    # ReLU details
    explain_relu()

    # GELU details
    explain_gelu()

    # Numerical comparison
    compare_functions_numerically()

    # Visual comparison
    visualize_activations(save_plot=False)

    # Gradient flow
    demonstrate_gradient_flow()

    # Recommendations
    practical_recommendations()

    # Implementation details
    implementation_details()

    print("\n" + "=" * 60)
    print("Key takeaway: GELU's smoothness improves transformer training!")
    print("=" * 60)
