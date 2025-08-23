"""
Chapter 5 - Text Generation Loss Calculation

This script focuses on understanding how we evaluate text generation quality
using cross-entropy loss. It demonstrates the step-by-step process of
calculating loss for generated text and why this metric is crucial for training.

Key concepts:
- Cross-entropy loss for text generation
- Converting logits to probabilities
- Why we use negative log-likelihood
- Batch processing for loss calculation
- Training vs validation loss

This is the foundation for understanding how we measure and improve
text generation quality during LLM training.

Usage: uv run python book/ch5/text_gen_loss.py
"""

import torch
import torch.nn.functional as F
from llm_from_scratch.model import GPTModel


# Small config for demonstration
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def explain_cross_entropy_loss():
    """Detailed explanation of cross-entropy loss for text generation."""
    print("=== Understanding Cross-Entropy Loss ===\n")

    print("Cross-entropy loss measures how well our model predicts the next token.")
    print("It's the standard loss function for classification tasks, including")
    print("next-token prediction in language models.\n")

    print("The Formula:")
    print("Loss = -log(P(correct_token))")
    print("Where P(correct_token) is the probability assigned to the correct token\n")

    print("Why negative log?")
    print("- Probability close to 1 → Loss close to 0 (good prediction)")
    print("- Probability close to 0 → Loss approaches infinity (bad prediction)")
    print("- Differentiable and smooth for gradient-based optimization")
    print("- Encourages the model to be confident in correct predictions\n")


def demonstrate_loss_calculation():
    """Step-by-step demonstration of loss calculation."""
    print("=== Step-by-Step Loss Calculation ===\n")

    # Create a simple example with small vocabulary
    print("Example: Predicting next word in 'The cat'")
    print("Possible next words: ['sat', 'ran', 'meowed', 'slept']\n")

    # Simulate model predictions (logits)
    logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])  # Raw model outputs for each word

    # Target: 'sat' is the correct next word (index 0)
    target = torch.tensor([0])

    print("Step 1: Model outputs logits (raw scores)")
    print(f"Logits: {logits[0].tolist()}")
    print("Higher logits = model thinks word is more likely\n")

    print("Step 2: Convert logits to probabilities using softmax")
    probabilities = F.softmax(logits, dim=-1)
    print(f"Probabilities: {[f'{p:.3f}' for p in probabilities[0]]}")
    print(f"Sum of probabilities: {probabilities.sum():.3f}\n")

    print("Step 3: Look at probability of correct token")
    correct_prob = probabilities[0, target[0]]
    print(f"Probability of correct word 'sat': {correct_prob:.3f}\n")

    print("Step 4: Calculate negative log-likelihood")
    manual_loss = -torch.log(correct_prob)
    print(f"Manual loss: -log({correct_prob:.3f}) = {manual_loss:.3f}\n")

    print("Step 5: Verify with PyTorch cross_entropy")
    pytorch_loss = F.cross_entropy(logits, target)
    print(f"PyTorch loss: {pytorch_loss:.3f}")
    print(f"Match: {abs(manual_loss - pytorch_loss) < 1e-6}\n")


def show_loss_behavior():
    """Show how loss behaves with different prediction qualities."""
    print("=== Loss Behavior Examples ===\n")

    print("How loss changes with prediction quality:\n")

    prediction_scenarios = [
        ("Perfect prediction", 0.999),
        ("Very confident", 0.9),
        ("Somewhat confident", 0.7),
        ("Uncertain", 0.5),
        ("Poor prediction", 0.1),
        ("Very poor", 0.01),
    ]

    for scenario, prob in prediction_scenarios:
        loss = -torch.log(torch.tensor(prob))
        print(f"{scenario:20s}: P={prob:5.3f} → Loss={loss:.3f}")

    print("\nKey insights:")
    print("- Perfect predictions have zero loss")
    print("- Poor predictions have high loss")
    print("- Loss increases exponentially as confidence decreases")
    print("- This encourages the model to be confident when correct\n")


def demonstrate_batch_loss():
    """Show how to calculate loss for batches of data."""
    print("=== Batch Loss Calculation ===\n")

    print("In practice, we process multiple examples at once (batches)")
    print("This is more efficient and provides better gradient estimates.\n")

    # Create batch of input sequences
    inputs = torch.tensor(
        [
            [16833, 3626, 6100],  # "every effort moves"
            [40, 1107, 588],  # "I really like"
        ]
    )

    # Targets are inputs shifted by one position
    targets = torch.tensor(
        [
            [3626, 6100, 345],  # " effort moves you"
            [1107, 588, 11311],  # " really like chocolate"
        ]
    )

    print(f"Batch size: {inputs.shape[0]}")
    print(f"Sequence length: {inputs.shape[1]}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}\n")

    # Initialize model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_SMALL)
    model.eval()

    print("Step 1: Get model predictions")
    with torch.no_grad():
        logits = model(inputs)

    print(f"Logits shape: {logits.shape}")
    print(f"[batch_size, sequence_length, vocab_size] = {list(logits.shape)}\n")

    print("Step 2: Flatten for loss calculation")
    # Cross entropy expects [N, C] and [N] shapes
    logits_flat = logits.flatten(0, 1)  # Combine batch and sequence dims
    targets_flat = targets.flatten()

    print(f"Flattened logits: {logits_flat.shape}")
    print(f"Flattened targets: {targets_flat.shape}\n")

    print("Step 3: Calculate batch loss")
    batch_loss = F.cross_entropy(logits_flat, targets_flat)
    print(f"Average loss across all predictions: {batch_loss:.3f}\n")

    print("What this means:")
    print("- Loss is averaged across all token predictions in the batch")
    print("- Lower loss = better at predicting next tokens")
    print("- This single number guides the training process")


def training_vs_validation_loss():
    """Explain the difference between training and validation loss."""
    print("=== Training vs Validation Loss ===\n")

    print("Two types of loss we track during training:\n")

    print("**Training Loss:**")
    print("- Calculated on data the model is learning from")
    print("- Used to update model weights via backpropagation")
    print("- Generally decreases as training progresses")
    print("- Can become artificially low due to memorization\n")

    print("**Validation Loss:**")
    print("- Calculated on data the model hasn't seen during training")
    print("- Not used for weight updates (no gradients)")
    print("- Measures how well model generalizes to new data")
    print("- Better indicator of true model performance\n")

    print("Relationship patterns:")
    print("✅ Both decreasing: Model is learning and generalizing")
    print("⚠️  Training << Validation: Possible overfitting")
    print("❌ Validation increasing: Model is memorizing, not learning\n")

    print("Why validation loss matters:")
    print("- Prevents overfitting to training data")
    print("- Helps with early stopping")
    print("- Better predictor of real-world performance")
    print("- Used for model selection and hyperparameter tuning")


def loss_calculation_function():
    """Show practical loss calculation functions."""
    print("=== Practical Loss Calculation Functions ===\n")

    print("Functions you'll use in training:")

    print("\n1. Single batch loss:")
    print(
        """
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
"""
    )

    print("2. Average loss over dataset:")
    print(
        """
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
"""
    )

    print("Key points:")
    print("- .item() converts tensor to Python number")
    print("- Average loss gives stable metric across different batch sizes")
    print("- num_batches parameter allows quick evaluation on subset")


def perplexity_explanation():
    """Explain perplexity as an alternative metric."""
    print("=== Perplexity: Alternative to Loss ===\n")

    print("Perplexity is another way to measure model performance:")
    print("Perplexity = exp(cross_entropy_loss)")
    print("Or: Perplexity = 2^(cross_entropy_loss_base2)\n")

    # Demonstrate relationship
    losses = [0.5, 1.0, 2.0, 3.0, 5.0]
    print("Loss → Perplexity conversion:")
    for loss in losses:
        perplexity = torch.exp(torch.tensor(loss))
        print(f"Loss {loss:.1f} → Perplexity {perplexity:.1f}")

    print("\nInterpretation:")
    print("- Lower perplexity = better model")
    print("- Perplexity of 1.0 = perfect prediction")
    print("- Perplexity of N = model is as confused as random choice from N options")
    print("- More interpretable than raw loss values")
    print("\nExample: Perplexity of 10 means the model is as uncertain")
    print("as if it were randomly choosing from 10 equally likely options.")


def main():
    """Main demonstration of text generation loss concepts."""
    print("=" * 60)
    print("Chapter 5: Text Generation Loss Calculation")
    print("=" * 60)

    # Core concepts
    explain_cross_entropy_loss()
    demonstrate_loss_calculation()
    show_loss_behavior()

    # Practical implementation
    demonstrate_batch_loss()
    training_vs_validation_loss()
    loss_calculation_function()

    # Alternative metrics
    perplexity_explanation()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- Cross-entropy loss measures prediction quality")
    print("- Lower loss = better next-token predictions")
    print("- Validation loss prevents overfitting")
    print("- Loss guides the training process")
    print("- Perplexity offers interpretable alternative")
    print("=" * 60)


if __name__ == "__main__":
    main()
