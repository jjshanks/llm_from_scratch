"""
Chapter 5 - Untrained Model Text Generation

This script demonstrates what happens when you try to generate text with
an untrained (randomly initialized) GPT model. It shows why training is
necessary and provides insight into the text generation process.

Key concepts:
- Random weight initialization
- Why untrained models produce gibberish
- The text generation process step-by-step
- Logits, probabilities, and token selection
- Importance of training for coherent output

This script helps build intuition for why the training process in Chapter 5
is so important for producing meaningful text.

Usage: uv run python book/ch5/untrained_text_gen.py
"""

import torch
import torch.nn.functional as F
import tiktoken
from llm_from_scratch.model import GPTModel


# Configuration for 124M parameter model
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs tensor."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs tensor to text."""
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())


def explain_random_initialization():
    """Explain why untrained models produce random output."""
    print("=== Why Untrained Models Produce Gibberish ===\n")

    print("When we initialize a GPT model, all weights are random:")
    print("- Embedding layers: random vectors for each token")
    print("- Attention weights: random query/key/value matrices")
    print("- Feed-forward weights: random transformation matrices")
    print("- Layer normalization: starts with reasonable defaults\n")

    print("This means:")
    print("- The model has no understanding of language")
    print("- No knowledge of word relationships or grammar")
    print("- No sense of context or meaning")
    print("- Predictions are essentially random\n")

    print("The model is like a newborn baby trying to speak:")
    print("- It knows the alphabet (tokens) but not words")
    print("- It can make sounds (generate tokens) but not meaning")
    print("- It needs to learn language patterns through training")


def demonstrate_random_predictions():
    """Show how random weights lead to random predictions."""
    print("\n=== Random Predictions from Untrained Model ===\n")

    # Initialize model and tokenizer
    torch.manual_seed(123)  # For reproducible "randomness"
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Test with simple input
    input_text = "The cat"
    input_ids = text_to_token_ids(input_text, tokenizer)

    print(f"Input text: '{input_text}'")
    print(f"Input token IDs: {input_ids[0].tolist()}")

    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # Focus on predictions for the last token
        last_logits = logits[0, -1, :]  # [vocab_size]

    print(f"Logits shape: {last_logits.shape}")
    print(f"Vocabulary size: {last_logits.shape[0]:,}")

    # Convert to probabilities
    probabilities = F.softmax(last_logits, dim=-1)

    # Show top predictions
    top_k = 10
    top_probs, top_indices = torch.topk(probabilities, top_k)

    print(f"\nTop {top_k} predictions for next token:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx])
        print(f"{i+1:2d}. '{token}' (ID: {idx:5d}) - Probability: {prob:.4f}")

    print(f"\nHighest probability: {top_probs[0]:.4f}")
    print("Notice how low this probability is! The model is very uncertain.")

    # Show how uniform the distribution is
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
    max_entropy = torch.log(torch.tensor(float(len(probabilities))))

    print(f"\nDistribution entropy: {entropy:.2f}")
    print(f"Maximum possible entropy: {max_entropy:.2f}")
    print(f"Relative entropy: {entropy/max_entropy:.3f}")
    print("(Close to 1.0 means nearly uniform/random distribution)")


def step_by_step_generation():
    """Show the generation process step by step."""
    print("\n\n=== Step-by-Step Generation Process ===\n")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    start_text = "Hello world"
    print(f"Starting text: '{start_text}'")

    # Convert to tokens
    token_ids = text_to_token_ids(start_text, tokenizer)
    print(f"Initial tokens: {token_ids[0].tolist()}")

    model.eval()
    max_new_tokens = 8

    print(f"\nGenerating {max_new_tokens} new tokens:\n")

    for step in range(max_new_tokens):
        print(f"--- Step {step + 1} ---")

        with torch.no_grad():
            # Forward pass
            logits = model(token_ids)
            # Get predictions for last position
            next_token_logits = logits[0, -1, :]

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Greedy selection (highest probability)
            next_token_id = torch.argmax(probs, dim=-1)

            # Show what's happening
            next_token = tokenizer.decode([next_token_id])
            prob = probs[next_token_id]

            print(f"Current sequence: {token_ids_to_text(token_ids, tokenizer)}")
            print(
                f"Next token: '{next_token}' (ID: {next_token_id}) - Prob: {prob:.4f}"
            )

            # Add to sequence
            token_ids = torch.cat(
                [token_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1
            )

    final_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\nFinal generated text: '{final_text}'")
    print("\nAs expected, this is complete gibberish!")


def compare_different_seeds():
    """Show how different random seeds produce different gibberish."""
    print("\n\n=== Different Random Seeds = Different Gibberish ===\n")

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "The quick brown"

    print(f"Starting text: '{start_text}'")
    print("Generating with different random seeds:\n")

    seeds = [42, 123, 456, 789, 999]

    for seed in seeds:
        torch.manual_seed(seed)
        model = GPTModel(GPT_CONFIG_124M)
        model.eval()

        token_ids = text_to_token_ids(start_text, tokenizer)

        # Generate 8 tokens
        with torch.no_grad():
            for _ in range(8):
                logits = model(token_ids)
                next_logits = logits[0, -1, :]
                next_id = torch.argmax(F.softmax(next_logits, dim=-1), dim=-1)
                token_ids = torch.cat(
                    [token_ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1
                )

        generated_text = token_ids_to_text(token_ids, tokenizer)
        print(f"Seed {seed:3d}: '{generated_text}'")

    print("\nEach seed produces completely different (and meaningless) text.")
    print("This confirms the model's predictions are essentially random.")


def analyze_prediction_confidence():
    """Analyze how confident untrained model is in its predictions."""
    print("\n\n=== Prediction Confidence Analysis ===\n")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    test_phrases = ["The cat", "I am", "Hello world", "Python is", "Machine learning"]

    model.eval()

    print("Confidence analysis for different input phrases:\n")

    for phrase in test_phrases:
        input_ids = text_to_token_ids(phrase, tokenizer)

        with torch.no_grad():
            logits = model(input_ids)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            # Get top prediction and its confidence
            top_prob, top_idx = torch.max(probs, dim=-1)
            top_token = tokenizer.decode([top_idx])

            # Calculate entropy (measure of uncertainty)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))

            print(f"Input: '{phrase}'")
            print(f"  Top prediction: '{top_token}' (confidence: {top_prob:.4f})")
            print(f"  Uncertainty (entropy): {entropy:.2f}")
            print()

    print("Observations:")
    print("- Very low confidence in all predictions (< 0.01 typically)")
    print("- High entropy indicates the model is very uncertain")
    print("- No clear preference for any particular next token")
    print("- This is expected behavior for random weights")


def explain_training_necessity():
    """Explain why training is necessary for good text generation."""
    print("\n\n=== Why Training is Essential ===\n")

    print("Training transforms the model from random to intelligent:")
    print()

    print("**Before Training (Random Weights):**")
    print("- Predictions are essentially random")
    print("- No understanding of language structure")
    print("- Cannot form coherent sentences")
    print("- High uncertainty in all predictions")
    print("- Output is meaningless gibberish")
    print()

    print("**After Training (Learned Weights):**")
    print("- Predictions based on learned language patterns")
    print("- Understanding of grammar, syntax, and semantics")
    print("- Can form coherent, contextually appropriate sentences")
    print("- High confidence in good predictions")
    print("- Output is meaningful and useful")
    print()

    print("**What Training Teaches the Model:**")
    print("- Which tokens commonly follow others")
    print("- Grammatical rules and sentence structure")
    print("- Semantic relationships between words")
    print("- Context-dependent word meanings")
    print("- Long-range dependencies in text")
    print()

    print("**The Training Process:**")
    print("1. Show model lots of text examples")
    print("2. Calculate loss (how wrong predictions are)")
    print("3. Adjust weights to reduce loss")
    print("4. Repeat millions of times")
    print("5. Gradually, predictions become less random and more accurate")
    print()

    print("Training is like teaching a child to speak:")
    print("- Start with babbling (random sounds/tokens)")
    print("- Through exposure and correction (loss calculation)")
    print("- Learn patterns and rules (weight updates)")
    print("- Eventually speak coherently (good text generation)")


def show_model_architecture_impact():
    """Show how model architecture affects untrained generation."""
    print("\n\n=== Model Architecture and Random Generation ===\n")

    print("Even with random weights, the model architecture matters:")
    print()

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    test_input = text_to_token_ids("Hello", tokenizer)

    # Test different model sizes
    configs = [
        (
            "Tiny",
            {
                "vocab_size": 50257,
                "context_length": 64,
                "emb_dim": 128,
                "n_heads": 4,
                "n_layers": 4,
                "drop_rate": 0.0,
                "qkv_bias": False,
            },
        ),
        (
            "Small",
            {
                "vocab_size": 50257,
                "context_length": 256,
                "emb_dim": 512,
                "n_heads": 8,
                "n_layers": 8,
                "drop_rate": 0.0,
                "qkv_bias": False,
            },
        ),
        ("Medium", GPT_CONFIG_124M),
    ]

    print("Parameter counts for different architectures:")

    for name, config in configs:
        model = GPTModel(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name:8s}: {param_count:,} parameters")

        model.eval()
        with torch.no_grad():
            logits = model(test_input)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top_prob = torch.max(probs)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))

        print(f"         Max probability: {top_prob:.4f}")
        print(f"         Entropy: {entropy:.2f}")
        print()

    print("Key insights:")
    print("- Larger models have more parameters to randomize")
    print("- All produce similarly random output when untrained")
    print("- Architecture affects computation, not meaningfulness (yet)")
    print("- Training is what makes the architecture useful")


def main():
    """Main demonstration of untrained model text generation."""
    print("=" * 70)
    print("Chapter 5: Untrained Model Text Generation")
    print("=" * 70)

    # Explain the concept
    explain_random_initialization()

    # Show random predictions
    demonstrate_random_predictions()

    # Step-by-step generation
    step_by_step_generation()

    # Different seeds
    compare_different_seeds()

    # Confidence analysis
    analyze_prediction_confidence()

    # Why training matters
    explain_training_necessity()

    # Architecture effects
    show_model_architecture_impact()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("- Untrained models produce meaningless gibberish")
    print("- Random weights lead to random, low-confidence predictions")
    print("- Training transforms random behavior into intelligent text generation")
    print(
        "- The architecture provides the capacity, but training provides the intelligence"
    )
    print("- Next: Learn how training creates this transformation!")
    print("=" * 70)


if __name__ == "__main__":
    main()
