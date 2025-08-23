"""
Chapter 5 - Advanced Text Generation Strategies

This script demonstrates advanced text generation techniques beyond simple
greedy decoding. It covers temperature scaling, top-k sampling, and other
methods for controlling the creativity and quality of generated text.

Key concepts:
- Temperature scaling for creativity control
- Top-k sampling for quality filtering
- Top-p (nucleus) sampling
- Comparison of different decoding strategies
- Trade-offs between coherence and diversity

These techniques are essential for practical text generation applications
where you need control over the style and quality of output.

Usage: uv run python book/ch5/advanced_generation.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Configuration for our model
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


def explain_decoding_strategies():
    """Explain different text generation strategies."""
    print("=== Text Generation Strategies ===\n")

    print("Beyond simple greedy decoding, we have several strategies:")
    print()

    print("🎯 **Greedy Decoding** (Chapter 4)")
    print("   - Always pick the highest probability token")
    print("   - Deterministic (same output every time)")
    print("   - Fast and simple")
    print("   - Can be repetitive and get stuck in loops")
    print("   - Example: 'The cat sat on the mat. The cat sat on the mat...'")
    print()

    print("🎲 **Random Sampling**")
    print("   - Sample token according to probability distribution")
    print("   - Non-deterministic (different outputs each time)")
    print("   - More diverse but can be incoherent")
    print("   - May select very low-probability tokens")
    print("   - Risk of generating nonsense")
    print()

    print("🌡️  **Temperature Scaling**")
    print("   - Modify probability distribution before sampling")
    print("   - Temperature > 1: More random (creative)")
    print("   - Temperature < 1: More focused (conservative)")
    print("   - Temperature = 1: Original distribution")
    print("   - Temperature → 0: Approaches greedy decoding")
    print()

    print("🔝 **Top-k Sampling**")
    print("   - Only consider top k most likely tokens")
    print("   - Sample from this reduced vocabulary")
    print("   - Prevents very unlikely tokens")
    print("   - Balances quality and diversity")
    print("   - Common k values: 10-50")
    print()

    print("🎯 **Top-p (Nucleus) Sampling**")
    print("   - Consider tokens until cumulative probability > p")
    print("   - Vocabulary size adapts to model confidence")
    print("   - More tokens when uncertain, fewer when confident")
    print("   - Often better than top-k")
    print("   - Common p values: 0.9-0.95")


def demonstrate_temperature_effects():
    """Show how temperature affects probability distributions."""
    print("\n\n=== Temperature Effects Demonstration ===\n")

    # Create example logits for a small vocabulary
    vocab = ["the", "cat", "dog", "ran", "quickly", "slowly", "very", "and"]
    logits = torch.tensor([3.0, 2.8, 2.5, 1.2, 0.8, 0.5, 0.3, -1.0])

    print("Example scenario: Predicting next word after 'The brown fox'")
    print(f"Vocabulary: {vocab}")
    print(f"Model logits: {[f'{logit:.1f}' for logit in logits]}\n")

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    print("How temperature affects probabilities:\n")
    print(f"{'Temp':>5} | {'Top 3 Probabilities':>20} | {'Top 3 Tokens'}")
    print("-" * 50)

    for temp in temperatures:
        if temp == 0:
            # Special case for greedy (T → 0)
            probs = torch.zeros_like(logits)
            probs[torch.argmax(logits)] = 1.0
        else:
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)

        # Get top 3
        top_probs, top_indices = torch.topk(probs, 3)
        top_tokens = [vocab[i] for i in top_indices]

        prob_str = " ".join(f"{p:.3f}" for p in top_probs)
        token_str = " ".join(f"'{t}'" for t in top_tokens)

        print(f"{temp:>5.1f} | {prob_str:>20} | {token_str}")

    print("\nObservations:")
    print("- Low temperature: Very peaked (conservative, repetitive)")
    print("- High temperature: Very flat (creative, potentially incoherent)")
    print("- Temperature 1.0: Original model distribution")
    print("- Sweet spot often between 0.7-1.2 for most applications")


def visualize_temperature_effects():
    """Create a visualization of temperature effects."""
    print("\n=== Visualizing Temperature Effects ===\n")

    # Create more realistic logits
    logits = torch.tensor([4.0, 3.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    vocab = ["the", "a", "an", "this", "that", "my", "his", "her"]

    temperatures = [0.2, 0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(1, len(temperatures), figsize=(15, 3))

    for i, temp in enumerate(temperatures):
        if temp == 0:
            probs = torch.zeros_like(logits)
            probs[torch.argmax(logits)] = 1.0
        else:
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)

        axes[i].bar(range(len(vocab)), probs.numpy())
        axes[i].set_title(f"T={temp}")
        axes[i].set_xticks(range(len(vocab)))
        axes[i].set_xticklabels(vocab, rotation=45)
        axes[i].set_ylim(0, 1)

        if i == 0:
            axes[i].set_ylabel("Probability")

    plt.tight_layout()
    plt.savefig("book/ch5/temperature_effects.png", dpi=300, bbox_inches="tight")
    print(
        "Temperature effects visualization saved as 'book/ch5/temperature_effects.png'"
    )


def demonstrate_top_k_sampling():
    """Show how top-k sampling works."""
    print("\n\n=== Top-k Sampling Demonstration ===\n")

    # Create example with clear probability rankings
    vocab = ["dog", "cat", "bird", "fish", "hamster", "rabbit", "turtle", "snake"]
    logits = torch.tensor([3.0, 2.8, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])
    probs = F.softmax(logits, dim=-1)

    print("Example: Predicting animal after 'I have a pet'")
    print("All tokens with probabilities:")
    for token, prob in zip(vocab, probs):
        print(f"  '{token}': {prob:.4f}")
    print()

    k_values = [3, 5, 8]  # 8 = all tokens

    for k in k_values:
        print(f"**Top-{k} Sampling:**")

        # Get top-k
        if k >= len(logits):
            # All tokens
            top_probs = probs
            top_indices = torch.arange(len(logits))
        else:
            top_probs, top_indices = torch.topk(probs, k)
            # Renormalize to sum to 1
            top_probs = top_probs / top_probs.sum()

        print(f"  Allowed tokens: {[vocab[i] for i in top_indices]}")
        print(f"  Probabilities: {[f'{p:.4f}' for p in top_probs]}")

        # Simulate sampling
        torch.manual_seed(42)
        sample_idx = torch.multinomial(top_probs, num_samples=5, replacement=True)
        sampled_tokens = [vocab[top_indices[i]] for i in sample_idx]
        print(f"  Sample draws: {sampled_tokens}")
        print()

    print("Key insights:")
    print("- Smaller k: More focused, higher quality")
    print("- Larger k: More diverse, potentially lower quality")
    print("- k=1 is equivalent to greedy decoding")
    print("- Common practice: k=20-50 for good balance")


def demonstrate_top_p_sampling():
    """Show how top-p (nucleus) sampling works."""
    print("\n=== Top-p (Nucleus) Sampling Demonstration ===\n")

    vocab = [
        "yes",
        "no",
        "maybe",
        "probably",
        "definitely",
        "never",
        "always",
        "sometimes",
    ]

    # Two different scenarios with different confidence levels
    scenarios = [
        ("High confidence", torch.tensor([4.0, 3.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])),
        ("Low confidence", torch.tensor([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6])),
    ]

    p_value = 0.8

    for scenario_name, logits in scenarios:
        print(f"**{scenario_name} scenario:**")
        probs = F.softmax(logits, dim=-1)

        # Show all probabilities
        print("All tokens:")
        for token, prob in zip(vocab, probs):
            print(f"  '{token}': {prob:.4f}")

        # Apply top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find cutoff point
        cutoff_idx = torch.where(cumulative_probs > p_value)[0]
        if len(cutoff_idx) > 0:
            cutoff = (
                cutoff_idx[0].item() + 1
            )  # Include the token that crosses threshold
        else:
            cutoff = len(sorted_probs)

        # Get nucleus tokens
        nucleus_probs = sorted_probs[:cutoff]
        nucleus_indices = sorted_indices[:cutoff]
        nucleus_tokens = [vocab[i] for i in nucleus_indices]

        print(f"\nTop-p={p_value} nucleus:")
        print(f"  Nucleus tokens: {nucleus_tokens}")
        print(f"  Nucleus size: {len(nucleus_tokens)}")
        print(f"  Cumulative prob: {nucleus_probs.sum():.4f}")
        print()

    print("Key insights:")
    print("- High confidence → smaller nucleus (fewer options)")
    print("- Low confidence → larger nucleus (more options)")
    print("- Adapts vocabulary size to model certainty")
    print("- More principled than fixed top-k")


def implement_advanced_generate():
    """Implement and demonstrate advanced generation function."""
    print("\n=== Advanced Generation Implementation ===\n")

    print("Here's how to implement generation with temperature and top-k:")

    code = '''
def generate_advanced(model, tokenizer, start_text, max_tokens=25,
                     temperature=1.0, top_k=None):
    """Generate text with advanced sampling strategies."""
    model.eval()
    token_ids = text_to_token_ids(start_text, tokenizer)

    for _ in range(max_tokens):
        with torch.no_grad():
            # Get logits from model
            logits = model(token_ids)
            next_token_logits = logits[0, -1, :]  # Last position

            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                min_val = top_logits[-1]
                next_token_logits = torch.where(
                    next_token_logits < min_val,
                    torch.tensor(float('-inf')),
                    next_token_logits
                )

            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, keepdim=True)

            # Add to sequence
            token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

    return token_ids_to_text(token_ids, tokenizer)
'''

    print(code)
    print("\nKey implementation details:")
    print("- Apply top-k before temperature (common practice)")
    print("- Use -inf to mask out tokens (gives 0 probability after softmax)")
    print("- torch.multinomial for sampling from probability distribution")
    print("- Temperature=0 defaults to greedy decoding")


def compare_strategies_practically():
    """Compare different generation strategies with a trained model."""
    print("\n=== Practical Strategy Comparison ===\n")

    # Note: This would work better with a trained model
    print("With a trained model, you would see these typical patterns:")
    print()

    print("**Greedy Decoding:**")
    print("Input: 'The weather today is'")
    print("Output: 'The weather today is very nice and sunny with clear skies.'")
    print("- Predictable, safe, but can be boring")
    print("- Same output every time")
    print()

    print("**High Temperature (T=1.5):**")
    print("Input: 'The weather today is'")
    print(
        "Output: 'The weather today is absolutely magnificent for skiing adventures!'"
    )
    print("- Creative and diverse")
    print("- May be less coherent or realistic")
    print()

    print("**Low Temperature (T=0.7):**")
    print("Input: 'The weather today is'")
    print("Output: 'The weather today is quite pleasant and warm.'")
    print("- More conservative than high temperature")
    print("- Still allows some variation")
    print()

    print("**Top-k=20:**")
    print("Input: 'The weather today is'")
    print("Output: 'The weather today is perfect for outdoor activities.'")
    print("- Balanced quality and diversity")
    print("- Prevents very unlikely completions")
    print()

    print("**Combined (T=0.8, k=30):**")
    print("Input: 'The weather today is'")
    print("Output: 'The weather today is wonderfully bright and cheerful.'")
    print("- Often the best of both worlds")
    print("- Most practical for real applications")


def generation_strategy_guidelines():
    """Provide practical guidelines for choosing generation strategies."""
    print("\n\n=== Practical Guidelines for Generation Strategies ===\n")

    print("**When to use each strategy:**")
    print()

    print("🎯 **Greedy Decoding:**")
    print("   - Factual question answering")
    print("   - Technical documentation")
    print("   - When consistency is crucial")
    print("   - Quick prototyping and testing")
    print()

    print("🌡️  **Low Temperature (0.3-0.7):**")
    print("   - Professional writing")
    print("   - Educational content")
    print("   - Code generation")
    print("   - When accuracy matters most")
    print()

    print("🌡️  **Medium Temperature (0.8-1.2):**")
    print("   - Creative writing")
    print("   - Conversational AI")
    print("   - Content brainstorming")
    print("   - Most general-purpose applications")
    print()

    print("🌡️  **High Temperature (1.3-2.0):**")
    print("   - Poetry and creative writing")
    print("   - Idea generation")
    print("   - When novelty is desired")
    print("   - Experimental content")
    print()

    print("🔝 **Top-k Sampling:**")
    print("   - k=10-20: Conservative, high quality")
    print("   - k=30-50: Balanced quality/diversity")
    print("   - k=100+: More diverse, potentially lower quality")
    print()

    print("🎯 **Top-p Sampling:**")
    print("   - p=0.95: Very inclusive, lots of diversity")
    print("   - p=0.9: Good balance")
    print("   - p=0.8: More focused, higher quality")
    print()

    print("**Recommended combinations:**")
    print("- General use: T=0.8, k=40")
    print("- Creative writing: T=1.1, p=0.9")
    print("- Technical content: T=0.5, k=20")
    print("- Factual answers: Greedy or T=0.3, k=10")


def advanced_techniques():
    """Discuss more advanced generation techniques."""
    print("\n=== Advanced Generation Techniques ===\n")

    print("Beyond basic sampling, there are more sophisticated approaches:")
    print()

    print("🔄 **Repetition Penalty:**")
    print("   - Reduce probability of recently used tokens")
    print("   - Prevents 'the the the...' loops")
    print("   - Typically multiply repeated token logits by 0.8-0.95")
    print()

    print("📏 **Length Penalty:**")
    print("   - Encourage longer or shorter sequences")
    print("   - Useful for controlling output length")
    print("   - Applied during beam search")
    print()

    print("🌳 **Beam Search:**")
    print("   - Keep top k sequences at each step")
    print("   - Explore multiple paths simultaneously")
    print("   - Better for tasks requiring exact answers")
    print("   - More expensive than sampling")
    print()

    print("🎭 **Contrastive Search:**")
    print("   - Balance model confidence with diversity")
    print("   - Penalize tokens too similar to previous context")
    print("   - Newer technique showing promising results")
    print()

    print("🛠️  **Constrained Generation:**")
    print("   - Force certain patterns or formats")
    print("   - Ensure valid JSON, XML, code syntax")
    print("   - Guide generation toward desired outputs")


def main():
    """Main demonstration of advanced generation strategies."""
    print("=" * 70)
    print("Chapter 5: Advanced Text Generation Strategies")
    print("=" * 70)

    # Core concepts
    explain_decoding_strategies()

    # Temperature effects
    demonstrate_temperature_effects()
    visualize_temperature_effects()

    # Sampling strategies
    demonstrate_top_k_sampling()
    demonstrate_top_p_sampling()

    # Implementation
    implement_advanced_generate()

    # Practical comparisons
    compare_strategies_practically()

    # Guidelines
    generation_strategy_guidelines()

    # Advanced techniques
    advanced_techniques()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("- Greedy decoding is simple but limited")
    print("- Temperature controls creativity vs coherence")
    print("- Top-k/top-p sampling prevents low-quality tokens")
    print("- Different strategies suit different applications")
    print("- Combining techniques often works best")
    print("- Always consider your specific use case")
    print("=" * 70)


if __name__ == "__main__":
    main()
