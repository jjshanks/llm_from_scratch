"""
Chapter 4 - Text Generation Process

This script demonstrates how GPT models generate text one token at a time.
It shows the autoregressive generation process, different decoding strategies,
and why untrained models produce gibberish.

Key concepts demonstrated:
- Autoregressive generation (one token at a time)
- Converting logits to probabilities
- Greedy decoding vs sampling
- Context window management
- Why untrained models fail

Text generation is where the model comes to life, turning learned patterns
into new text sequences.

Usage: uv run python book/ch4/text_generation.py
"""

import torch
import torch.nn.functional as F
import tiktoken
from llm_from_scratch.model import GPTModel
from configuration import GPT_CONFIG_124M


def explain_text_generation():
    """Explain the text generation process."""
    print("=== Text Generation Process ===\n")

    print("GPT generates text autoregressively:")
    print("1. Start with a prompt (initial tokens)")
    print("2. Feed tokens through the model")
    print("3. Get predictions for next token")
    print("4. Choose next token from predictions")
    print("5. Append chosen token to sequence")
    print("6. Repeat until done")

    print("\nIt's like autocomplete on steroids!")
    print("Each new token depends on all previous tokens")


def visualize_generation_process():
    """ASCII visualization of the generation process."""
    print("\n\n=== Generation Process Visualization ===")

    print("\nStep 1: Start with prompt")
    print("['Hello', ',', 'I', 'am'] → Model → Predictions")
    print("                                        ↓")
    print("                              P('a')=0.15, P('the')=0.08, ...")
    print("                                        ↓")
    print("                              Choose 'a' (highest probability)")

    print("\nStep 2: Add chosen token and repeat")
    print("['Hello', ',', 'I', 'am', 'a'] → Model → Predictions")
    print("                                            ↓")
    print("                              P('student')=0.12, P('teacher')=0.10, ...")
    print("                                            ↓")
    print("                              Choose 'student'")

    print("\nStep 3: Continue...")
    print("['Hello', ',', 'I', 'am', 'a', 'student'] → Model → ...")

    print("\nThis continues until:")
    print("• Reaching max length")
    print("• Generating end-of-text token")
    print("• Manual stopping")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Simple greedy text generation.

    Args:
        model: The GPT model
        idx: Starting token IDs [batch_size, sequence_length]
        max_new_tokens: How many new tokens to generate
        context_size: Maximum context length (truncate if needed)
    """
    # Set model to evaluation mode
    model.eval()

    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions (no gradient needed for inference)
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last token's predictions
        logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Convert logits to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Pick the most likely next token (greedy decoding)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def demonstrate_generation():
    """Demonstrate text generation with an untrained model."""
    print("\n\n=== Text Generation Demonstration ===")

    # Initialize model and tokenizer
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Starting prompt
    start_context = "Hello, I am"
    print(f"\nStarting prompt: '{start_context}'")

    # Convert to token IDs
    encoded = tokenizer.encode(start_context)
    print(f"Token IDs: {encoded}")
    print(f"Number of tokens: {len(encoded)}")

    # Convert to tensor and add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # [1, num_tokens]
    print(f"Tensor shape: {encoded_tensor.shape}")

    # Put model in evaluation mode
    model.eval()

    print("\nGenerating 6 new tokens...")
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print(f"\nGenerated token IDs: {out[0].tolist()}")
    print(f"Total length: {len(out[0])} tokens (4 original + 6 new)")

    # Decode back to text
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"\nGenerated text: '{decoded_text}'")

    print("\n=== Why is the output gibberish? ===")
    print("\nThe model has random weights - it hasn't been trained yet!")
    print("It's like asking a newborn baby to write a novel.")
    print("\nAfter training (Chapter 5), the model will:")
    print("• Learn language patterns from data")
    print("• Produce coherent, meaningful text")
    print("• Complete prompts in sensible ways")


def explain_logits_to_probabilities():
    """Explain the conversion from logits to probabilities."""
    print("\n\n=== From Logits to Probabilities ===")

    print("\nLogits are raw scores from the model:")
    print("• Can be any real number (-∞ to +∞)")
    print("• Higher = model thinks token is more likely")
    print("• Not probabilities yet!")

    print("\nSoftmax converts logits to probabilities:")
    print("• All values between 0 and 1")
    print("• Sum to 1.0 (proper probability distribution)")
    print("• Formula: exp(x_i) / sum(exp(x_j))")

    # Example
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = F.softmax(logits, dim=-1)

    print("\nExample:")
    print(f"Logits: {logits.tolist()}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"Sum of probabilities: {probs.sum().item():.3f}")

    print("\n--- Important Note: Softmax is Monotonic ---")
    print("\nFrom Chapter 4: 'The softmax function is monotonic, meaning it")
    print("preserves the order of its inputs when transformed into outputs.'")

    print("\nWhat does this mean?")
    print("• If logit_A > logit_B, then prob_A > prob_B")
    print("• The highest logit always becomes the highest probability")
    print("• Order is preserved, only magnitudes change")

    # Demonstrate monotonicity
    print("\nDemonstration:")
    logits = torch.tensor([3.0, 1.0, 2.0, -1.0])
    probs = F.softmax(logits, dim=-1)

    # Get rankings
    logit_ranks = torch.argsort(logits, descending=True)
    prob_ranks = torch.argsort(probs, descending=True)

    print(f"\nLogits: {logits.tolist()}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"\nLogit ranking (indices): {logit_ranks.tolist()}")
    print(f"Probability ranking: {prob_ranks.tolist()}")
    print("Notice: Rankings are identical!")

    print("\nImplication for greedy decoding:")
    print("• argmax(logits) = argmax(softmax(logits))")
    print("• For greedy decoding, softmax is technically redundant")
    print("• We could select highest logit directly")
    print("• But we compute softmax to see actual probabilities")


def explain_decoding_strategies():
    """Explain different decoding strategies."""
    print("\n\n=== Decoding Strategies ===")

    print("\n1. **Greedy Decoding** (what we used)")
    print("   - Always pick highest probability token")
    print("   - Deterministic (same output every time)")
    print("   - Can be repetitive or get stuck")
    print("   - Fast and simple")

    print("\n2. **Random Sampling**")
    print("   - Sample according to probability distribution")
    print("   - More diverse outputs")
    print("   - Can produce low-quality tokens")
    print("   - Non-deterministic")

    print("\n3. **Top-k Sampling**")
    print("   - Only consider top k most likely tokens")
    print("   - Sample from this reduced set")
    print("   - Balances quality and diversity")
    print("   - Common k values: 10-50")

    print("\n4. **Top-p (Nucleus) Sampling**")
    print("   - Consider tokens until cumulative prob > p")
    print("   - Adapts to confidence (fewer tokens when certain)")
    print("   - Often better than top-k")
    print("   - Common p values: 0.9-0.95")

    print("\n5. **Temperature Scaling**")
    print("   - Divide logits by temperature before softmax")
    print("   - T < 1: More focused (conservative)")
    print("   - T > 1: More diverse (creative)")
    print("   - T = 1: Normal behavior")


def demonstrate_sampling_strategies():
    """Demonstrate different sampling strategies."""
    print("\n\n=== Sampling Strategy Examples ===")

    # Example logits for next token prediction
    vocab_subset = ["dog", "cat", "car", "the", "and", "xyz"]
    logits = torch.tensor([3.0, 2.8, 0.5, 1.0, 0.9, -2.0])

    print(f"\nExample vocabulary: {vocab_subset}")
    print(f"Logits: {logits.tolist()}")

    # Greedy
    probs = F.softmax(logits, dim=-1)
    greedy_idx = torch.argmax(probs)
    print(
        f"\n**Greedy**: Always picks '{vocab_subset[greedy_idx]}' (p={probs[greedy_idx]:.3f})"
    )

    # Temperature effects
    print("\n**Temperature Scaling**:")
    for temp in [0.5, 1.0, 2.0]:
        scaled_logits = logits / temp
        temp_probs = F.softmax(scaled_logits, dim=-1)
        print(f"T={temp}: ", end="")
        for word, prob in zip(vocab_subset[:3], temp_probs[:3]):
            print(f"{word}={prob:.3f} ", end="")
        print()

    # Top-k
    print("\n**Top-k Sampling (k=3)**:")
    top_k = 3
    top_probs, top_indices = torch.topk(probs, top_k)
    print(f"Consider only: {[vocab_subset[i] for i in top_indices]}")
    print(f"With probs: {[f'{p:.3f}' for p in top_probs]}")


def explain_context_window():
    """Explain context window limitations."""
    print("\n\n=== Context Window Management ===")

    print("\nGPT has a fixed context window (1024 tokens for our model)")
    print("\nWhat happens when generating long texts?")
    print("• Must truncate to fit in context window")
    print("• Keep most recent tokens, drop oldest")
    print("• Model 'forgets' earlier parts")

    print("\nExample with context_length=8:")
    print("Tokens: [A, B, C, D, E, F, G, H]       (fits)")
    print("Add I:  [A, B, C, D, E, F, G, H, I]    (too long!)")
    print("Truncate: [B, C, D, E, F, G, H, I]     (drop A)")

    print("\nImplications:")
    print("• Can't generate infinitely long texts")
    print("• Long-range dependencies are lost")
    print("• Need strategies for very long documents")


def advanced_generation_features():
    """Discuss advanced generation features."""
    print("\n\n=== Advanced Generation Features ===")

    print("\n**Beam Search**")
    print("• Keep top k sequences at each step")
    print("• Explore multiple paths simultaneously")
    print("• Better for tasks needing exact answers")

    print("\n**Repetition Penalty**")
    print("• Reduce probability of repeated tokens")
    print("• Prevents 'the the the...' loops")
    print("• Adjustable strength")

    print("\n**Length Penalty**")
    print("• Encourage longer/shorter sequences")
    print("• Useful for controlling output length")

    print("\n**Constrained Generation**")
    print("• Force certain tokens/patterns")
    print("• Useful for structured outputs")
    print("• E.g., ensuring valid JSON")

    print("\n**Early Stopping**")
    print("• Stop at punctuation or newlines")
    print("• Stop at special tokens")
    print("• Quality-based stopping")


def generation_quality_factors():
    """Discuss factors affecting generation quality."""
    print("\n\n=== Factors Affecting Generation Quality ===")

    print("\n1. **Model Training**")
    print("   - Quality of training data")
    print("   - Amount of training")
    print("   - Model size")

    print("\n2. **Prompt Engineering**")
    print("   - Clear, specific prompts")
    print("   - Providing examples")
    print("   - Setting the right context")

    print("\n3. **Decoding Parameters**")
    print("   - Temperature")
    print("   - Top-k/Top-p values")
    print("   - Repetition penalty")

    print("\n4. **Post-processing**")
    print("   - Filtering inappropriate content")
    print("   - Grammar correction")
    print("   - Fact checking")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Text Generation Process")
    print("=" * 60)

    # Explain the process
    explain_text_generation()

    # Visualize generation
    visualize_generation_process()

    # Demonstrate generation
    demonstrate_generation()

    # Explain conversions
    explain_logits_to_probabilities()

    # Decoding strategies
    explain_decoding_strategies()

    # Demonstrate strategies
    demonstrate_sampling_strategies()

    # Context window
    explain_context_window()

    # Advanced features
    advanced_generation_features()

    # Quality factors
    generation_quality_factors()

    print("\n" + "=" * 60)
    print("Next: Chapter 5 will show how to train the model")
    print("so it generates meaningful text instead of gibberish!")
    print("=" * 60)
