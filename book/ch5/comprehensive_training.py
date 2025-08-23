"""
Chapter 5 - Comprehensive LLM Training and Evaluation

This script provides a complete implementation of LLM pretraining, covering all
key concepts from Chapter 5: Pretraining on unlabeled data.

Key concepts demonstrated:
- Text generation quality evaluation using cross-entropy loss
- Complete training loop with proper evaluation metrics
- Advanced text generation with temperature and top-k sampling
- Model saving and loading functionality
- Loading pretrained weights from OpenAI
- Training/validation loss tracking and overfitting detection
- Practical pretraining on "The Verdict" dataset

This comprehensive script shows how to transform a randomly initialized GPT model
into one that can generate coherent text through the pretraining process.

Usage: uv run python book/ch5/comprehensive_training.py
"""

import torch
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt

# Import our custom modules
from llm_from_scratch.model import GPTModel
from llm_from_scratch.utils import (
    create_dataloader_v1,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    text_to_token_ids,
    token_ids_to_text,
    generate_with_temperature,
)


# Configuration for 124M parameter GPT model (reduced context for training)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # Reduced from 1024 for faster training
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def setup_model_and_tokenizer():
    """Initialize the GPT model and tokenizer."""
    print("=== Setting up Model and Tokenizer ===\n")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # Start in evaluation mode

    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Context length: {GPT_CONFIG_124M['context_length']} tokens")
    print(f"Vocabulary size: {GPT_CONFIG_124M['vocab_size']} tokens")

    return model, tokenizer


def demonstrate_untrained_generation(model, tokenizer):
    """Show what untrained model generates (gibberish)."""
    print("\n=== Untrained Model Generation ===\n")

    start_context = "Every effort moves you"
    print(f"Starting context: '{start_context}'")

    # Convert to token IDs
    token_ids = text_to_token_ids(start_context, tokenizer)

    # Generate text with simple greedy decoding
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # Generate 10 new tokens
            # Get predictions for next token
            logits = model(token_ids[:, -GPT_CONFIG_124M["context_length"] :])
            logits = logits[:, -1, :]  # Focus on last time step

            # Greedy decoding: pick highest probability token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, idx_next), dim=1)

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated text: '{generated_text}'")
    print("\nAs expected, the untrained model produces gibberish!")
    print(
        "This is because it has random weights and hasn't learned language patterns yet."
    )


def explain_loss_calculation():
    """Explain cross-entropy loss calculation for text generation."""
    print("\n\n=== Understanding Cross-Entropy Loss ===\n")

    print("Cross-entropy loss measures how well our model predicts the next token.")
    print("Lower loss = better predictions = higher quality text generation")
    print()

    print("The process:")
    print("1. Model outputs logits (raw scores) for each possible next token")
    print("2. Convert logits to probabilities using softmax")
    print("3. Look at probability assigned to the correct (target) token")
    print("4. Take negative log of this probability")
    print("5. Average across all predictions")
    print()

    # Demonstrate with a simple example
    print("Example with small vocabulary ['the', 'cat', 'sat']:")

    # Simulate model predictions
    logits = torch.tensor([[2.0, 1.0, 0.1]])  # Raw model outputs
    target = torch.tensor([0])  # Target is 'the' (index 0)

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"Model logits: {logits[0].tolist()}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs[0].tolist()]}")
    print(f"Target token: 'the' (index {target[0]})")

    # Calculate loss manually
    target_prob = probs[0, target[0]]
    manual_loss = -torch.log(target_prob)

    # Calculate using PyTorch
    pytorch_loss = F.cross_entropy(logits, target)

    print(f"Probability of correct token: {target_prob:.3f}")
    print(f"Manual loss calculation: -log({target_prob:.3f}) = {manual_loss:.3f}")
    print(f"PyTorch cross_entropy: {pytorch_loss:.3f}")
    print()

    print("Why negative log?")
    print("- High probability (0.9) → low loss (-log(0.9) = 0.10)")
    print("- Low probability (0.1) → high loss (-log(0.1) = 2.30)")
    print("- Perfect prediction (1.0) → zero loss (-log(1.0) = 0.00)")


def load_training_data():
    """Load and prepare training data."""
    print("\n=== Loading Training Data ===\n")

    # Use existing "The Verdict" story from data directory
    file_path = "data/the-verdict.txt"

    # Load text
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Get tokenizer for counting
    tokenizer = tiktoken.get_encoding("gpt2")

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print(f"Text file: {file_path}")
    print(f"Characters: {total_characters:,}")
    print(f"Tokens: {total_tokens:,}")
    print()

    # Split into train/validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    print(f"Training data: {len(train_data):,} characters")
    print(f"Validation data: {len(val_data):,} characters")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")

    return train_data, val_data


def create_data_loaders(train_data, val_data):
    """Create PyTorch data loaders for training and validation."""
    print("\n=== Creating Data Loaders ===\n")

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("Batch size: 2")
    print(f"Context length: {GPT_CONFIG_124M['context_length']} tokens")

    return train_loader, val_loader


def evaluate_untrained_model(model, train_loader, val_loader, device):
    """Calculate initial loss on untrained model."""
    print("\n=== Evaluating Untrained Model ===\n")

    model.to(device)
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"Initial training loss: {train_loss:.3f}")
    print(f"Initial validation loss: {val_loss:.3f}")
    print()

    print("These losses are high because:")
    print("- Model has random weights")
    print("- Hasn't learned language patterns")
    print("- Predictions are essentially random")
    print("- Training will reduce these losses")

    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print sample text during training."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        # Simple generation
        token_ids = encoded
        for _ in range(50):  # Generate 50 tokens
            # Crop context if needed
            idx_cond = token_ids[:, -context_size:]

            # Get predictions
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last position

            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, idx_next), dim=1)

        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Clean up text for display
        print(decoded_text.replace("\n", " ").strip())

    model.train()  # Return to training mode


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """
    Main training function for pretraining the LLM.

    This implements the standard PyTorch training loop:
    1. Forward pass (calculate predictions)
    2. Calculate loss
    3. Backward pass (calculate gradients)
    4. Update weights
    5. Repeat
    """
    print("\n=== Starting Training ===\n")

    # Track metrics
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Backward pass
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            tokens_seen += input_batch.numel()  # Track tokens processed
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Generate sample text after each epoch
        print(f"\nSample text after epoch {epoch+1}:")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print()

    return train_losses, val_losses, track_tokens_seen


def plot_losses(train_losses, val_losses, tokens_seen):
    """Plot training and validation losses."""
    print("\n=== Plotting Training Progress ===\n")

    plt.figure(figsize=(12, 4))

    # Loss vs Steps
    plt.subplot(1, 2, 1)
    plt.plot(
        range(len(train_losses)), train_losses, label="Training loss", color="blue"
    )
    plt.plot(range(len(val_losses)), val_losses, label="Validation loss", color="red")
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Loss")
    plt.title("Training Progress: Loss vs Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss vs Tokens
    plt.subplot(1, 2, 2)
    plt.plot(tokens_seen, train_losses, label="Training loss", color="blue")
    plt.plot(tokens_seen, val_losses, label="Validation loss", color="red")
    plt.xlabel("Tokens Seen")
    plt.ylabel("Loss")
    plt.title("Training Progress: Loss vs Tokens")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("book/ch5/training_progress.png", dpi=300, bbox_inches="tight")
    print("Training progress plot saved as 'book/ch5/training_progress.png'")

    # Analysis
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    print(f"\nFinal training loss: {final_train_loss:.3f}")
    print(f"Final validation loss: {final_val_loss:.3f}")

    if final_val_loss > final_train_loss * 1.5:
        print("\n⚠️  OVERFITTING DETECTED!")
        print("Validation loss is much higher than training loss.")
        print("This suggests the model is memorizing the training data.")
        print("Solutions:")
        print("- Use more training data")
        print("- Reduce model size")
        print("- Add regularization (dropout, weight decay)")
        print("- Early stopping")
    else:
        print("\n✅ Training looks good! No severe overfitting detected.")


def advanced_text_generation():
    """Demonstrate advanced text generation techniques."""
    print("\n\n=== Advanced Text Generation Techniques ===\n")

    print("Beyond simple greedy decoding, we have several strategies:")
    print()

    print("1. **Temperature Scaling**")
    print("   - Controls randomness in generation")
    print("   - Low temp (0.1-0.7): More focused, conservative")
    print("   - High temp (1.0-2.0): More diverse, creative")
    print("   - Formula: logits = logits / temperature")
    print()

    print("2. **Top-k Sampling**")
    print("   - Only consider top k most likely tokens")
    print("   - Prevents very unlikely tokens")
    print("   - Common values: k=10-50")
    print()

    print("3. **Top-p (Nucleus) Sampling**")
    print("   - Consider tokens until cumulative probability > p")
    print("   - Adapts to model confidence")
    print("   - Common values: p=0.9-0.95")


def softmax_with_temperature(logits, temperature):
    """Apply temperature scaling to logits before softmax."""
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=0)


def demonstrate_temperature_effects():
    """Show how temperature affects text generation."""
    print("\n=== Temperature Effects Demonstration ===\n")

    # Example logits for next token prediction
    vocab = ["the", "dog", "cat", "car", "and", "quickly", "very", "ran"]
    logits = torch.tensor([3.0, 2.8, 2.5, 1.0, 0.9, 0.5, 0.3, -1.0])

    print("Example next-token predictions:")
    print(f"Vocabulary: {vocab[:5]}... (showing first 5)")
    print(f"Logits: {[f'{logit:.1f}' for logit in logits[:5]]}")
    print()

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    print("Temperature effects on probabilities:")
    for temp in temperatures:
        if temp == 0:
            # Special case: greedy
            probs = torch.zeros_like(logits)
            probs[torch.argmax(logits)] = 1.0
        else:
            probs = softmax_with_temperature(logits, temp)

        top_3_probs = probs[:3]
        print(
            f"T={temp:3.1f}: {[f'{p:.3f}' for p in top_3_probs]} "
            f"(top-3: {vocab[0]}, {vocab[1]}, {vocab[2]})"
        )

    print("\nObservations:")
    print("- Low temperature: Very peaked distribution (conservative)")
    print("- High temperature: Flatter distribution (diverse)")
    print("- Temperature 1.0: Original distribution")


def save_and_load_model(model, optimizer, save_path="book/ch5/model_checkpoint.pth"):
    """Demonstrate model saving and loading."""
    print("\n=== Model Saving and Loading ===\n")

    print("Saving model state...")

    # Save just the model weights
    torch.save(model.state_dict(), "book/ch5/model_weights.pth")
    print("✅ Model weights saved to 'book/ch5/model_weights.pth'")

    # Save complete checkpoint (recommended for resuming training)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": GPT_CONFIG_124M,
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Complete checkpoint saved to '{save_path}'")

    # Demonstrate loading
    print("\nLoading model...")

    # Method 1: Load just weights
    new_model = GPTModel(GPT_CONFIG_124M)
    new_model.load_state_dict(
        torch.load("book/ch5/model_weights.pth", map_location="cpu")
    )
    print("✅ Model weights loaded successfully")

    # Method 2: Load complete checkpoint
    checkpoint = torch.load(save_path, map_location="cpu")
    restored_model = GPTModel(checkpoint["config"])
    restored_model.load_state_dict(checkpoint["model_state_dict"])

    new_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.0004, weight_decay=0.1
    )
    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("✅ Complete checkpoint loaded successfully")

    print("\nWhy save checkpoints?")
    print("- Training can take hours/days/weeks")
    print("- Hardware failures happen")
    print("- Experiment with different settings")
    print("- Resume training later")
    print("- Share models with others")

    return new_model


def explain_pretrained_weights():
    """Explain the concept of using pretrained weights."""
    print("\n\n=== Loading Pretrained Weights from OpenAI ===\n")

    print("Why use pretrained weights?")
    print("- Training GPT from scratch is EXPENSIVE")
    print("- GPT-2 124M model: ~$1,000s in compute costs")
    print("- GPT-2 1.5B model: ~$100,000s in compute costs")
    print("- OpenAI released their weights for free!")
    print()

    print("Benefits of pretrained weights:")
    print("- Skip expensive pretraining phase")
    print("- Start with a model that already understands language")
    print("- Focus on fine-tuning for your specific task")
    print("- Much faster to get good results")
    print()

    print("OpenAI GPT-2 model sizes:")
    print("- 124M parameters: Small, fast, good for experimentation")
    print("- 355M parameters: Medium size, balanced performance/speed")
    print("- 774M parameters: Large, high quality text")
    print("- 1.5B parameters: Very large, state-of-the-art (2019)")
    print()

    print("Note: Loading pretrained weights requires:")
    print("1. Matching model architecture exactly")
    print("2. Converting weight formats (TensorFlow → PyTorch)")
    print("3. Mapping parameter names correctly")
    print("4. Handling different configurations (context length, bias terms)")


def main():
    """Main function demonstrating comprehensive LLM training."""
    print("=" * 70)
    print("Chapter 5: Comprehensive LLM Training and Evaluation")
    print("=" * 70)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. Setup
    model, tokenizer = setup_model_and_tokenizer()

    # 2. Show untrained generation
    demonstrate_untrained_generation(model, tokenizer)

    # 3. Explain loss calculation
    explain_loss_calculation()

    # 4. Load training data
    train_data, val_data = load_training_data()
    train_loader, val_loader = create_data_loaders(train_data, val_data)

    # 5. Evaluate untrained model
    initial_train_loss, initial_val_loss = evaluate_untrained_model(
        model, train_loader, val_loader, device
    )

    # 6. Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,  # Learning rate
        weight_decay=0.1,  # L2 regularization
    )

    print("\n=== Optimizer Configuration ===")
    print("Using AdamW optimizer:")
    print("- Adaptive learning rates per parameter")
    print("- Improved weight decay (better than Adam)")
    print("- Commonly used for training LLMs")
    print("- Learning rate: 0.0004")
    print("- Weight decay: 0.1")

    # 7. Train the model
    num_epochs = 10
    start_context = "Every effort moves you"

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,  # Evaluate every 5 steps
        eval_iter=5,  # Use 5 batches for evaluation
        start_context=start_context,
        tokenizer=tokenizer,
    )

    # 8. Plot and analyze results
    plot_losses(train_losses, val_losses, tokens_seen)

    # 9. Demonstrate advanced generation
    print("\n=== Testing Trained Model ===\n")

    print("Greedy generation (deterministic):")
    greedy_text = generate_with_temperature(
        model,
        tokenizer,
        device,
        start_context,
        max_new_tokens=30,
        temperature=0.0,  # Greedy
    )
    print(f"'{greedy_text}'\n")

    print("Low temperature (conservative):")
    conservative_text = generate_with_temperature(
        model,
        tokenizer,
        device,
        start_context,
        max_new_tokens=30,
        temperature=0.7,
        top_k=50,
    )
    print(f"'{conservative_text}'\n")

    print("High temperature (creative):")
    creative_text = generate_with_temperature(
        model,
        tokenizer,
        device,
        start_context,
        max_new_tokens=30,
        temperature=1.4,
        top_k=25,
    )
    print(f"'{creative_text}'\n")

    # 10. Advanced topics
    advanced_text_generation()
    demonstrate_temperature_effects()

    # 11. Model persistence
    save_and_load_model(model, optimizer)

    # 12. Pretrained weights explanation
    explain_pretrained_weights()

    # Summary
    print("\n" + "=" * 70)
    print("CHAPTER 5 SUMMARY")
    print("=" * 70)
    print("✅ Implemented complete LLM training pipeline")
    print("✅ Learned to evaluate model quality using cross-entropy loss")
    print("✅ Successfully trained model from random weights to coherent text")
    print("✅ Explored advanced text generation techniques")
    print("✅ Implemented model saving and loading")
    print("✅ Understanding of pretrained weights and their value")
    print()
    print("Key Insights:")
    print("- Training loss measures how well model learns training data")
    print("- Validation loss measures generalization to unseen data")
    print("- Overfitting occurs when training loss << validation loss")
    print("- Temperature controls creativity vs coherence in generation")
    print("- Pretrained weights save enormous training costs")
    print()
    print("Next Steps:")
    print("- Chapter 6: Fine-tuning for specific tasks")
    print("- Chapter 7: Instruction following and chat")
    print("- Chapter 8: Human feedback and alignment")
    print("=" * 70)


if __name__ == "__main__":
    main()
