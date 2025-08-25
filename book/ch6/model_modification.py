"""
Chapter 6 - Modifying Pretrained Models for Classification

This script demonstrates how to adapt a pretrained GPT model for
classification tasks by replacing the output layer.

Key concepts demonstrated:
- Loading pretrained GPT weights
- Understanding model architecture for classification
- Replacing the vocabulary-sized output with class outputs
- Freezing pretrained layers to preserve knowledge
- Selective unfreezing for better performance
- Analyzing parameter counts and trainable layers

This follows Section 6.5, showing the architectural changes needed
to transform a language model into a classifier.

Usage: uv run python book/ch6/model_modification.py
"""

import torch
import tiktoken
from llm_from_scratch.model import GPTModel


# Configuration for GPT-2 Small (124M parameters)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def explain_classification_architecture():
    """Explain how language models are adapted for classification."""
    print("=== Adapting Language Models for Classification ===\n")

    print("Key architectural changes needed:")
    print("\n1. **Output Layer Replacement:**")
    print("   • Original: Maps hidden_dim → vocab_size (50,257 tokens)")
    print("   • Classification: Maps hidden_dim → num_classes (e.g., 2 for spam/ham)")
    print("   • Reduces output from 50,257 dimensions to just 2")

    print("\n2. **Input Processing:**")
    print("   • Language model: Processes all token positions")
    print("   • Classification: Typically uses only last token position")
    print("   • Or uses pooled representation (mean/max of all positions)")

    print("\n3. **Training Objective:**")
    print("   • Language model: Predict next token at every position")
    print("   • Classification: Predict single class label per sequence")
    print("   • Cross-entropy loss on classification head instead of vocabulary")

    print("\n4. **Layer Freezing Strategy:**")
    print("   • Freeze most pretrained layers to preserve language knowledge")
    print("   • Fine-tune only top layers + new classification head")
    print("   • Balances adaptation with stability")


def visualize_architecture_change():
    """Create ASCII visualization of the architectural modification."""
    print("\n=== Architecture Modification Visualization ===")

    print("\nORIGINAL GPT MODEL (Language Modeling):")
    print("┌─────────────────────────────────────────────────────┐")
    print("│                Input Token IDs                      │")
    print("│                [batch, seq_len]                     │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│           Token + Position Embeddings               │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│              12 Transformer Blocks                  │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│              Final LayerNorm                        │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│           Language Modeling Head                    │")
    print("│          [emb_dim] → [vocab_size=50257]             │")
    print("│         Output: [batch, seq_len, vocab_size]        │")
    print("└─────────────────────────────────────────────────────┘")

    print("\nMODIFIED GPT MODEL (Classification):")
    print("┌─────────────────────────────────────────────────────┐")
    print("│                Input Token IDs                      │")
    print("│                [batch, seq_len]                     │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│        Token + Position Embeddings (FROZEN)        │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│         11 Transformer Blocks (FROZEN)              │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│          Last Transformer Block (TRAINABLE)        │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│         Final LayerNorm (TRAINABLE)                 │")
    print("│            [batch, seq_len, emb_dim]                │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│            Take Last Token Only                     │")
    print("│              [:, -1, :]                             │")
    print("│             [batch, emb_dim]                        │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│         Classification Head (NEW)                   │")
    print("│           [emb_dim] → [num_classes=2]               │")
    print("│            Output: [batch, num_classes]             │")
    print("└─────────────────────────────────────────────────────┘")

    print("\nKey changes highlighted:")
    print("🔒 FROZEN: Layers that preserve pretrained knowledge")
    print("🔧 TRAINABLE: Layers that adapt to classification task")
    print("🆕 NEW: Classification head replacing language modeling head")


def initialize_base_model():
    """Initialize a GPT model with random weights for demonstration."""
    print("\n=== Initializing Base GPT Model ===")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    print("✓ GPT model initialized")
    print(
        f"  Architecture: {GPT_CONFIG_124M['n_layers']} layers, {GPT_CONFIG_124M['n_heads']} heads"
    )
    print(f"  Embedding dim: {GPT_CONFIG_124M['emb_dim']}")
    print(f"  Vocabulary size: {GPT_CONFIG_124M['vocab_size']:,}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def analyze_original_architecture(model):
    """Analyze the original model architecture before modification."""
    print("\n=== Original Model Architecture Analysis ===")

    print("Model structure:")
    for name, module in model.named_children():
        if hasattr(module, "__len__"):
            print(f"  {name}: {len(module)} modules")
        else:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {param_count:,} parameters")

    # Check output layer specifically
    print("\nOriginal output layer (out_head):")
    print(f"  Type: {type(model.out_head).__name__}")
    print(f"  Input features: {model.out_head.in_features}")
    print(f"  Output features: {model.out_head.out_features}")
    print(f"  Parameters: {sum(p.numel() for p in model.out_head.parameters()):,}")

    # Test forward pass with example input
    print("\nTesting forward pass:")
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "This is a test message"
    tokens = tokenizer.encode(sample_text)
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

    print(f"  Input shape: {input_tensor.shape}")

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print(f"  Output shape: {output.shape}")
    print(
        f"  Interpretation: [batch_size={output.shape[0]}, seq_len={output.shape[1]}, vocab_size={output.shape[2]}]"
    )

    return output


def freeze_pretrained_layers(model):
    """Freeze the pretrained layers to preserve learned representations."""
    print("\n=== Freezing Pretrained Layers ===")

    print("Freezing strategy: Preserve most pretrained knowledge")
    print("• Freeze: Token/position embeddings + first 11 transformer blocks")
    print("• Keep trainable: Last transformer block + final LayerNorm + output head")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    frozen_params = 0
    trainable_params = 0

    # Count frozen parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params += param.numel()

    print(f"\nFrozen parameters: {frozen_params:,}")

    # Now unfreeze specific layers for fine-tuning
    print("\nUnfreezing layers for fine-tuning:")

    # Unfreeze last transformer block
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
        trainable_params += param.numel()
    print(
        f"  ✓ Last transformer block ({sum(p.numel() for p in model.trf_blocks[-1].parameters()):,} params)"
    )

    # Unfreeze final layer normalization
    for param in model.final_norm.parameters():
        param.requires_grad = True
        trainable_params += param.numel()
    print(
        f"  ✓ Final LayerNorm ({sum(p.numel() for p in model.final_norm.parameters()):,} params)"
    )

    # Output head will be replaced, so we'll handle it separately

    print("\nParameter summary after freezing:")
    print(f"  Frozen: {frozen_params:,} parameters")
    print(f"  Trainable (before head replacement): {trainable_params:,} parameters")

    return model


def replace_classification_head(model, num_classes=2):
    """Replace the language modeling head with a classification head."""
    print("\n=== Replacing Output Head for Classification ===")

    print("Creating classification head:")
    print("  Task: Binary classification (spam vs not spam)")
    print(f"  Number of classes: {num_classes}")
    print(f"  Input features: {GPT_CONFIG_124M['emb_dim']} (from transformer)")
    print(f"  Output features: {num_classes} (class logits)")

    # Store old head info
    old_head = model.out_head
    old_params = sum(p.numel() for p in old_head.parameters())

    # Create new classification head
    torch.manual_seed(123)  # For reproducible initialization
    new_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes
    )

    # Replace the head
    model.out_head = new_head

    # New head is trainable by default
    new_params = sum(p.numel() for p in new_head.parameters())

    print("\nOutput layer replacement:")
    print(
        f"  Old head: Linear({old_head.in_features}, {old_head.out_features}) = {old_params:,} params"
    )
    print(
        f"  New head: Linear({new_head.in_features}, {new_head.out_features}) = {new_params:,} params"
    )
    print(
        f"  Parameter reduction: {((old_params - new_params) / old_params) * 100:.1f}%"
    )

    return model


def analyze_modified_architecture(model):
    """Analyze the model architecture after modification."""
    print("\n=== Modified Model Architecture Analysis ===")

    # Count trainable vs frozen parameters
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    print("Parameter breakdown:")
    print(f"  Total: {total_params:,} parameters")
    print(
        f"  Trainable: {trainable_params:,} parameters ({(trainable_params/total_params)*100:.1f}%)"
    )
    print(
        f"  Frozen: {frozen_params:,} parameters ({(frozen_params/total_params)*100:.1f}%)"
    )

    # Show trainable layers
    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}: {param.numel():,} params")

    # Test forward pass with new architecture
    print("\nTesting modified model:")
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_texts = ["This is a test message", "WIN £1000 CASH! Call now!"]

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            tokens = tokenizer.encode(text)
            input_tensor = torch.tensor(tokens).unsqueeze(0)

            output = model(input_tensor)

            # For classification, we typically use the last token
            last_token_output = output[:, -1, :]

            print(f'  Example {i+1}: "{text}"')
            print(f"    Input shape: {input_tensor.shape}")
            print(f"    Full output: {output.shape}")
            print(f"    Last token: {last_token_output.shape}")
            print(
                f"    Logits: [{last_token_output[0, 0].item():.3f}, {last_token_output[0, 1].item():.3f}]"
            )

            # Apply softmax to get probabilities
            probabilities = torch.softmax(last_token_output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

            class_name = "not spam" if predicted_class == 0 else "spam"
            print(
                f"    Prediction: {predicted_class} ({class_name}, {confidence:.3f} confidence)"
            )


def demonstrate_training_setup(model):
    """Demonstrate how to set up the model for training."""
    print("\n=== Training Setup Demonstration ===")

    print("Setting up optimizer for fine-tuning:")

    # Show which parameters will be updated
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    print(f"  Optimizer will update {len(trainable_params)} parameter groups")
    print(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Create optimizer (AdamW is common for fine-tuning)
    learning_rate = 5e-5  # Common fine-tuning learning rate
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    print("  Optimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print("  Weight decay: 0.01")

    # Demonstrate loss calculation
    print("\nLoss calculation for classification:")
    print("  Loss function: CrossEntropyLoss")
    print("  Input: Model logits [batch_size, num_classes]")
    print("  Target: Class labels [batch_size]")

    # Example loss calculation
    loss_fn = torch.nn.CrossEntropyLoss()

    # Simulate a batch
    batch_size = 4
    num_classes = 2

    # Fake logits and labels for demonstration
    fake_logits = torch.randn(batch_size, num_classes)
    fake_labels = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(fake_logits, fake_labels)

    print("  Example calculation:")
    print(f"    Logits shape: {fake_logits.shape}")
    print(f"    Labels shape: {fake_labels.shape}")
    print(f"    Loss: {loss.item():.4f}")

    return optimizer, loss_fn


def compare_architectures():
    """Compare the original and modified architectures side by side."""
    print("\n=== Architecture Comparison Summary ===")

    original_output_size = GPT_CONFIG_124M["vocab_size"]
    classification_output_size = 2
    emb_dim = GPT_CONFIG_124M["emb_dim"]

    # Calculate parameter counts
    original_head_params = emb_dim * original_output_size
    classification_head_params = emb_dim * classification_output_size

    print(f"{'Aspect':<25} {'Original (LM)':<20} {'Modified (Classification)':<25}")
    print("-" * 70)
    print(f"{'Task':<25} {'Language Modeling':<20} {'Text Classification':<25}")
    print(
        f"{'Output Layer Size':<25} {f'{emb_dim} → {original_output_size:,}':<20} {f'{emb_dim} → {classification_output_size}':<25}"
    )
    print(
        f"{'Output Layer Params':<25} {f'{original_head_params:,}':<20} {f'{classification_head_params:,}':<25}"
    )
    print(f"{'Output Shape':<25} {'[B, T, 50257]':<20} {'[B, 2]':<25}")
    print(
        f"{'Loss Function':<25} {'CrossEntropy (tokens)':<20} {'CrossEntropy (classes)':<25}"
    )
    print(f"{'Trainable Layers':<25} {'All layers':<20} {'Last block + head':<25}")
    print(
        f"{'Training Objective':<25} {'Next token prediction':<20} {'Class label prediction':<25}"
    )

    # Calculate efficiency gains
    param_reduction = (
        (original_head_params - classification_head_params) / original_head_params
    ) * 100

    print("\nEfficiency improvements:")
    print(f"  Output layer parameter reduction: {param_reduction:.1f}%")
    print(
        f"  Memory savings: ~{(original_output_size / classification_output_size):.0f}x less output memory"
    )
    print(
        f"  Computation savings: ~{(original_output_size / classification_output_size):.0f}x fewer output calculations"
    )


def demonstrate_inference_usage(model):
    """Demonstrate how to use the modified model for inference."""
    print("\n=== Inference Usage Demonstration ===")

    print("Using the modified model for spam classification:")

    # Example messages
    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! You've won £5000! Click here immediately!",
        "Thanks for the meeting notes. See you next week.",
        "FREE MONEY! Call 0800-WINNER now! Limited time offer!",
    ]

    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = 50  # Reasonable for SMS messages
    pad_token_id = 50256  # <|endoftext|>

    model.eval()

    print(f"\nClassifying {len(test_messages)} test messages:")

    with torch.no_grad():
        for i, message in enumerate(test_messages):
            # Tokenize and pad
            tokens = tokenizer.encode(message)

            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Pad if too short
            tokens = tokens + [pad_token_id] * (max_length - len(tokens))

            # Convert to tensor
            input_tensor = torch.tensor(tokens).unsqueeze(0)  # [1, max_length]

            # Forward pass
            logits = model(input_tensor)  # [1, max_length, 2]

            # Use last token for classification
            last_token_logits = logits[:, -1, :]  # [1, 2]

            # Get probabilities and prediction
            probabilities = torch.softmax(last_token_logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

            class_name = "NOT SPAM" if predicted_class == 0 else "SPAM"

            print(
                f"\n  Message {i+1}: \"{message[:60]}{'...' if len(message) > 60 else ''}\""
            )
            print(f"    Tokens: {len(tokenizer.encode(message))}")
            print(f"    Prediction: {class_name}")
            print(f"    Confidence: {confidence:.3f}")
            print(
                f"    Probabilities: [not spam: {probabilities[0, 0]:.3f}, spam: {probabilities[0, 1]:.3f}]"
            )


def main():
    """Main function demonstrating model modification for classification."""
    print("=" * 70)
    print("Chapter 6: Modifying Pretrained Models for Classification")
    print("=" * 70)

    # Step 1: Explain the architectural changes needed
    explain_classification_architecture()

    # Step 2: Visualize the architecture change
    visualize_architecture_change()

    # Step 3: Initialize base model
    model = initialize_base_model()

    # Step 4: Analyze original architecture
    analyze_original_architecture(model)

    # Step 5: Freeze pretrained layers
    model = freeze_pretrained_layers(model)

    # Step 6: Replace classification head
    model = replace_classification_head(model, num_classes=2)

    # Step 7: Analyze modified architecture
    analyze_modified_architecture(model)

    # Step 8: Set up for training
    optimizer, loss_fn = demonstrate_training_setup(model)

    # Step 9: Compare architectures
    compare_architectures()

    # Step 10: Demonstrate inference usage
    demonstrate_inference_usage(model)

    print("\n" + "=" * 70)
    print("Model modification completed successfully!")
    print("✓ Base GPT model initialized")
    print("✓ Pretrained layers frozen to preserve knowledge")
    print("✓ Classification head added for spam detection")
    print("✓ Training setup configured")
    print("✓ Inference usage demonstrated")
    print(
        f"✓ Model ready for fine-tuning with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )
    print("\nNext step: Run fine_tuning_strategies.py to explore training approaches")
    print("=" * 70)


if __name__ == "__main__":
    main()
