"""
Chapter 6 - Complete Classification Fine-tuning

This script provides the full training loop for fine-tuning a pretrained
GPT model on the spam classification task.

Key concepts demonstrated:
- Adapting the pretraining loop for classification
- Computing classification loss with cross-entropy
- Tracking accuracy alongside loss
- Early stopping and overfitting detection
- Learning rate scheduling
- Model checkpointing
- Training visualization

Following Section 6.7, this shows how fine-tuning differs from
pretraining in terms of objectives and evaluation.

Usage: uv run python book/ch6/classification_training.py
"""

import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader

from llm_from_scratch.model import GPTModel
from llm_from_scratch.classification import (
    SpamDataset,
    calc_accuracy_loader,
    calc_loss_loader,
    extract_last_token_logits,
)
from llm_from_scratch.utils import (
    download_and_load_gpt2,
    load_weights_into_gpt,
    get_gpt2_model_configs,
)


# Configuration for GPT-2 Small (reduced context for training efficiency)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,  # Must match pretrained GPT-2 weights
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,  # Add dropout for regularization
    "qkv_bias": True,
}


def explain_classification_training():
    """Explain how classification training differs from pretraining."""
    print("=== Classification Fine-tuning vs Pretraining ===\n")

    print("Key differences in the training process:")

    print("\n**Objective Function:**")
    print("• Pretraining: Predict next token at every position")
    print("• Classification: Predict single class label per sequence")
    print("• Loss calculation: Cross-entropy on class logits vs token logits")

    print("\n**Model Output Usage:**")
    print("• Pretraining: Uses all token positions [batch_size, seq_len, vocab_size]")
    print("• Classification: Uses only last token [batch_size, num_classes]")
    print("• Why last token? It has attended to the entire sequence")

    print("\n**Evaluation Metrics:**")
    print("• Pretraining: Perplexity (exp of cross-entropy loss)")
    print("• Classification: Accuracy, precision, recall, F1-score")
    print("• Focus shifts from language quality to task performance")

    print("\n**Training Dynamics:**")
    print("• Learning rate: Typically 10-100x lower than pretraining")
    print("• Epochs: Fewer epochs (3-10 vs 100s for pretraining)")
    print("• Convergence: Much faster due to pretrained initialization")
    print("• Overfitting: Higher risk with smaller task-specific datasets")
    print("• Transfer learning: Leverages language understanding from pretraining")


def setup_model_and_tokenizer():
    """Initialize model and tokenizer for classification fine-tuning."""
    print("\n=== Setting up Model and Tokenizer ===")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load pretrained GPT-2 weights
    print("Loading pretrained GPT-2 124M model...")
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2_pretrained"
    )

    # Get model configurations and merge with our config
    model_configs = get_gpt2_model_configs()
    config = GPT_CONFIG_124M.copy()
    config.update(model_configs["gpt2-small (124M)"])

    # Initialize model with proper configuration
    torch.manual_seed(123)
    model = GPTModel(config)

    # Load pretrained weights
    load_weights_into_gpt(model, params)
    model.eval()

    print("✓ Pretrained GPT-2 model loaded successfully")
    print(f"  Architecture: {config['n_layers']} layers, {config['n_heads']} heads")
    print(f"  Context length: {config['context_length']} tokens")
    print("  Loaded from OpenAI's pretrained weights")

    # Modify for classification
    num_classes = 2  # Binary classification (spam/not spam)
    original_head_params = sum(p.numel() for p in model.out_head.parameters())

    model.out_head = nn.Linear(config["emb_dim"], num_classes)
    new_head_params = sum(p.numel() for p in model.out_head.parameters())

    print(f"  Original output layer: {original_head_params:,} parameters")
    print(f"  Classification head: {new_head_params:,} parameters")
    print(
        f"  Parameter reduction: {((original_head_params - new_head_params) / original_head_params) * 100:.1f}%"
    )

    # Apply freezing strategy (last block + head)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last transformer block
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # Unfreeze final layer norm
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Unfreeze classification head
    for param in model.out_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"  Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)"
    )
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    return model, tokenizer


def load_datasets_and_create_loaders(tokenizer, batch_size=8, max_length=120):
    """Load datasets and create data loaders."""
    print("\n=== Loading Datasets ===")

    try:
        # Load datasets
        train_dataset = SpamDataset(
            "book/ch6/train.csv", tokenizer, max_length=max_length
        )
        val_dataset = SpamDataset(
            "book/ch6/validation.csv", tokenizer, max_length=max_length
        )
        test_dataset = SpamDataset(
            "book/ch6/test.csv", tokenizer, max_length=max_length
        )

        print("✓ Loaded datasets:")
        print(f"  Training: {len(train_dataset):,} examples")
        print(f"  Validation: {len(val_dataset):,} examples")
        print(f"  Test: {len(test_dataset):,} examples")
        print(f"  Sequence length: {max_length} tokens")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    except FileNotFoundError as e:
        print(f"⚠️  Dataset files not found: {e}")
        print("Please run spam_dataset_preparation.py first to create the datasets.")
        return None, None, None


def evaluate_before_training(model, train_loader, val_loader, device):
    """Evaluate model performance before fine-tuning (pretrained but not task-specific)."""
    print("\n=== Baseline Performance (Before Fine-tuning) ===")

    model.eval()

    # Calculate initial accuracy and loss
    train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=10)

    # Calculate loss using our classification loss function
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)

    print("Initial performance (pretrained model, new classification head):")
    print(f"  Training accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"  Validation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
    print(f"  Training loss: {train_loss:.3f}")
    print(f"  Validation loss: {val_loss:.3f}")

    print("\nExpected for pretrained model with new classification head:")
    print("  Accuracy: May be close to random (~50%) initially")
    print("  But should improve much faster than random initialization")
    print("  The pretrained representations provide a strong foundation")

    if train_acc > 0.6:
        print("✓ Model already showing some task understanding!")
    elif abs(train_acc - 0.5) < 0.05:
        print("✓ Starting near random as expected with new classification head")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def create_optimizer_and_scheduler(model, learning_rate=5e-5):
    """Create optimizer and learning rate scheduler."""
    print("\n=== Training Setup ===")

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} parameter groups")

    # Use AdamW optimizer (common for transformer fine-tuning)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,  # Regularization
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,  # Will adjust based on actual steps
        eta_min=learning_rate * 0.1,
    )

    print("✓ Optimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print("  Weight decay: 0.01")
    print("✓ Scheduler: CosineAnnealingLR")

    return optimizer, scheduler


def train_classification_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
    eval_freq=50,
    early_stopping_patience=3,
):
    """
    Complete training loop for classification fine-tuning.
    """
    print("\n=== Training Loop ===")
    print(f"Training for {num_epochs} epochs...")

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    step_numbers = []

    best_val_acc = 0.0
    patience_counter = 0
    global_step = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()

        epoch_train_loss = 0
        epoch_steps = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Move to device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Get logits for last non-padding token position
            model_output = model(input_batch)  # [batch_size, seq_len, num_classes]
            logits = extract_last_token_logits(
                model_output, input_batch
            )  # [batch_size, num_classes]
            loss = loss_fn(logits, target_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_train_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                model.eval()

                # Calculate metrics on subset for speed
                train_loss_eval = calc_loss_loader(
                    train_loader, model, device, num_batches=5
                )
                val_loss_eval = calc_loss_loader(
                    val_loader, model, device, num_batches=5
                )
                train_acc_eval = calc_accuracy_loader(
                    train_loader, model, device, num_batches=5
                )
                val_acc_eval = calc_accuracy_loader(
                    val_loader, model, device, num_batches=5
                )

                # Store metrics
                train_losses.append(train_loss_eval)
                val_losses.append(val_loss_eval)
                train_accs.append(train_acc_eval)
                val_accs.append(val_acc_eval)
                step_numbers.append(global_step)

                print(
                    f"  Step {global_step:4d}: "
                    f"Train Loss {train_loss_eval:.3f}, "
                    f"Val Loss {val_loss_eval:.3f}, "
                    f"Train Acc {train_acc_eval:.3f}, "
                    f"Val Acc {val_acc_eval:.3f}"
                )

                model.train()

        # End of epoch evaluation
        model.eval()
        epoch_train_loss /= epoch_steps

        # Full evaluation
        final_train_acc = calc_accuracy_loader(train_loader, model, device)
        final_val_acc = calc_accuracy_loader(val_loader, model, device)
        final_train_loss = calc_loss_loader(train_loader, model, device)
        final_val_loss = calc_loss_loader(val_loader, model, device)

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(
            f"  Train: Loss {final_train_loss:.3f}, Acc {final_train_acc:.3f} ({final_train_acc*100:.1f}%)"
        )
        print(
            f"  Val:   Loss {final_val_loss:.3f}, Acc {final_val_acc:.3f} ({final_val_acc*100:.1f}%)"
        )
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping check
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": final_val_acc,
                    "train_acc": final_train_acc,
                },
                "best_spam_classifier.pth",
            )
            print(f"  ✓ New best validation accuracy: {best_val_acc:.3f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered after {patience_counter} epochs without improvement"
            )
            break

    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "step_numbers": step_numbers,
        "best_val_acc": best_val_acc,
        "total_time": total_time,
    }


def visualize_training_progress(history):
    """Create plots showing training progress."""
    print("\n=== Visualizing Training Progress ===")

    if not history["step_numbers"]:
        print("No training history to plot")
        return

    # Create subplots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))

    steps = history["step_numbers"]

    # Plot losses
    ax1.plot(steps, history["train_losses"], label="Training Loss", color="blue")
    ax1.plot(steps, history["val_losses"], label="Validation Loss", color="red")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(
        steps,
        [acc * 100 for acc in history["train_accuracies"]],
        label="Training Accuracy",
        color="blue",
    )
    ax2.plot(
        steps,
        [acc * 100 for acc in history["val_accuracies"]],
        label="Validation Accuracy",
        color="red",
    )
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        "book/ch6/classification_training_progress.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Training plots saved as 'book/ch6/classification_training_progress.png'")

    # Print key insights
    print("\n--- Training Insights ---")
    final_train_acc = history["train_accuracies"][-1] * 100
    final_val_acc = history["val_accuracies"][-1] * 100
    best_val_acc = history["best_val_acc"] * 100

    print(f"Final training accuracy: {final_train_acc:.1f}%")
    print(f"Final validation accuracy: {final_val_acc:.1f}%")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")

    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    if acc_gap > 10:
        print(f"⚠️  Potential overfitting detected (train-val gap: {acc_gap:.1f}%)")
        print("   Consider: more regularization, early stopping, or data augmentation")
    elif acc_gap < 2:
        print(f"✓ Good generalization (train-val gap: {acc_gap:.1f}%)")
    else:
        print(f"✓ Reasonable generalization (train-val gap: {acc_gap:.1f}%)")


def test_final_model(model, test_loader, device):
    """Evaluate the final model on the test set."""
    print("\n=== Final Test Set Evaluation ===")

    # Load best model if available
    if os.path.exists("best_spam_classifier.pth"):
        checkpoint = torch.load("best_spam_classifier.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"✓ Loaded best model from checkpoint (val acc: {checkpoint['val_acc']:.3f})"
        )
    else:
        print("Using current model state (no checkpoint found)")

    model.eval()

    # Calculate comprehensive metrics
    test_acc = calc_accuracy_loader(test_loader, model, device)
    test_loss = calc_loss_loader(test_loader, model, device)

    print("\nFinal test set performance:")
    print(f"  Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"  Test loss: {test_loss:.3f}")

    # Detailed analysis
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            input_batch = input_batch.to(device)

            model_output = model(input_batch)
            logits = extract_last_token_logits(model_output, input_batch)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.numpy())

    # Simple confusion matrix
    tp = sum(1 for p, t in zip(all_predictions, all_targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(all_predictions, all_targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(all_predictions, all_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(all_predictions, all_targets) if p == 0 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print("\nDetailed metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-score: {f1:.3f}")

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Not Spam  Spam")
    print(f"Actual Not Spam   {tn:3d}   {fp:3d}")
    print(f"       Spam       {fn:3d}   {tp:3d}")

    return {
        "accuracy": test_acc,
        "loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def demonstrate_inference(model, tokenizer, device, max_length=120):
    """Demonstrate using the trained model for inference."""
    print("\n=== Inference Demonstration ===")

    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! You've won £5000! Call now to claim your prize!",
        "Thanks for the meeting notes. See you next week.",
        "FREE MONEY! Limited time offer! Click here now!",
        "Can you pick up milk on your way home?",
        "WINNER! Claim your cash prize by calling 0800-CASH-NOW!",
    ]

    model.eval()
    pad_token_id = 50256

    print(f"Classifying {len(test_messages)} test messages:")

    with torch.no_grad():
        for i, message in enumerate(test_messages, 1):
            # Tokenize and pad
            tokens = tokenizer.encode(message)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            tokens += [pad_token_id] * (max_length - len(tokens))

            # Convert to tensor
            input_tensor = torch.tensor(tokens, device=device).unsqueeze(0)

            # Get prediction using last non-padding token
            model_output = model(input_tensor)
            logits = extract_last_token_logits(model_output, input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

            class_name = "NOT SPAM" if predicted_class == 0 else "SPAM"
            prob_spam = probabilities[0, 1].item()

            print(f"\n{i}. \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            print(f"   Prediction: {class_name}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Spam probability: {prob_spam:.3f}")


def main():
    """Main function for complete classification training demonstration."""
    print("=" * 70)
    print("Chapter 6: Complete Classification Fine-tuning")
    print("=" * 70)

    # Step 1: Explain the training process
    explain_classification_training()

    # Step 2: Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Step 3: Load datasets and create data loaders
    train_loader, val_loader, test_loader = load_datasets_and_create_loaders(tokenizer)

    if train_loader is None:
        print("Cannot proceed without datasets. Exiting...")
        return

    # Step 4: Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\nUsing device: {device}")

    # Step 5: Evaluate before training
    evaluate_before_training(model, train_loader, val_loader, device)

    # Step 6: Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, learning_rate=5e-5)

    # Step 7: Train the model
    history = train_classification_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=5,
        eval_freq=50,
        early_stopping_patience=3,
    )

    # Step 8: Visualize training progress
    visualize_training_progress(history)

    # Step 9: Test final model
    test_metrics = test_final_model(model, test_loader, device)

    # Step 10: Demonstrate inference
    demonstrate_inference(model, tokenizer, device)

    print("\n" + "=" * 70)
    print("Classification training completed successfully!")
    print("✓ Pretrained GPT-2 124M model loaded from OpenAI weights")
    print("✓ Model fine-tuned for spam classification task")
    print("✓ Training loop with proper evaluation implemented")
    print("✓ Early stopping and model checkpointing applied")
    print("✓ Training progress visualized and saved")
    print("✓ Final model tested and evaluated")
    print("✓ Inference capabilities demonstrated")
    print(f"✓ Achieved {test_metrics['accuracy']*100:.1f}% test accuracy")
    print("\nModel saved as 'best_spam_classifier.pth'")
    print("Benefits of pretrained initialization:")
    print("• Much faster convergence compared to random initialization")
    print("• Better final performance due to transfer learning")
    print("• Leverages language understanding from pretraining")
    print("Next step: Run evaluation_metrics.py for comprehensive analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
