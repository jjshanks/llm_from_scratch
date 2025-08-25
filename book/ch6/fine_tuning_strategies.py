"""
Chapter 6 - Exploring Fine-tuning Strategies

This script compares different strategies for fine-tuning pretrained models,
demonstrating the trade-offs between training efficiency and performance.

Key concepts demonstrated:
- Full fine-tuning vs partial fine-tuning
- Last layer only vs last N layers
- Learning rate strategies for fine-tuning
- Preventing catastrophic forgetting
- Memory and computation trade-offs
- Practical recommendations

The script explores beyond the book to show various fine-tuning approaches
and their impacts on model performance.

Usage: uv run python book/ch6/fine_tuning_strategies.py
"""

import torch
import torch.nn as nn
import tiktoken
import time
from llm_from_scratch.model import GPTModel
from llm_from_scratch.classification import (
    SpamDataset,
    calc_accuracy_loader,
)


# Configuration for GPT-2 Small
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def explain_fine_tuning_strategies():
    """Explain different approaches to fine-tuning pretrained models."""
    print("=== Fine-tuning Strategy Overview ===\n")

    print("Key questions when fine-tuning:")
    print("1. **Which layers to update?** (Freeze vs unfreeze)")
    print("2. **How much to update?** (Learning rates)")
    print("3. **What's the trade-off?** (Speed vs performance)")

    print("\n--- Common Strategies ---")

    print("\n**Strategy 1: Classification Head Only**")
    print("• Freeze: All pretrained layers")
    print("• Train: Only new classification head")
    print("• Pros: Fast, minimal compute, prevents forgetting")
    print("• Cons: Limited adaptation capability")
    print("• Use when: Data is similar to pretraining data")

    print("\n**Strategy 2: Last Layer + Head**")
    print("• Freeze: All except last transformer block")
    print("• Train: Last transformer block + classification head")
    print("• Pros: Good balance of adaptation and stability")
    print("• Cons: Moderate compute requirements")
    print("• Use when: Moderate domain shift from pretraining")

    print("\n**Strategy 3: Last N Layers + Head**")
    print("• Freeze: Early transformer blocks")
    print("• Train: Last N blocks + classification head")
    print("• Pros: High adaptation capability")
    print("• Cons: More compute, risk of overfitting")
    print("• Use when: Significant domain shift")

    print("\n**Strategy 4: Full Fine-tuning**")
    print("• Freeze: Nothing")
    print("• Train: All parameters")
    print("• Pros: Maximum adaptation capability")
    print("• Cons: Highest compute, risk of catastrophic forgetting")
    print("• Use when: Very different domain or small pretrained model")


def visualize_freezing_strategies():
    """Create ASCII visualization of different freezing strategies."""
    print("\n=== Freezing Strategy Visualization ===")

    strategies = [
        (
            "Head Only",
            [
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔧",
            ],
        ),
        (
            "Last 1 + Head",
            [
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔧",
                "🔧",
            ],
        ),
        (
            "Last 3 + Head",
            [
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔒",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
            ],
        ),
        (
            "Full Model",
            [
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
                "🔧",
            ],
        ),
    ]

    print("\nTransformer Layers:  1   2   3   4   5   6   7   8   9  10  11  12 Head")
    print("                    " + "─" * 60)

    for strategy_name, layer_states in strategies:
        layers_str = " ".join([f" {state} " for state in layer_states])
        print(f"{strategy_name:>12}: {layers_str}")

    print("\nLegend: 🔒 = Frozen (preserved)  🔧 = Trainable (updated)")

    print("\n--- Memory & Compute Requirements ---")
    print("Head Only     < Last 1 + Head < Last 3 + Head < Full Model")
    print("Fastest       <               <               < Slowest")
    print("Least memory  <               <               < Most memory")


class FineTuningExperiment:
    """Class to run and compare different fine-tuning strategies."""

    def __init__(self, model_config, device="cpu"):
        self.model_config = model_config
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.results = {}

    def setup_model_for_strategy(self, strategy_name, num_classes=2):
        """Setup model with specific freezing strategy."""
        torch.manual_seed(123)  # Consistent initialization
        model = GPTModel(self.model_config)

        # Replace output head for classification
        model.out_head = nn.Linear(self.model_config["emb_dim"], num_classes)

        # Apply freezing strategy
        if strategy_name == "head_only":
            # Freeze everything except the classification head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.out_head.parameters():
                param.requires_grad = True

        elif strategy_name == "last_1_block":
            # Freeze everything except last block + head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.trf_blocks[-1].parameters():
                param.requires_grad = True
            for param in model.final_norm.parameters():
                param.requires_grad = True
            for param in model.out_head.parameters():
                param.requires_grad = True

        elif strategy_name == "last_3_blocks":
            # Freeze everything except last 3 blocks + head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.trf_blocks[-3:].parameters():
                param.requires_grad = True
            for param in model.final_norm.parameters():
                param.requires_grad = True
            for param in model.out_head.parameters():
                param.requires_grad = True

        elif strategy_name == "full_model":
            # All parameters trainable (default)
            pass

        model.to(self.device)
        return model

    def count_trainable_params(self, model):
        """Count trainable vs frozen parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": (trainable / total) * 100,
        }

    def quick_training_test(
        self, model, train_loader, val_loader, strategy_name, epochs=2
    ):
        """Run a quick training test to compare strategies."""
        print(f"\nTesting strategy: {strategy_name}")

        # Count parameters
        param_counts = self.count_trainable_params(model)
        print(
            f"  Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_pct']:.1f}%)"
        )

        # Setup optimizer with lower learning rate for fine-tuning
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Measure training time
        start_time = time.time()

        # Quick training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for input_batch, target_batch in train_loader:
                if num_batches >= 10:  # Limit batches for speed
                    break

                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = model(input_batch)[:, -1, :]  # Last token for classification
                loss = loss_fn(logits, target_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"    Epoch {epoch+1}: Avg loss = {avg_loss:.4f}")

        training_time = time.time() - start_time

        # Evaluate accuracy
        model.eval()
        train_acc = calc_accuracy_loader(
            train_loader, model, self.device, num_batches=5
        )
        val_acc = calc_accuracy_loader(val_loader, model, self.device, num_batches=5)

        # Store results
        self.results[strategy_name] = {
            "trainable_params": param_counts["trainable"],
            "trainable_pct": param_counts["trainable_pct"],
            "training_time": training_time,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
        }

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Final train accuracy: {train_acc:.3f}")
        print(f"  Final val accuracy: {val_acc:.3f}")

        return model


def compare_strategies_empirically():
    """Empirically compare different fine-tuning strategies."""
    print("\n=== Empirical Strategy Comparison ===")

    # Check if datasets exist
    try:
        # Load small subsets for quick experimentation
        tokenizer = tiktoken.get_encoding("gpt2")

        train_dataset = SpamDataset("book/ch6/train.csv", tokenizer, max_length=50)
        val_dataset = SpamDataset("book/ch6/validation.csv", tokenizer, max_length=50)

        print(f"Loaded datasets: {len(train_dataset)} train, {len(val_dataset)} val")

    except FileNotFoundError:
        print("⚠️  Dataset files not found. Run spam_dataset_preparation.py first.")
        print("Showing theoretical comparison instead...")
        show_theoretical_comparison()
        return

    # Create data loaders with small batches for quick testing
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(
        f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches"
    )

    # Initialize experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use smaller config for faster experimentation
    small_config = GPT_CONFIG_124M.copy()
    small_config["context_length"] = 256  # Reduce for memory efficiency
    small_config["n_layers"] = 6  # Smaller model for faster testing

    experiment = FineTuningExperiment(small_config, device)

    # Test different strategies
    strategies = ["head_only", "last_1_block", "last_3_blocks", "full_model"]

    print(f"\nRunning quick training tests on {len(strategies)} strategies...")
    print("(Using reduced model and limited batches for demonstration)")

    for strategy in strategies:
        model = experiment.setup_model_for_strategy(strategy)
        experiment.quick_training_test(model, train_loader, val_loader, strategy)

    # Display results
    display_strategy_results(experiment.results)


def show_theoretical_comparison():
    """Show theoretical comparison when datasets aren't available."""
    print("\n=== Theoretical Strategy Comparison ===")

    # Simulate parameter counts for full GPT-2 124M model
    total_params = 124_000_000  # Approximate
    emb_dim = 768
    n_layers = 12

    # Estimate parameters per component
    layer_params = total_params // n_layers  # Rough estimate per layer
    head_params = emb_dim * 2  # Classification head

    strategies = {
        "Head Only": head_params,
        "Last 1 + Head": layer_params + head_params,
        "Last 3 + Head": (3 * layer_params) + head_params,
        "Full Model": total_params,
    }

    print(
        f"{'Strategy':<15} {'Trainable Params':<15} {'Percentage':<12} {'Relative Speed':<15}"
    )
    print("-" * 65)

    for strategy, params in strategies.items():
        percentage = (params / total_params) * 100
        # Speed is roughly inverse to parameter count
        rel_speed = total_params / params

        print(f"{strategy:<15} {params:>13,} {percentage:>10.1f}% {rel_speed:>13.1f}x")

    print("\nExpected trade-offs:")
    print("• Head Only: Fastest training, may underfit complex tasks")
    print("• Last 1 + Head: Good balance for most classification tasks")
    print("• Last 3 + Head: Better adaptation, higher computational cost")
    print("• Full Model: Maximum flexibility, highest resource requirements")


def display_strategy_results(results):
    """Display comparison of empirical results."""
    if not results:
        return

    print("\n=== Strategy Comparison Results ===")

    print(
        f"{'Strategy':<15} {'Trainable':<12} {'Time (s)':<10} {'Train Acc':<12} {'Val Acc':<10}"
    )
    print("-" * 70)

    for strategy, metrics in results.items():
        print(
            f"{strategy:<15} "
            f"{metrics['trainable_params']:>10,} "
            f"{metrics['training_time']:>8.1f} "
            f"{metrics['final_train_acc']:>10.3f} "
            f"{metrics['final_val_acc']:>8.3f}"
        )

    # Find best performing strategy
    best_val_strategy = max(results.keys(), key=lambda k: results[k]["final_val_acc"])
    fastest_strategy = min(results.keys(), key=lambda k: results[k]["training_time"])

    print("\n--- Key Findings ---")
    print(f"Best validation accuracy: {best_val_strategy}")
    print(f"Fastest training: {fastest_strategy}")

    # Show trade-offs
    best_val_acc = results[best_val_strategy]["final_val_acc"]
    fastest_time = results[fastest_strategy]["training_time"]

    print("\nTrade-off analysis:")
    for strategy, metrics in results.items():
        acc_diff = metrics["final_val_acc"] - best_val_acc
        time_ratio = metrics["training_time"] / fastest_time

        print(
            f"  {strategy}: {acc_diff:+.3f} accuracy, {time_ratio:.1f}x time vs fastest"
        )


def learning_rate_strategies():
    """Discuss learning rate strategies for fine-tuning."""
    print("\n=== Learning Rate Strategies ===")

    print("Fine-tuning typically uses lower learning rates than pretraining:")

    print("\n**Why Lower Learning Rates?**")
    print("• Pretrained weights are already good representations")
    print("• Large updates can destroy useful learned features")
    print("• Fine-tuning is about gentle adaptation, not learning from scratch")

    print("\n**Common Learning Rate Ranges:**")
    print("• Pretraining: 1e-3 to 3e-4 (relatively high)")
    print("• Fine-tuning: 1e-5 to 1e-4 (10x lower)")
    print("• Classification head: 1e-4 to 1e-3 (can be higher than base)")

    print("\n**Advanced Strategies:**")

    print("\n1. **Discriminative Learning Rates**")
    print("   • Different rates for different layers")
    print("   • Lower rates for early layers (more general)")
    print("   • Higher rates for later layers (more task-specific)")
    print("   Example: [1e-6, 2e-6, 4e-6, ..., 1e-4] for layers 1-12")

    print("\n2. **Gradual Unfreezing**")
    print("   • Start with frozen base model")
    print("   • Gradually unfreeze layers during training")
    print("   • Helps prevent catastrophic forgetting")

    print("\n3. **Warmup + Decay**")
    print("   • Start with very low learning rate")
    print("   • Gradually increase (warmup)")
    print("   • Then decay over training")


def catastrophic_forgetting_discussion():
    """Explain catastrophic forgetting and mitigation strategies."""
    print("\n=== Catastrophic Forgetting ===")

    print("**What is Catastrophic Forgetting?**")
    print("• Model 'forgets' pretrained knowledge when learning new tasks")
    print("• Particularly problematic with high learning rates")
    print("• Can destroy useful language representations")

    print("\n**Signs of Catastrophic Forgetting:**")
    print("• Training loss decreases but validation loss increases")
    print("• Model generates nonsensical text")
    print("• Performance degrades on original pretraining tasks")

    print("\n**Mitigation Strategies:**")

    print("\n1. **Layer Freezing** (most common)")
    print("   • Freeze early layers that capture general language")
    print("   • Only train task-specific layers")
    print("   • Preserves core linguistic knowledge")

    print("\n2. **Lower Learning Rates**")
    print("   • Gentler updates preserve pretrained features")
    print("   • 10-100x lower than pretraining rates")

    print("\n3. **Regularization Techniques**")
    print("   • Weight decay prevents large parameter changes")
    print("   • Dropout maintains robustness")
    print("   • Early stopping prevents overfitting")

    print("\n4. **Knowledge Distillation**")
    print("   • Use original model as 'teacher'")
    print("   • Maintain similar outputs on general text")
    print("   • Advanced technique for critical applications")


def practical_recommendations():
    """Provide practical recommendations for choosing strategies."""
    print("\n=== Practical Recommendations ===")

    print("**Choose your strategy based on:**")

    print("\n📊 **Dataset Size:**")
    print("• Small dataset (< 1K examples): Head only or Last 1 block")
    print("• Medium dataset (1K-10K): Last 1-3 blocks")
    print("• Large dataset (> 10K): Can use more layers or full model")

    print("\n🎯 **Task Similarity to Pretraining:**")
    print("• Similar domain (e.g., general text): Minimal fine-tuning needed")
    print("• Moderate shift (e.g., formal → informal): Last few layers")
    print("• Large shift (e.g., English → code): More extensive fine-tuning")

    print("\n💻 **Computational Resources:**")
    print("• Limited resources: Head only, smaller batches")
    print("• Moderate resources: Last 1-3 blocks")
    print("• High resources: Full model with advanced techniques")

    print("\n🎲 **Risk Tolerance:**")
    print("• Conservative (preserve pretrained model): Freeze more layers")
    print("• Aggressive (maximize task performance): Train more layers")

    print("\n**Recommended Starting Points:**")
    print("1. **Default choice**: Last transformer block + classification head")
    print("2. **If underfitting**: Add more trainable layers")
    print("3. **If overfitting**: Freeze more layers or add regularization")
    print("4. **If slow training**: Reduce trainable parameters")

    print("\n**Red Flags to Watch For:**")
    print("⚠️  Training accuracy >> Validation accuracy (overfitting)")
    print("⚠️  Model generates gibberish (catastrophic forgetting)")
    print("⚠️  Loss increases after initial decrease (learning rate too high)")
    print("⚠️  No improvement after many epochs (learning rate too low)")


def main():
    """Main function demonstrating fine-tuning strategies."""
    print("=" * 70)
    print("Chapter 6: Fine-tuning Strategies Comparison")
    print("=" * 70)

    # Step 1: Explain strategies conceptually
    explain_fine_tuning_strategies()

    # Step 2: Visualize different approaches
    visualize_freezing_strategies()

    # Step 3: Empirical comparison (if datasets available)
    compare_strategies_empirically()

    # Step 4: Learning rate strategies
    learning_rate_strategies()

    # Step 5: Catastrophic forgetting
    catastrophic_forgetting_discussion()

    # Step 6: Practical recommendations
    practical_recommendations()

    print("\n" + "=" * 70)
    print("Fine-tuning strategies analysis completed!")
    print("✓ Strategy concepts explained")
    print("✓ Parameter and compute trade-offs analyzed")
    print("✓ Empirical comparison performed")
    print("✓ Learning rate strategies covered")
    print("✓ Catastrophic forgetting prevention discussed")
    print("✓ Practical recommendations provided")
    print(
        "\nKey takeaway: Start with last block + head, then adjust based on performance"
    )
    print("Next step: Run classification_training.py for complete training example")
    print("=" * 70)


if __name__ == "__main__":
    main()
