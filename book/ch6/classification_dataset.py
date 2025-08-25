"""
Chapter 6 - Creating Classification Datasets and DataLoaders

This script shows how to build PyTorch datasets and dataloaders specifically
for classification tasks, handling variable-length text inputs.

Key concepts demonstrated:
- Tokenizing text messages with padding strategies
- Finding optimal sequence lengths for the dataset
- Creating PyTorch Dataset classes for classification
- Batch processing with uniform sequence lengths
- Handling special tokens for padding
- Memory-efficient data loading

The implementation follows Section 6.3, demonstrating the differences
between pretraining and classification data preparation.

Usage: uv run python book/ch6/classification_dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
import numpy as np


def explain_classification_vs_pretraining():
    """Explain differences between classification and pretraining data loading."""
    print("=== Classification vs Pretraining Data Loading ===\n")

    print("Key differences from pretraining:")
    print("\n**Pretraining DataLoader:**")
    print("• Input: Token sequences")
    print("• Target: Next token predictions")
    print("• Format: [input_tokens] → [shifted_tokens]")
    print("• Length: Fixed context window (e.g., 256, 1024 tokens)")
    print("• Goal: Language modeling objective")

    print("\n**Classification DataLoader:**")
    print("• Input: Variable-length text messages")
    print("• Target: Single class label per message")
    print("• Format: [padded_tokens] → [class_label]")
    print("• Length: Varies by message (requires padding)")
    print("• Goal: Classification objective")

    print("\nChallenges for classification:")
    print("1. **Variable lengths**: Messages range from few words to long texts")
    print("2. **Padding strategy**: How to handle different lengths in batches")
    print("3. **Memory efficiency**: Balance between padding waste and performance")
    print("4. **Label format**: Single label per sequence vs per-token labels")


def analyze_text_lengths():
    """Analyze the length distribution of text messages."""
    print("\n=== Analyzing Text Length Distribution ===")

    # Load the datasets if they exist
    datasets = {}
    for name in ["train", "validation", "test"]:
        try:
            df = pd.read_csv(f"book/ch6/{name}.csv")
            datasets[name] = df
            print(f"✓ Loaded book/ch6/{name}.csv: {len(df):,} examples")
        except FileNotFoundError:
            print(
                f"⚠️  book/ch6/{name}.csv not found. Run spam_dataset_preparation.py first."
            )
            return None

    if not datasets:
        print("No datasets found. Please run spam_dataset_preparation.py first.")
        return None

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Analyze lengths for training set
    train_df = datasets["train"]

    print(f"\nTokenizing {len(train_df):,} training messages...")

    # Tokenize all texts and get lengths
    token_lengths = []
    char_lengths = []

    for text in train_df["Text"]:
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))
        char_lengths.append(len(text))

    # Calculate statistics
    token_stats = {
        "min": min(token_lengths),
        "max": max(token_lengths),
        "mean": np.mean(token_lengths),
        "median": np.median(token_lengths),
        "std": np.std(token_lengths),
    }

    char_stats = {
        "min": min(char_lengths),
        "max": max(char_lengths),
        "mean": np.mean(char_lengths),
        "median": np.median(char_lengths),
        "std": np.std(char_lengths),
    }

    print("\n--- Character Length Statistics ---")
    print(f"Min: {char_stats['min']} characters")
    print(f"Max: {char_stats['max']} characters")
    print(f"Mean: {char_stats['mean']:.1f} characters")
    print(f"Median: {char_stats['median']:.1f} characters")
    print(f"Std Dev: {char_stats['std']:.1f} characters")

    print("\n--- Token Length Statistics ---")
    print(f"Min: {token_stats['min']} tokens")
    print(f"Max: {token_stats['max']} tokens")
    print(f"Mean: {token_stats['mean']:.1f} tokens")
    print(f"Median: {token_stats['median']:.1f} tokens")
    print(f"Std Dev: {token_stats['std']:.1f} tokens")

    # Show percentiles for padding decisions
    percentiles = [50, 75, 90, 95, 99, 100]
    print("\n--- Token Length Percentiles ---")
    for p in percentiles:
        length = np.percentile(token_lengths, p)
        coverage = (np.array(token_lengths) <= length).mean() * 100
        print(
            f"{p:3d}th percentile: {length:3.0f} tokens (covers {coverage:.1f}% of data)"
        )

    # Recommend max_length
    recommended_length = int(np.percentile(token_lengths, 95))
    print(f"\n💡 Recommended max_length: {recommended_length} tokens")
    print("   This covers 95% of messages without truncation")
    print("   Remaining 5% will be truncated")

    return token_lengths, datasets, tokenizer


def visualize_length_distribution(token_lengths):
    """Create an ASCII histogram of token length distribution."""
    print("\n=== Token Length Distribution Visualization ===")

    # Create histogram bins
    num_bins = 20

    # Calculate histogram
    hist, bin_edges = np.histogram(token_lengths, bins=num_bins)

    # Normalize for ASCII display
    max_count = max(hist)
    max_bar_width = 50

    print("\nToken Length Histogram:")
    print("(Each █ represents ~{:.0f} messages)".format(max_count / max_bar_width))
    print()

    for i, count in enumerate(hist):
        bin_start = int(bin_edges[i])
        bin_end = int(bin_edges[i + 1])

        # Create bar
        bar_length = int((count / max_count) * max_bar_width) if max_count > 0 else 0
        bar = "█" * bar_length

        print(f"{bin_start:3d}-{bin_end:3d}: {bar} {count:,}")

    print(f"\nTotal messages: {len(token_lengths):,}")


def demonstrate_padding_strategies():
    """Demonstrate different padding strategies and their trade-offs."""
    print("\n=== Padding Strategy Comparison ===")

    # Example messages of different lengths
    tokenizer = tiktoken.get_encoding("gpt2")
    example_messages = [
        "Ok",  # Very short
        "Thanks for the update!",  # Short
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",  # Medium
        "CONGRATULATIONS! You are a winner you have been specially selected to receive $1000 cash or a $2000 award. To claim call 09061701461 claim code KL341.",  # Long spam
    ]

    print("Example messages and their token lengths:")
    tokenized_examples = []
    for i, msg in enumerate(example_messages):
        tokens = tokenizer.encode(msg)
        tokenized_examples.append(tokens)
        print(f"{i+1}. ({len(tokens):2d} tokens) {msg}")

    # Strategy 1: Pad to longest in batch
    print("\n--- Strategy 1: Pad to Longest in Batch ---")
    max_len_batch = max(len(tokens) for tokens in tokenized_examples)
    print(f"Longest message: {max_len_batch} tokens")

    pad_token_id = 50256  # <|endoftext|> token

    for i, tokens in enumerate(tokenized_examples):
        padded = tokens + [pad_token_id] * (max_len_batch - len(tokens))
        padding_added = len(padded) - len(tokens)
        print(
            f"Message {i+1}: {len(tokens)} → {len(padded)} tokens (+{padding_added} padding)"
        )

    total_tokens = len(tokenized_examples) * max_len_batch
    actual_tokens = sum(len(tokens) for tokens in tokenized_examples)
    waste_percentage = ((total_tokens - actual_tokens) / total_tokens) * 100

    print(f"Total tokens: {total_tokens} (actual: {actual_tokens})")
    print(f"Padding waste: {waste_percentage:.1f}%")

    # Strategy 2: Fixed max length
    print("\n--- Strategy 2: Fixed Max Length ---")
    fixed_max_length = 50  # Reasonable for this dataset

    print(f"Fixed max_length: {fixed_max_length} tokens")

    for i, tokens in enumerate(tokenized_examples):
        # Truncate if too long, pad if too short
        if len(tokens) > fixed_max_length:
            action = f"truncated ({len(tokens)} → {fixed_max_length})"
        else:
            action = f"padded ({len(tokens)} → {fixed_max_length})"

        print(f"Message {i+1}: {action}")

    fixed_total_tokens = len(tokenized_examples) * fixed_max_length
    fixed_waste_percentage = (
        (fixed_total_tokens - actual_tokens) / fixed_total_tokens
    ) * 100

    print(f"Total tokens: {fixed_total_tokens} (actual: {actual_tokens})")
    print(f"Padding waste: {fixed_waste_percentage:.1f}%")

    print("\n--- Strategy Comparison ---")
    print(f"Batch-adaptive padding: {waste_percentage:.1f}% waste, variable memory")
    print(
        f"Fixed-length padding: {fixed_waste_percentage:.1f}% waste, predictable memory"
    )
    print("\nRecommendation: Use fixed-length for simplicity and memory predictability")


class SpamDataset(Dataset):
    """PyTorch Dataset for spam classification with text padding."""

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Initialize the spam dataset.

        Args:
            csv_file: Path to CSV file with 'Text' and 'Label' columns
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length (None for auto-detect)
            pad_token_id: Token ID to use for padding
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

        print(f"Loading {csv_file}...")
        print(f"Found {len(self.data):,} examples")

        # Tokenize all texts
        print("Tokenizing texts...")
        self.encoded_texts = []
        for text in self.data["Text"]:
            tokens = tokenizer.encode(text)
            self.encoded_texts.append(tokens)

        # Determine max_length
        if max_length is None:
            self.max_length = self._find_optimal_length()
            print(f"Auto-detected max_length: {self.max_length}")
        else:
            self.max_length = max_length
            print(f"Using provided max_length: {self.max_length}")

        # Apply truncation and padding
        print("Applying truncation and padding...")
        self._process_sequences()

        print(f"✓ Dataset ready: {len(self):,} examples, {self.max_length} tokens each")

    def _find_optimal_length(self):
        """Find optimal sequence length covering 95% of data."""
        lengths = [len(tokens) for tokens in self.encoded_texts]
        return int(np.percentile(lengths, 95))

    def _process_sequences(self):
        """Apply truncation and padding to all sequences."""
        processed_texts = []
        truncated_count = 0

        for tokens in self.encoded_texts:
            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
                truncated_count += 1

            # Pad if too short
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            processed_texts.append(tokens)

        self.encoded_texts = processed_texts

        if truncated_count > 0:
            truncate_pct = (truncated_count / len(self.data)) * 100
            print(f"  Truncated {truncated_count:,} texts ({truncate_pct:.1f}%)")

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.

        Returns:
            tuple: (input_tokens, label) where:
                - input_tokens: tensor of token IDs [max_length]
                - label: tensor with class label [1]
        """
        encoded_text = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]

        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def get_text_info(self, idx):
        """Get human-readable info about an example."""
        original_text = self.data.iloc[idx]["Text"]
        tokens = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]

        # Count actual vs padding tokens
        actual_tokens = sum(1 for token in tokens if token != self.pad_token_id)
        padding_tokens = len(tokens) - actual_tokens

        return {
            "text": original_text,
            "label": label,
            "total_tokens": len(tokens),
            "actual_tokens": actual_tokens,
            "padding_tokens": padding_tokens,
            "token_ids": tokens[:10] + ["..."] if actual_tokens > 10 else tokens,
        }


def demonstrate_dataset_class():
    """Demonstrate the SpamDataset class functionality."""
    print("\n=== Demonstrating SpamDataset Class ===")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Check if datasets exist
    try:
        print("Creating SpamDataset instances...")

        # Create datasets with consistent max_length
        max_length = 120  # Based on analysis from spam_dataset_preparation.py

        train_dataset = SpamDataset(
            csv_file="book/ch6/train.csv", tokenizer=tokenizer, max_length=max_length
        )

        val_dataset = SpamDataset(
            csv_file="book/ch6/validation.csv",
            tokenizer=tokenizer,
            max_length=max_length,
        )

        test_dataset = SpamDataset(
            csv_file="book/ch6/test.csv", tokenizer=tokenizer, max_length=max_length
        )

        print("\n--- Dataset Sizes ---")
        print(f"Training: {len(train_dataset):,} examples")
        print(f"Validation: {len(val_dataset):,} examples")
        print(f"Test: {len(test_dataset):,} examples")

        return train_dataset, val_dataset, test_dataset

    except FileNotFoundError as e:
        print(f"⚠️  Dataset file not found: {e}")
        print("Please run spam_dataset_preparation.py first to create the datasets.")
        return None, None, None


def examine_dataset_examples(dataset, num_examples=3):
    """Examine specific examples from the dataset."""
    if dataset is None:
        return

    print(f"\n=== Examining {num_examples} Dataset Examples ===")

    for i in range(min(num_examples, len(dataset))):
        print(f"\n--- Example {i+1} ---")

        # Get the processed tensors
        input_tokens, label = dataset[i]

        # Get human-readable information
        info = dataset.get_text_info(i)

        print(
            f"Original text: \"{info['text'][:100]}{'...' if len(info['text']) > 100 else ''}\""
        )
        print(
            f"Label: {info['label']} ({'spam' if info['label'] == 1 else 'not spam'})"
        )
        print("Token breakdown:")
        print(f"  Total tokens: {info['total_tokens']}")
        print(f"  Actual tokens: {info['actual_tokens']}")
        print(f"  Padding tokens: {info['padding_tokens']}")
        print(
            f"  Padding ratio: {(info['padding_tokens'] / info['total_tokens']) * 100:.1f}%"
        )

        # Show tensor shapes
        print("Tensor shapes:")
        print(f"  Input: {input_tokens.shape} (dtype: {input_tokens.dtype})")
        print(f"  Label: {label.shape} (dtype: {label.dtype})")

        # Show first few token IDs
        first_tokens = input_tokens[: min(10, len(input_tokens))].tolist()
        print(f"First tokens: {first_tokens}")


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=8):
    """Create PyTorch DataLoaders for the datasets."""
    print(f"\n=== Creating DataLoaders (batch_size={batch_size}) ===")

    if None in [train_dataset, val_dataset, test_dataset]:
        print("⚠️  Cannot create DataLoaders - datasets not available")
        return None, None, None

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0,  # Simple single-threaded loading
        drop_last=True,  # Drop incomplete batches for consistent training
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=0,
        drop_last=False,  # Keep all validation examples
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=0,
        drop_last=False,  # Keep all test examples
    )

    print("DataLoader statistics:")
    print(
        f"  Training: {len(train_loader):,} batches ({len(train_loader) * batch_size:,} examples)"
    )
    print(f"  Validation: {len(val_loader):,} batches ({len(val_dataset):,} examples)")
    print(f"  Test: {len(test_loader):,} batches ({len(test_dataset):,} examples)")

    return train_loader, val_loader, test_loader


def examine_batch_structure(data_loader, loader_name="training"):
    """Examine the structure of batches from a DataLoader."""
    if data_loader is None:
        return

    print(f"\n=== Examining {loader_name.title()} Batch Structure ===")

    # Get first batch
    batch_iter = iter(data_loader)
    input_batch, target_batch = next(batch_iter)

    print("Batch dimensions:")
    print(f"  Input batch: {input_batch.shape}")
    print(f"    - Batch size: {input_batch.shape[0]}")
    print(f"    - Sequence length: {input_batch.shape[1]}")
    print(f"    - Data type: {input_batch.dtype}")

    print(f"  Target batch: {target_batch.shape}")
    print(f"    - Batch size: {target_batch.shape[0]}")
    print(f"    - Data type: {target_batch.dtype}")

    # Show target distribution in this batch
    unique_labels, counts = torch.unique(target_batch, return_counts=True)
    print("  Label distribution in batch:")
    for label, count in zip(unique_labels, counts):
        class_name = "not spam" if label == 0 else "spam"
        print(f"    {label} ({class_name}): {count} examples")

    # Show memory usage
    input_memory = input_batch.numel() * input_batch.element_size() / (1024 * 1024)
    target_memory = target_batch.numel() * target_batch.element_size() / (1024 * 1024)

    print("Memory usage per batch:")
    print(f"  Input: {input_memory:.2f} MB")
    print(f"  Target: {target_memory:.4f} MB")
    print(f"  Total: {input_memory + target_memory:.2f} MB")


def demonstrate_padding_visualization(data_loader):
    """Visualize padding in actual batches."""
    if data_loader is None:
        return

    print("\n=== Padding Visualization in Real Batch ===")

    # Get a batch
    batch_iter = iter(data_loader)
    input_batch, target_batch = next(batch_iter)

    pad_token_id = 50256  # <|endoftext|>

    print(f"Analyzing padding in batch of {input_batch.shape[0]} examples:")
    print(f"Sequence length: {input_batch.shape[1]} tokens")

    padding_stats = []

    for i in range(input_batch.shape[0]):
        sequence = input_batch[i]
        label = target_batch[i].item()

        # Count padding tokens
        padding_mask = sequence == pad_token_id
        actual_tokens = (~padding_mask).sum().item()
        padding_tokens = padding_mask.sum().item()

        padding_stats.append(
            {
                "example": i + 1,
                "label": label,
                "actual_tokens": actual_tokens,
                "padding_tokens": padding_tokens,
                "padding_ratio": padding_tokens / len(sequence),
            }
        )

    # Display stats
    print(f"\n{'Example':>7} {'Label':>5} {'Actual':>7} {'Padding':>8} {'Ratio':>8}")
    print("-" * 40)

    total_actual = 0
    total_padding = 0

    for stats in padding_stats:
        class_name = "spam" if stats["label"] == 1 else "ham"
        print(
            f"{stats['example']:>7} {class_name:>5} {stats['actual_tokens']:>7} "
            f"{stats['padding_tokens']:>8} {stats['padding_ratio']*100:>6.1f}%"
        )

        total_actual += stats["actual_tokens"]
        total_padding += stats["padding_tokens"]

    # Summary statistics
    total_tokens = total_actual + total_padding
    overall_padding_ratio = total_padding / total_tokens

    print("-" * 40)
    print(
        f"{'Total':>7} {'':<5} {total_actual:>7} {total_padding:>8} {overall_padding_ratio*100:>6.1f}%"
    )

    print("\nBatch efficiency:")
    print(f"  Actual content: {(1 - overall_padding_ratio)*100:.1f}%")
    print(f"  Padding waste: {overall_padding_ratio*100:.1f}%")


def compare_with_pretraining_data():
    """Compare classification data format with pretraining data format."""
    print("\n=== Comparison with Pretraining Data Format ===")

    print("Data format differences:")

    print("\n**Pretraining Data:**")
    print("  Input shape: [batch_size, context_length]")
    print("  Target shape: [batch_size, context_length]")
    print("  Example:")
    print("    Input:  [   15,   25,   13,   42, ...]  # Token sequence")
    print("    Target: [   25,   13,   42,   91, ...]  # Shifted by 1")
    print("  Loss: CrossEntropy on every position")

    print("\n**Classification Data:**")
    print("  Input shape: [batch_size, max_length]")
    print("  Target shape: [batch_size]")
    print("  Example:")
    print("    Input:  [   15,   25,   13, 50256, ...]  # Padded sequence")
    print("    Target: 1                               # Single class label")
    print("  Loss: CrossEntropy on final prediction only")

    print("\nImplications for model architecture:")
    print("• **Pretraining**: Uses all token positions for prediction")
    print(
        "• **Classification**: Uses only last token position (or pooled representation)"
    )
    print(
        "• **Attention masking**: Classification may need different attention patterns"
    )
    print("• **Output layer**: Different dimensions (vocab_size vs num_classes)")


def main():
    """Main function demonstrating classification dataset creation and usage."""
    print("=" * 70)
    print("Chapter 6: Classification Dataset Creation and Data Loading")
    print("=" * 70)

    # Step 1: Explain differences from pretraining
    explain_classification_vs_pretraining()

    # Step 2: Analyze text lengths (if data available)
    length_analysis = analyze_text_lengths()

    if length_analysis is not None:
        token_lengths, datasets, tokenizer = length_analysis

        # Step 3: Visualize length distribution
        visualize_length_distribution(token_lengths)

        # Step 4: Demonstrate padding strategies
        demonstrate_padding_strategies()

        # Step 5: Create dataset instances
        train_dataset, val_dataset, test_dataset = demonstrate_dataset_class()

        if train_dataset is not None:
            # Step 6: Examine dataset examples
            examine_dataset_examples(train_dataset, num_examples=3)

            # Step 7: Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                train_dataset, val_dataset, test_dataset, batch_size=8
            )

            # Step 8: Examine batch structure
            examine_batch_structure(train_loader, "training")
            examine_batch_structure(val_loader, "validation")

            # Step 9: Visualize padding in real batches
            demonstrate_padding_visualization(train_loader)

            # Step 10: Compare with pretraining format
            compare_with_pretraining_data()

            print("\n" + "=" * 70)
            print("Classification dataset preparation completed successfully!")
            print("✓ Text length analysis performed")
            print("✓ Optimal padding strategy determined")
            print("✓ PyTorch Dataset classes created")
            print("✓ DataLoaders configured for training")
            print("✓ Batch structure verified")
            print("✓ Memory efficiency analyzed")
            print("\nDatasets ready for model modification and fine-tuning!")
            print(
                "Next step: Run model_modification.py to prepare GPT for classification"
            )
            print("=" * 70)

        else:
            print("\n⚠️  Could not create datasets. Please check that CSV files exist.")

    else:
        print(
            "\n⚠️  Could not analyze dataset. Please run spam_dataset_preparation.py first."
        )


if __name__ == "__main__":
    main()
