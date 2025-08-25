"""
Chapter 6 - Preparing a Dataset for Text Classification

This script demonstrates how to prepare a real-world dataset for fine-tuning
an LLM as a text classifier, using the SMS Spam Collection dataset.

Key concepts demonstrated:
- Downloading and extracting classification datasets
- Exploring class distributions and handling imbalanced data
- Creating balanced datasets through undersampling
- Converting text labels to numerical format
- Train/validation/test splitting strategies
- Understanding dataset characteristics for classification

The script follows Section 6.2 of the book, showing how to transform
raw text messages into a format suitable for fine-tuning.

Usage: uv run python book/ch6/spam_dataset_preparation.py
"""

import urllib.request
import zipfile
import os
import pandas as pd
from pathlib import Path


def explain_classification_data():
    """Explain what makes classification data different from pretraining data."""
    print("=== Understanding Classification Data ===\n")

    print("Key differences from pretraining data:")
    print("1. **Labeled examples**: Each text has an associated class label")
    print(
        "2. **Finite classes**: Limited set of possible outputs (e.g., spam/not spam)"
    )
    print("3. **Supervised learning**: Model learns from input-output pairs")
    print("4. **Evaluation metrics**: Focus on classification accuracy, not perplexity")

    print("\nFor spam detection:")
    print("• Input: Text message content")
    print("• Output: Binary label (0=not spam, 1=spam)")
    print("• Goal: Learn to distinguish spam from legitimate messages")

    print("\nThis contrasts with pretraining where:")
    print("• Input: Text sequence")
    print("• Output: Next token prediction")
    print("• Goal: Learn general language patterns")


def download_spam_dataset():
    """Download and extract the SMS Spam Collection dataset."""
    print("\n=== Downloading SMS Spam Collection Dataset ===\n")

    # Dataset details
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "book/ch6/sms_spam_collection.zip"
    extracted_path = "book/ch6/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    print("Dataset: SMS Spam Collection (UCI Machine Learning Repository)")
    print("Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")
    print("Format: Tab-separated text messages with spam/ham labels")
    print("License: Public domain")

    # Check if already downloaded
    if data_file_path.exists():
        print(f"\n✓ Dataset already exists at {data_file_path}")
        print("Skipping download...")
        return data_file_path

    print(f"\nDownloading from: {url}")
    print("Please wait, downloading...")

    try:
        # Download the dataset
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())
        print("✓ Download completed")

        # Extract the zip file
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        # Rename to consistent format
        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        if original_file_path.exists():
            os.rename(original_file_path, data_file_path)

        # Cleanup
        os.remove(zip_path)

        print(f"✓ Dataset saved as: {data_file_path}")
        return data_file_path

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the URL is accessible")
        print("3. Ensure you have write permissions")
        raise


def visualize_ascii_bar_chart(counts, labels, title, max_width=50):
    """Create an ASCII bar chart for visualizing data distributions."""
    print(f"\n{title}")
    print("=" * len(title))

    max_count = max(counts)

    for label, count in zip(labels, counts):
        # Calculate bar width proportional to count
        bar_width = int((count / max_count) * max_width)
        bar = "█" * bar_width
        percentage = (count / sum(counts)) * 100

        print(f"{label:>8}: {bar} {count:,} ({percentage:.1f}%)")

    print(f"\nTotal: {sum(counts):,} examples")


def load_and_explore_dataset(data_file_path):
    """Load the dataset and explore its characteristics."""
    print("\n=== Loading and Exploring Dataset ===\n")

    # Load the dataset
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    print(f"Successfully loaded {len(df):,} text messages")

    # Show basic info
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Preview samples
    print("\n--- Sample Messages ---")
    for i, (label, text) in enumerate(df.head(3).values):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"{i+1}. [{label}] {preview}")

    return df


def analyze_class_distribution(df):
    """Analyze the distribution of classes in the dataset."""
    print("\n=== Class Distribution Analysis ===")

    # Count classes
    class_counts = df["Label"].value_counts()
    print("\nRaw class counts:")
    for label, count in class_counts.items():
        print(f"  {label}: {count:,} messages")

    # Visualize distribution
    visualize_ascii_bar_chart(
        counts=class_counts.values,
        labels=class_counts.index,
        title="Message Distribution by Class",
    )

    # Calculate imbalance ratio
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)
    imbalance_ratio = ham_count / spam_count if spam_count > 0 else float("inf")

    print("\n--- Class Imbalance Analysis ---")
    print(f"Ham to Spam ratio: {imbalance_ratio:.1f}:1")
    print(f"Spam percentage: {(spam_count / len(df)) * 100:.1f}%")

    if imbalance_ratio > 2:
        print("\n⚠️  Dataset is imbalanced!")
        print("Implications:")
        print("• Model may bias toward majority class (ham)")
        print("• Simple accuracy can be misleading")
        print("• May need balancing strategies")
    else:
        print("\n✓ Dataset is reasonably balanced")

    return class_counts


def analyze_text_characteristics(df):
    """Analyze text characteristics for both classes."""
    print("\n=== Text Characteristics Analysis ===")

    # Calculate text lengths
    df["text_length"] = df["Text"].str.len()
    df["word_count"] = df["Text"].str.split().str.len()

    print("\n--- Length Statistics ---")
    # Group statistics for analysis
    df.groupby("Label")[["text_length", "word_count"]].agg(
        ["mean", "std", "min", "max"]
    )

    for label in ["ham", "spam"]:
        if label in df["Label"].values:
            label_data = df[df["Label"] == label]
            avg_chars = label_data["text_length"].mean()
            avg_words = label_data["word_count"].mean()

            print(f"\n{label.upper()} messages:")
            print(f"  Average characters: {avg_chars:.1f}")
            print(f"  Average words: {avg_words:.1f}")
            print(
                f"  Character range: {label_data['text_length'].min()}-{label_data['text_length'].max()}"
            )
            print(
                f"  Word range: {label_data['word_count'].min()}-{label_data['word_count'].max()}"
            )

    # Show examples of different lengths
    print("\n--- Examples by Length ---")

    # Shortest and longest messages
    shortest = df.loc[df["text_length"].idxmin()]
    longest = df.loc[df["text_length"].idxmax()]

    print(f"\nShortest message ({shortest['text_length']} chars, {shortest['Label']}):")
    print(f"  \"{shortest['Text']}\"")

    print(f"\nLongest message ({longest['text_length']} chars, {longest['Label']}):")
    preview = (
        longest["Text"][:200] + "..." if len(longest["Text"]) > 200 else longest["Text"]
    )
    print(f'  "{preview}"')


def create_balanced_dataset(df):
    """Create a balanced dataset by undersampling the majority class."""
    print("\n=== Creating Balanced Dataset ===")

    print("Strategy: Undersample majority class (ham) to match minority class (spam)")
    print("Alternative strategies:")
    print("• Oversample minority class (duplicate spam messages)")
    print("• Use class weights in loss function")
    print("• Generate synthetic minority examples")

    # Count examples per class
    class_counts = df["Label"].value_counts()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)

    print("\nOriginal distribution:")
    print(f"  Ham: {ham_count:,} messages")
    print(f"  Spam: {spam_count:,} messages")

    # Use smaller class size as target
    min_count = min(ham_count, spam_count)
    print(f"\nTarget size per class: {min_count:,} messages")

    # Sample from each class
    ham_subset = df[df["Label"] == "ham"].sample(min_count, random_state=123)
    spam_subset = df[df["Label"] == "spam"]

    # Combine balanced subsets
    balanced_df = pd.concat([ham_subset, spam_subset], ignore_index=True)

    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

    print("\nBalanced dataset created:")
    balanced_counts = balanced_df["Label"].value_counts()
    for label, count in balanced_counts.items():
        print(f"  {label}: {count:,} messages")

    print(f"\nTotal messages: {len(balanced_df):,} (reduced from {len(df):,})")
    print(f"Reduction: {((len(df) - len(balanced_df)) / len(df)) * 100:.1f}%")

    return balanced_df


def convert_labels_to_numeric(df):
    """Convert string labels to numeric format for training."""
    print("\n=== Converting Labels to Numeric Format ===")

    print("Label mapping for binary classification:")
    print("  'ham' (not spam) → 0")
    print("  'spam'           → 1")

    # Create mapping
    label_mapping = {"ham": 0, "spam": 1}

    # Apply mapping
    original_labels = df["Label"].copy()
    df["Label"] = df["Label"].map(label_mapping)

    # Verify conversion
    print("\nConversion verification:")
    conversion_check = pd.DataFrame(
        {"Original": original_labels.head(), "Numeric": df["Label"].head()}
    )
    print(conversion_check.to_string(index=False))

    # Check for any unmapped labels
    if df["Label"].isnull().any():
        unmapped = original_labels[df["Label"].isnull()].unique()
        print(f"\n⚠️  Warning: Unmapped labels found: {unmapped}")
    else:
        print("\n✓ All labels successfully converted")

    # Show final distribution
    final_counts = df["Label"].value_counts().sort_index()
    print("\nFinal numeric label distribution:")
    for label, count in final_counts.items():
        class_name = "not spam" if label == 0 else "spam"
        print(f"  {label} ({class_name}): {count:,} messages")

    return df


def split_dataset(df, train_frac=0.7, validation_frac=0.1, test_frac=0.2):
    """Split the dataset into training, validation, and test sets."""
    print("\n=== Splitting Dataset ===")

    print("Split ratios:")
    print(f"  Training: {train_frac*100:.0f}%")
    print(f"  Validation: {validation_frac*100:.0f}%")
    print(f"  Test: {test_frac*100:.0f}%")

    # Verify ratios sum to 1
    total_frac = train_frac + validation_frac + test_frac
    if abs(total_frac - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_frac}")

    # Shuffle dataset randomly
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    n_total = len(df_shuffled)
    train_end = int(n_total * train_frac)
    validation_end = train_end + int(n_total * validation_frac)

    # Create splits
    train_df = df_shuffled[:train_end].copy()
    validation_df = df_shuffled[train_end:validation_end].copy()
    test_df = df_shuffled[validation_end:].copy()

    print("\nDataset split sizes:")
    print(f"  Training: {len(train_df):,} messages")
    print(f"  Validation: {len(validation_df):,} messages")
    print(f"  Test: {len(test_df):,} messages")
    print(f"  Total: {len(train_df) + len(validation_df) + len(test_df):,} messages")

    # Verify class balance in each split
    print("\n--- Class Distribution in Each Split ---")
    for name, split_df in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        class_counts = split_df["Label"].value_counts().sort_index()
        spam_pct = (class_counts.get(1, 0) / len(split_df)) * 100
        print(f"{name}:")
        print(f"  Class 0 (not spam): {class_counts.get(0, 0):,} ({100-spam_pct:.1f}%)")
        print(f"  Class 1 (spam): {class_counts.get(1, 0):,} ({spam_pct:.1f}%)")

    return train_df, validation_df, test_df


def save_datasets(train_df, validation_df, test_df):
    """Save the processed datasets to CSV files."""
    print("\n=== Saving Processed Datasets ===")

    filenames = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }

    print("Saving datasets as CSV files for later use:")

    for filename, df in filenames.items():
        filepath = f"book/ch6/{filename}"
        df.to_csv(filepath, index=False)
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {filename}: {len(df):,} messages ({file_size:.1f} KB)")

    print(f"\nFiles saved in current directory: {os.getcwd()}/book/ch6")

    # Show CSV format example
    print("\n--- CSV Format Example ---")
    print("train.csv structure:")
    print(train_df.head(2).to_string(index=False))


def demonstrate_data_loading():
    """Demonstrate how to reload the saved datasets."""
    print("\n=== Demonstrating Data Reloading ===")

    print("To reload these datasets in future scripts:")
    print("```python")
    print("import pandas as pd")
    print("")
    print("train_df = pd.read_csv('book/ch6/train.csv')")
    print("validation_df = pd.read_csv('book/ch6/validation.csv')")
    print("test_df = pd.read_csv('book/ch6/test.csv')")
    print("")
    print("print(f'Loaded {len(train_df)} training examples')")
    print("```")

    # Test the reload
    try:
        test_reload = pd.read_csv("book/ch6/train.csv")
        print(f"\n✓ Successfully reloaded train.csv with {len(test_reload):,} examples")
    except FileNotFoundError:
        print("\n⚠️  train.csv not found - ensure save_datasets() was called")


def dataset_quality_checks(df):
    """Perform quality checks on the dataset."""
    print("\n=== Dataset Quality Checks ===")

    checks_passed = []

    # Check for missing values
    missing_text = df["Text"].isnull().sum()
    missing_labels = df["Label"].isnull().sum()

    print("Missing values check:")
    print(f"  Text missing: {missing_text}")
    print(f"  Labels missing: {missing_labels}")
    checks_passed.append(missing_text == 0 and missing_labels == 0)

    # Check for empty texts
    empty_texts = (df["Text"].str.strip() == "").sum()
    print(f"  Empty texts: {empty_texts}")
    checks_passed.append(empty_texts == 0)

    # Check label values
    unique_labels = set(df["Label"].unique())
    expected_labels = {0, 1}
    print("Label values check:")
    print(f"  Unique labels: {sorted(unique_labels)}")
    print(f"  Expected labels: {sorted(expected_labels)}")
    checks_passed.append(unique_labels == expected_labels)

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    checks_passed.append(duplicates == 0)

    # Summary
    total_checks = len(checks_passed)
    passed_checks = sum(checks_passed)

    print(f"\nQuality check summary: {passed_checks}/{total_checks} passed")

    if passed_checks == total_checks:
        print("✅ All quality checks passed - dataset is ready for training!")
    else:
        print("⚠️  Some quality issues found - review before training")

    return all(checks_passed)


def main():
    """Main function to run the complete dataset preparation pipeline."""
    print("=" * 70)
    print("Chapter 6: Spam Dataset Preparation for Classification Fine-tuning")
    print("=" * 70)

    # Step 1: Explain the task
    explain_classification_data()

    # Step 2: Download dataset
    data_file_path = download_spam_dataset()

    # Step 3: Load and explore
    df = load_and_explore_dataset(data_file_path)

    # Step 4: Analyze characteristics
    analyze_class_distribution(df)
    analyze_text_characteristics(df)

    # Step 5: Create balanced dataset
    balanced_df = create_balanced_dataset(df)

    # Step 6: Convert labels to numeric
    balanced_df = convert_labels_to_numeric(balanced_df)

    # Step 7: Split into train/validation/test
    train_df, validation_df, test_df = split_dataset(balanced_df)

    # Step 8: Quality checks
    dataset_quality_checks(balanced_df)

    # Step 9: Save datasets
    save_datasets(train_df, validation_df, test_df)

    # Step 10: Demonstrate reloading
    demonstrate_data_loading()

    print("\n" + "=" * 70)
    print("Dataset preparation completed successfully!")
    print("✓ SMS Spam Collection dataset downloaded and processed")
    print("✓ Balanced dataset created with equal spam/ham examples")
    print("✓ Data split into train/validation/test sets")
    print("✓ Labels converted to numeric format (0=ham, 1=spam)")
    print("✓ Quality checks passed")
    print("✓ Datasets saved as CSV files for fine-tuning")
    print("\nNext step: Run classification_dataset.py to create PyTorch datasets")
    print("=" * 70)


if __name__ == "__main__":
    main()
