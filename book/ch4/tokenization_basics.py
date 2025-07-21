"""
Chapter 4 - Tokenization and Batch Processing

This script demonstrates how text is converted to numbers for processing by
the GPT model, and how multiple texts are batched together for efficiency.

Key concepts demonstrated:
- Using tiktoken for GPT-2 compatible tokenization
- Converting text to token IDs
- Creating batches for parallel processing
- Understanding batch dimensions and shapes

Before feeding text to a neural network, we must convert it to numbers.
The tokenizer maps text to a sequence of integers (token IDs), where
each ID corresponds to a token in the vocabulary.

Usage: uv run python book/ch4/tokenization_basics.py
"""

import torch
import tiktoken


def demonstrate_tokenization():
    """Show how text is converted to token IDs."""
    print("=== Basic Tokenization ===\n")

    # Initialize the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Example text
    text = "Hello, world! How are you?"

    # Tokenize
    token_ids = tokenizer.encode(text)

    print(f"Original text: '{text}'")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")

    # Show individual tokens
    print("\nToken breakdown:")
    for i, token_id in enumerate(token_ids):
        token_bytes = tokenizer.decode_single_token_bytes(token_id)
        # Decode bytes to string, handling potential errors
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = f"[bytes: {token_bytes.hex()}]"
        print(f"  Position {i}: ID={token_id:5d} → '{token_str}'")

    # Decode back to text
    decoded_text = tokenizer.decode(token_ids)
    print(f"\nDecoded text: '{decoded_text}'")
    print("Note: Tokenization is reversible!")


def explain_special_tokens():
    """Explain special tokens in the vocabulary."""
    print("\n\n=== Special Tokens ===\n")

    tokenizer = tiktoken.get_encoding("gpt2")

    # GPT-2's special tokens
    special_tokens = {
        50256: "<|endoftext|>",  # End of text marker
    }

    print("GPT-2 uses special tokens for specific purposes:")
    for token_id, description in special_tokens.items():
        print(f"  Token ID {token_id}: {description}")

    # Example with special token
    text_with_special = "First document.<|endoftext|>Second document."
    tokens = tokenizer.encode(text_with_special, allowed_special={"<|endoftext|>"})

    print(f"\nExample text: '{text_with_special}'")
    print(f"Token IDs: {tokens}")
    print("The <|endoftext|> token helps the model understand document boundaries")


def create_batch_example():
    """Demonstrate batch processing of multiple texts."""
    print("\n\n=== What is a Batch? ===")
    print("A batch groups multiple inputs for efficient parallel processing")
    print("Like baking 12 cookies at once instead of 1 at a time")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Our example texts
    texts = ["Every effort moves you", "Every day holds a"]

    print(f"\nOur batch will contain {len(texts)} text samples:")
    for i, text in enumerate(texts):
        print(f"Text {i+1}: '{text}'")

    # Tokenize each text
    print("\n=== Tokenization Process ===")
    print("Before feeding text to the model, we convert words to numbers (token IDs)")

    batch_list = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        print(f"'{text}' → {token_ids}")
        batch_list.append(torch.tensor(token_ids))

    # Stack into a batch tensor
    batch = torch.stack(batch_list, dim=0)

    print("\nNow we stack these into a 2D tensor (like a spreadsheet):")
    print("- Each row = one text sample")
    print("- Each column = token position (1st word, 2nd word, etc.)")
    print("\nTokenized batch:")
    print(batch)
    print(f"\nBatch shape: {batch.shape}")
    print("This means: 2 texts, with 4 tokens each")
    print("The model will process both texts simultaneously!")

    return batch, tokenizer


def explain_batch_dimensions():
    """Explain the meaning of batch dimensions in detail."""
    print("\n\n=== Understanding Batch Dimensions ===")

    batch, tokenizer = create_batch_example()

    print("\nOur batch tensor has shape:", batch.shape)
    print("Let's break this down:\n")

    print("Dimension 0 (Batch Size):")
    print(f"  - Size: {batch.shape[0]}")
    print("  - Meaning: Number of text samples")
    print("  - Why batch? Processing multiple texts together is more efficient")
    print("  - Like a classroom processing multiple students at once")

    print("\nDimension 1 (Sequence Length):")
    print(f"  - Size: {batch.shape[1]}")
    print("  - Meaning: Number of tokens per text")
    print("  - All texts in a batch must have the same length")
    print("  - Shorter texts are padded, longer texts are truncated")

    print("\n=== Accessing Individual Elements ===")
    print(f"batch[0] = {batch[0]} (first text)")
    print(f"batch[1] = {batch[1]} (second text)")
    print(f"batch[0, 0] = {batch[0, 0]} (first token of first text)")
    print(f"batch[1, 3] = {batch[1, 3]} (last token of second text)")

    # Decode to show what each position represents
    print("\n=== Decoding Individual Positions ===")
    for i in range(batch.shape[0]):
        print(f"\nText {i+1}:")
        for j in range(batch.shape[1]):
            token_id = batch[i, j].item()
            token_str = tokenizer.decode([token_id])
            print(f"  Position [{i},{j}]: {token_id:5d} → '{token_str}'")


def demonstrate_variable_length_texts():
    """Show how to handle texts of different lengths."""
    print("\n\n=== Handling Variable Length Texts ===")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Texts of different lengths
    texts = [
        "Short text",
        "This is a medium length text example",
        "This is a much longer text that demonstrates how we handle variable length inputs in batches",
    ]

    print("Original texts (different lengths):")
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        print(f"  Text {i+1}: {len(tokens)} tokens - '{text}'")

    # Method 1: Truncate to shortest
    print("\nMethod 1: Truncate all to shortest length")
    min_length = min(len(tokenizer.encode(text)) for text in texts)
    print(f"Shortest text has {min_length} tokens")

    batch_truncated = []
    for text in texts:
        tokens = tokenizer.encode(text)[:min_length]  # Truncate
        batch_truncated.append(torch.tensor(tokens))
    batch_truncated = torch.stack(batch_truncated)
    print("Truncated batch shape:", batch_truncated.shape)

    # Method 2: Pad to longest (more common)
    print("\nMethod 2: Pad all to longest length (more common)")
    max_length = max(len(tokenizer.encode(text)) for text in texts)
    print(f"Longest text has {max_length} tokens")

    pad_token_id = 50256  # Using endoftext as pad token
    batch_padded = []
    for text in texts:
        tokens = tokenizer.encode(text)
        # Pad with pad_token_id
        padded = tokens + [pad_token_id] * (max_length - len(tokens))
        batch_padded.append(torch.tensor(padded))
    batch_padded = torch.stack(batch_padded)
    print("Padded batch shape:", batch_padded.shape)

    # Show the padded batch
    print("\nPadded batch (50256 is the padding token):")
    for i, row in enumerate(batch_padded):
        print(f"Text {i+1}: {row.tolist()}")


def explain_why_batching():
    """Explain the benefits of batch processing."""
    print("\n\n=== Why Use Batches? ===\n")

    print("1. **Computational Efficiency**")
    print("   - GPUs are designed for parallel processing")
    print("   - Processing 32 texts at once is much faster than 32 sequential texts")
    print("   - Matrix operations on batches utilize GPU cores efficiently")

    print("\n2. **Stable Training**")
    print("   - Gradients averaged over batch are less noisy")
    print("   - Larger batches provide better gradient estimates")
    print("   - Like surveying 100 people vs 1 person for better statistics")

    print("\n3. **Memory Utilization**")
    print("   - Better use of GPU memory bandwidth")
    print("   - Amortizes fixed overhead across multiple samples")

    print("\n4. **Typical Batch Sizes**")
    print("   - Training: 8-512 depending on model size and GPU memory")
    print(
        "   - Inference: Can be 1 (for interactive use) or larger (for bulk processing)"
    )
    print("   - Larger models require smaller batches due to memory constraints")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Tokenization and Batch Processing")
    print("=" * 60)

    # Basic tokenization
    demonstrate_tokenization()

    # Special tokens
    explain_special_tokens()

    # Batch creation
    create_batch_example()

    # Understanding dimensions
    explain_batch_dimensions()

    # Variable length handling
    demonstrate_variable_length_texts()

    # Why batching matters
    explain_why_batching()

    print("\n" + "=" * 60)
    print("Summary: Tokenization converts text to numbers,")
    print("and batching processes multiple texts efficiently!")
    print("=" * 60)
