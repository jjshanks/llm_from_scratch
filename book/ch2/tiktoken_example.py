"""
Chapter 2 - Byte Pair Encoding (BPE) with tiktoken

This script demonstrates the use of tiktoken, which implements the BPE tokenizer
used by GPT-2 and GPT-3. BPE is superior to simple word-based tokenization because
it can handle unknown words by breaking them into subword units.

Key advantages of BPE:
- Handles unknown words without <|unk|> tokens
- Breaks unknown words into subword units or individual characters
- More efficient than character-level tokenization
- Used by modern LLMs like GPT-2, GPT-3, and ChatGPT

Usage: uv run python book/ch2/tiktoken_example.py
"""

from importlib.metadata import version
import tiktoken

# Check the version of tiktoken library
print("tiktoken version:", version("tiktoken"))

# Initialize the GPT-2 BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Example text containing unknown words and special tokens
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

# Encode text to token IDs, allowing the <|endoftext|> special token
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Token IDs:", integers)

# Decode token IDs back to text
strings = tokenizer.decode(integers)
print("Decoded text:", strings)

# Demonstrate how BPE handles unknown words
print("\nHow BPE tokenizes unknown words:")
unknown_word = "Akwirw ier"
bpe = tokenizer.encode(unknown_word)

# Show each token ID and its corresponding text
for id in bpe:
    token_text = tokenizer.decode([id])
    print(f"Token ID {id}: '{token_text}'")
"""expected output:
tiktoken version: 0.9.0
Token IDs: [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
Decoded text: Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.

How BPE tokenizes unknown words:
Token ID 33901: 'Ak'
Token ID 86: 'w'
Token ID 343: 'ir'
Token ID 86: 'w'
Token ID 220: ' '
Token ID 959: 'ier'
"""
