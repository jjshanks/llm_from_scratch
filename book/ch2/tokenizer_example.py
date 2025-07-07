"""
Chapter 2 - Custom Tokenizer Implementation

This script demonstrates how to use a custom tokenizer implementation that handles
special tokens like <|unk|> for unknown words and <|endoftext|> for document boundaries.

The custom tokenizer (SimpleTokenizerV1) implements:
1. Text tokenization using regular expressions
2. Token-to-ID mapping using a vocabulary
3. Handling of unknown words with <|unk|> tokens
4. Special context tokens like <|endoftext|>
5. Encoding and decoding methods

This shows the evolution from basic tokenization to more sophisticated approaches
that can handle real-world text processing challenges.

Usage: uv run python book/ch2/tokenizer_example.py
"""
from llm_from_scratch.tokenizer import SimpleTokenizerV1
import re

# Load and preprocess the training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize the text using regular expressions
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Create vocabulary from unique tokens
all_words = sorted(set(preprocessed))
vocab = {token:integer for integer,token in enumerate(all_words)}

# Initialize the custom tokenizer
tokenizer = SimpleTokenizerV1(raw_text)

# Test the tokenizer with text from the training set
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""

# Encode text to token IDs
ids = tokenizer.encode(text)
print("Token IDs:", ids)

# Decode token IDs back to text
decoded_text = tokenizer.decode(ids)
print("Decoded text:", decoded_text)

# Test handling of unknown words
print(f"\nUnknown token ID: {tokenizer.UNKNOWN_TOKEN_INDEX}")
unknown_text = "Hello, do you like tea?"
unknown_ids = tokenizer.encode(unknown_text)
print(f"Unknown words encoded: {unknown_ids}")

# Test special context tokens
print(f"\nEnd of text token ID: {tokenizer.END_OF_TEXT_INDEX}")
mixed_text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
mixed_ids = tokenizer.encode(mixed_text)
print(f"Text with <|endoftext|> encoded: {mixed_ids}")

# Show how unknown words are handled in decoding
decoded_mixed = tokenizer.decode(mixed_ids)
print(f"Decoded with <|unk|> tokens: {decoded_mixed}")
"""expected output:
Token IDs: [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
Decoded text: " It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.

Unknown token ID: 1131
Unknown words encoded: [1131, 5, 355, 1126, 628, 975, 10]

End of text token ID: 1130
Text with <|endoftext|> encoded: [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
Decoded with <|unk|> tokens: <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
"""