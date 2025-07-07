"""
Chapter 2 - Text Tokenization with Regular Expressions

This script demonstrates basic text tokenization using regular expressions.
Tokenization is the process of breaking down text into individual tokens
(words and punctuation) that can be processed by machine learning models.

The tokenization approach used here:
1. Splits text on whitespace and common punctuation
2. Removes empty strings and whitespace-only tokens
3. Preserves punctuation as separate tokens

This creates a foundation for understanding how text preprocessing works
before moving to more sophisticated methods like BPE.

Usage: uv run python book/ch2/tokenize_story.py
"""
import re

# Load the text file
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize using regular expressions
# This pattern splits on:
# - Whitespace (\s)
# - Common punctuation: , . : ; ? _ ! " ( ) '
# - Double dashes (--)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Remove empty strings and whitespace-only tokens
# This cleans up the tokenized output
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Display tokenization results
print(f"Total tokens: {len(preprocessed)}")
print(f"First 30 tokens: {preprocessed[:30]}")
"""expected output:
Total tokens: 4690
First 30 tokens: ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
"""