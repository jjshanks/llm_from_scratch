"""
Chapter 2 - Reading and Preparing Text Data

This script demonstrates the first step in preparing text data for LLM training:
loading a text file and examining its basic properties. We use "The Verdict" by
Edith Wharton as our training text, which is in the public domain.

The script shows how to:
1. Load text from a file
2. Check the total character count
3. Preview the beginning of the text

Usage: uv run python book/ch2/read_story.py
"""

# Load the text file - "The Verdict" by Edith Wharton
# This is a public domain short story suitable for LLM training
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Display basic statistics about the text
print("Total number of character:", len(raw_text))

# Preview the first 99 characters to understand the text structure
print(raw_text[:99])
"""expected output:
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no
"""
