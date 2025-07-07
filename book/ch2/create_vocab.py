"""
Chapter 2 - Building a Vocabulary from Tokenized Text

This script demonstrates how to create a vocabulary from tokenized text data.
A vocabulary is a mapping from unique tokens to integer IDs, which is required
for converting text tokens into numerical representations that can be processed
by neural networks.

Usage: uv run python book/ch2/create_vocab.py
"""

import re

# Read the training text - "The Verdict" by Edith Wharton
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize the text using regular expressions
# This splits on whitespace and common punctuation marks
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# Remove empty strings and whitespace-only tokens
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Create a sorted list of unique tokens for the vocabulary
# Sorting ensures consistent ordering across runs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# Create the vocabulary dictionary: token -> integer ID
vocab = {token: integer for integer, token in enumerate(all_words)}

# Display the first 50 vocabulary entries
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
"""expected output:
1130
('!', 0)
('"', 1)
("'", 2)
('(', 3)
(')', 4)
(',', 5)
('--', 6)
('.', 7)
(':', 8)
(';', 9)
('?', 10)
('A', 11)
('Ah', 12)
('Among', 13)
('And', 14)
('Are', 15)
('Arrt', 16)
('As', 17)
('At', 18)
('Be', 19)
('Begin', 20)
('Burlington', 21)
('But', 22)
('By', 23)
('Carlo', 24)
('Chicago', 25)
('Claude', 26)
('Come', 27)
('Croft', 28)
('Destroyed', 29)
('Devonshire', 30)
('Don', 31)
('Dubarry', 32)
('Emperors', 33)
('Florence', 34)
('For', 35)
('Gallery', 36)
('Gideon', 37)
('Gisburn', 38)
('Gisburns', 39)
('Grafton', 40)
('Greek', 41)
('Grindle', 42)
('Grindles', 43)
('HAD', 44)
('Had', 45)
('Hang', 46)
('Has', 47)
('He', 48)
('Her', 49)
('Hermia', 50)
"""
