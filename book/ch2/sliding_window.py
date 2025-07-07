"""
Chapter 2 - Sliding Window Approach for Input-Target Pairs

This script demonstrates how to create input-target pairs for LLM training using
a sliding window approach. The key insight is that for next-word prediction, each
target token is simply the input shifted by one position.

The sliding window approach is fundamental to LLM training because it allows the model
to learn from overlapping sequences, maximizing the use of training data.
"""

import tiktoken

# Initialize the GPT-2 BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load and tokenize the training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Convert text to token IDs using BPE tokenizer
enc_text = tokenizer.encode(raw_text)
print(f"Total tokens in text: {len(enc_text)}")

# Skip the first 50 tokens for a more interesting example
enc_sample = enc_text[50:]

# Define the context size (sequence length)
context_size = 4

# Create input-target pairs: targets are inputs shifted by 1 position
x = enc_sample[:context_size]  # Input tokens
y = enc_sample[1 : context_size + 1]  # Target tokens (shifted by 1)

print(f"Input tokens (x):  {x}")
print(f"Target tokens (y): {y}")

print("\nShowing how each input predicts its corresponding target:")
# Demonstrate the next-word prediction task
for i in range(1, context_size + 1):
    context = enc_sample[:i]  # Input context of length i
    desired = enc_sample[i]  # Target token to predict
    print(f"{context} ----> {desired}")

print("\nSame relationships shown as text:")
# Show the same relationships in readable text format
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # Decode token IDs back to text for readability
    context_text = tokenizer.decode(context)
    target_text = tokenizer.decode([desired])
    print(f"'{context_text}' ----> '{target_text}'")

"""expected output:
Total tokens in text: 5145
Input tokens (x):  [290, 4920, 2241, 287]
Target tokens (y): [4920, 2241, 287, 257]

Showing how each input predicts its corresponding target:
[290] ----> 4920
[290, 4920] ----> 2241
[290, 4920, 2241] ----> 287
[290, 4920, 2241, 287] ----> 257

Same relationships shown as text:
' and' ----> ' established'
' and established' ----> ' himself'
' and established himself' ----> ' in'
' and established himself in' ----> ' a'
"""
