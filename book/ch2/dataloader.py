"""
Chapter 2 - Data Sampling with Sliding Window Approach

This script demonstrates how to create a DataLoader for LLM training using a sliding
window approach to generate input-target pairs. The DataLoader efficiently batches
sequences of tokens where each target is the next token in the sequence.

The sliding window approach moves across the text to create overlapping training samples,
which is essential for training LLMs to predict the next word in a sequence.
"""
import tiktoken
from torch.utils.data import Dataset, DataLoader
from llm_from_scratch.dataset import GPTDatasetV1   

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Create a DataLoader for GPT-style training data.
    
    Args:
        txt: Raw text to tokenize and create samples from
        batch_size: Number of samples per batch
        max_length: Maximum sequence length (context window)
        stride: Step size for sliding window (controls overlap)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader that yields (input_ids, target_ids) pairs
    """
    # Use GPT-2's BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset using sliding window approach
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # Create DataLoader for efficient batching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Load the training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Demonstrate DataLoader with batch_size=1 and stride=1
# This creates overlapping sequences for maximum data utilization
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)

# Show how consecutive batches are shifted by the stride
first_batch = next(data_iter)
print("First batch (input_ids, target_ids):")
print(first_batch)

second_batch = next(data_iter)
print("\nSecond batch (shifted by stride=1):")
print(second_batch)

# Demonstrate DataLoader with larger batch_size and stride=max_length
# This creates non-overlapping sequences for faster training
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nBatch with 8 samples:")
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

"""expected output:
First batch (input_ids, target_ids):
[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

Second batch (shifted by stride=1):
[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

Batch with 8 samples:
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
"""