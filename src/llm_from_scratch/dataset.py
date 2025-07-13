import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for creating input-target pairs for GPT model training.

    This dataset implements a sliding window approach to create overlapping
    sequences from tokenized text. Each sample consists of an input sequence
    and a target sequence, where the target is the input shifted by one token.
    This setup is used for next-token prediction during language model training.

    The sliding window approach with configurable stride allows for efficient
    use of training data by creating multiple overlapping sequences from the
    same text.

    Args:
        txt: Raw text string to create dataset from
        tokenizer: Tokenizer instance with encode() method to convert text to token IDs
        max_length: Length of each input/target sequence (context window size)
        stride: Number of tokens to slide the window by. Use stride=max_length
               for non-overlapping sequences, or stride<max_length for overlap
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialize the dataset by tokenizing text and creating input-target pairs.

        Args:
            txt: Input text to tokenize and split into sequences
            tokenizer: Tokenizer with encode() method for text-to-token conversion
            max_length: Fixed length for each sequence (must be less than text length)
            stride: Step size for sliding window (controls overlap between sequences)
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text once
        token_ids = tokenizer.encode(txt)

        # Create input-target pairs using sliding window approach
        # Stop when there aren't enough tokens left for a complete window
        for i in range(0, len(token_ids) - max_length, stride):
            # Input chunk: tokens from position i to i+max_length
            input_chunk = token_ids[i : i + max_length]
            # Target chunk: same tokens shifted by 1 position for next-token prediction
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            # Convert to tensors and store
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Return the number of input-target pairs in the dataset.

        Returns:
            Number of sequences created from the input text
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Get a single input-target pair by index.

        Args:
            idx: Index of the sequence pair to retrieve

        Returns:
            Tuple of (input_ids, target_ids) tensors, both of shape (max_length,)
        """
        return self.input_ids[idx], self.target_ids[idx]
