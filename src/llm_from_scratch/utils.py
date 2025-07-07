"""
Utility functions for LLM training and data processing.

This module provides convenient functions for creating DataLoaders, managing tokenizers,
and other common operations used throughout the LLM training pipeline.
"""

from typing import Any, Optional, Union
import tiktoken
from torch.utils.data import DataLoader

from .dataset import GPTDatasetV1
from .tokenizer import SimpleTokenizerV1


def create_tokenizer(
    text: Optional[str] = None, use_bpe: bool = True, encoding_name: str = "gpt2"
) -> Union[SimpleTokenizerV1, Any]:
    """
    Create a tokenizer for text processing.

    Args:
        text: Raw text to build vocabulary from (required for SimpleTokenizerV1)
        use_bpe: Whether to use BPE tokenizer (tiktoken) or SimpleTokenizerV1
        encoding_name: Name of the BPE encoding to use (default: "gpt2")

    Returns:
        Tokenizer instance (either SimpleTokenizerV1 or tiktoken encoding)

    Raises:
        ValueError: If use_bpe=False but no text provided for SimpleTokenizerV1

    Examples:
        >>> # Create BPE tokenizer
        >>> tokenizer = create_tokenizer(use_bpe=True)

        >>> # Create simple tokenizer
        >>> tokenizer = create_tokenizer(text="sample text", use_bpe=False)
    """
    if use_bpe:
        return tiktoken.get_encoding(encoding_name)
    else:
        if text is None:
            raise ValueError("text parameter is required when use_bpe=False")
        return SimpleTokenizerV1(text)


def create_dataloader_v1(
    txt: str,
    tokenizer: Optional[Union[SimpleTokenizerV1, Any]] = None,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    use_bpe: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for GPT-style training data.

    This function creates a DataLoader that yields input-target pairs for training
    language models. It uses a sliding window approach to generate overlapping
    sequences from the input text.

    Args:
        txt: Raw text to tokenize and create samples from
        tokenizer: Custom tokenizer to use. If None, creates appropriate tokenizer
        batch_size: Number of samples per batch
        max_length: Maximum sequence length (context window)
        stride: Step size for sliding window (controls overlap)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of workers for data loading
        use_bpe: Whether to use BPE tokenizer when tokenizer is None

    Returns:
        DataLoader that yields (input_ids, target_ids) pairs

    Raises:
        ValueError: If max_length <= 0 or stride <= 0
        ValueError: If batch_size <= 0
        ValueError: If txt is empty

    Examples:
        >>> # Using default BPE tokenizer
        >>> dataloader = create_dataloader_v1(text)

        >>> # Using custom tokenizer
        >>> custom_tokenizer = SimpleTokenizerV1(text)
        >>> dataloader = create_dataloader_v1(text, tokenizer=custom_tokenizer)

        >>> # With specific parameters
        >>> dataloader = create_dataloader_v1(
        ...     text,
        ...     batch_size=8,
        ...     max_length=512,
        ...     stride=256
        ... )
    """
    # Parameter validation
    if not txt or not txt.strip():
        raise ValueError("txt parameter cannot be empty")

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(text=txt, use_bpe=use_bpe)

    # Create dataset using sliding window approach
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create DataLoader for efficient batching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
