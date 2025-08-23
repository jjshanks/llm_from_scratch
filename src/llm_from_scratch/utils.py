"""
Utility functions for LLM training and data processing.

This module provides convenient functions for creating DataLoaders, managing tokenizers,
and other common operations used throughout the LLM training pipeline.
"""

from typing import Any, Optional, Union
import torch
import torch.nn.functional as F
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


def generate_text_simple(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Simple greedy text generation.

    Args:
        model: The GPT model.
        idx: Starting token IDs of shape (batch_size, sequence_length).
        max_new_tokens: The number of new tokens to generate.
        context_size: The maximum context length for the model.

    Returns:
        The generated token IDs of shape (batch_size, sequence_length + max_new_tokens).
    """
    model.eval()  # Set model to evaluation mode

    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions (no gradient needed for inference)
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last token's predictions
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Pick the most likely next token (greedy decoding)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate loss for a single batch.

    Args:
        input_batch: Input token IDs of shape (batch_size, sequence_length)
        target_batch: Target token IDs of shape (batch_size, sequence_length)
        model: The GPT model
        device: Device to run computation on

    Returns:
        Cross-entropy loss tensor
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over entire data loader.

    Args:
        data_loader: PyTorch DataLoader with (input, target) pairs
        model: The GPT model
        device: Device to run computation on
        num_batches: Optional limit on number of batches to evaluate

    Returns:
        Average cross-entropy loss as float
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on training and validation sets.

    Args:
        model: The GPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run computation on
        eval_iter: Number of batches to evaluate on each dataset

    Returns:
        Tuple of (train_loss, val_loss) as floats
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Return to training mode
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs tensor.

    Args:
        text: Raw text string to encode
        tokenizer: Tokenizer instance (usually tiktoken encoding)

    Returns:
        Token IDs tensor of shape (1, sequence_length)
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs tensor to text.

    Args:
        token_ids: Token IDs tensor of shape (1, sequence_length) or (sequence_length,)
        tokenizer: Tokenizer instance (usually tiktoken encoding)

    Returns:
        Decoded text string
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_with_temperature(
    model,
    tokenizer,
    device,
    start_context,
    max_new_tokens=25,
    temperature=1.0,
    top_k=None,
):
    """
    Generate text with advanced sampling techniques.

    Args:
        model: The GPT model
        tokenizer: Tokenizer instance (usually tiktoken encoding)
        device: Device to run computation on
        start_context: Starting text string
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more diverse)
        top_k: Limit to top-k tokens for sampling (None = no limit)

    Returns:
        Generated text string including the start_context
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = encoded

        for _ in range(max_new_tokens):
            # Crop to context size
            idx_cond = token_ids[:, -context_size:]

            # Get predictions
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last position

            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            # Apply temperature and sampling
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            token_ids = torch.cat((token_ids, idx_next), dim=1)

    model.train()
    return token_ids_to_text(token_ids, tokenizer)
