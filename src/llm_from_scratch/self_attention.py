import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """
    Simple self-attention mechanism using randomly initialized weight matrices.

    This implementation uses basic nn.Parameter for weight matrices with random
    initialization. It demonstrates the core self-attention computation:
    1. Transform inputs to queries, keys, and values
    2. Compute attention scores via dot product
    3. Apply scaled softmax to get attention weights
    4. Weight values by attention to get context vectors

    Args:
        d_in: Input dimension (embedding size of input tokens)
        d_out: Output dimension (embedding size of output context vectors)
    """

    def __init__(self, d_in, d_out):
        """
        Initialize the self-attention module with random weight matrices.

        Args:
            d_in: Input dimension size
            d_out: Output dimension size for queries, keys, and values
        """
        super().__init__()
        # Initialize weight matrices for queries, keys, and values
        # Using random initialization (not optimal for training)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        Compute self-attention for input sequence.

        Args:
            x: Input tensor of shape (seq_len, d_in) or (batch_size, seq_len, d_in)

        Returns:
            Context vectors of shape (seq_len, d_out) or (batch_size, seq_len, d_out)
        """
        # Transform inputs to keys, queries, and values
        keys = x @ self.W_key  # Shape: (..., seq_len, d_out)
        queries = x @ self.W_query  # Shape: (..., seq_len, d_out)
        values = x @ self.W_value  # Shape: (..., seq_len, d_out)

        # Compute attention scores (omega) via dot product of queries and keys
        attn_scores = queries @ keys.T  # Shape: (..., seq_len, seq_len)

        # Apply scaled softmax to get normalized attention weights
        # Scaling by sqrt(d_k) prevents gradients from becoming too small
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Compute weighted sum of values to get context vectors
        context_vec = attn_weights @ values  # Shape: (..., seq_len, d_out)
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Improved self-attention mechanism using nn.Linear layers.

    This implementation uses nn.Linear instead of nn.Parameter, which provides:
    1. Better weight initialization (Xavier/Glorot initialization)
    2. Optional bias terms
    3. More stable training dynamics

    This is the preferred implementation for practical use in transformers.

    Args:
        d_in: Input dimension (embedding size of input tokens)
        d_out: Output dimension (embedding size of output context vectors)
        qkv_bias: Whether to include bias terms in linear transformations
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        Initialize the self-attention module with linear layers.

        Args:
            d_in: Input dimension size
            d_out: Output dimension size for queries, keys, and values
            qkv_bias: If True, adds learnable bias to query, key, value projections
        """
        super().__init__()
        # Use nn.Linear for better initialization and optional bias
        # Note: nn.Linear uses optimized weight initialization by default
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Compute self-attention for input sequence.

        Args:
            x: Input tensor of shape (seq_len, d_in) or (batch_size, seq_len, d_in)

        Returns:
            Context vectors of shape (seq_len, d_out) or (batch_size, seq_len, d_out)
        """
        # Transform inputs using linear layers (includes matmul + optional bias)
        keys = self.W_key(x)  # Shape: (..., seq_len, d_out)
        queries = self.W_query(x)  # Shape: (..., seq_len, d_out)
        values = self.W_value(x)  # Shape: (..., seq_len, d_out)

        # Compute attention scores via dot product
        attn_scores = queries @ keys.T  # Shape: (..., seq_len, seq_len)

        # Apply scaled softmax for stable gradients
        # The scaling factor 1/sqrt(d_k) is crucial for training stability
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Apply attention weights to values
        context_vec = attn_weights @ values  # Shape: (..., seq_len, d_out)
        return context_vec
