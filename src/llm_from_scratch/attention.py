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


class CausalAttention(nn.Module):
    """
    Causal self-attention mechanism for autoregressive models like GPT.

    This implements the masked self-attention mechanism that prevents the model
    from accessing future tokens when generating sequences. The causal mask ensures
    that predictions for position i can only depend on positions less than i.

    Key features:
    1. Causal masking: Prevents information leakage from future tokens
    2. Scaled dot-product attention: Divides scores by sqrt(d_k) for stability
    3. Dropout regularization: Applied to attention weights during training
    4. Efficient batched computation: Handles multiple sequences simultaneously

    Args:
        d_in: Input dimension (embedding size of input tokens)
        d_out: Output dimension (embedding size of output context vectors)
        context_length: Maximum sequence length for the causal mask
        dropout: Dropout probability for attention weights regularization
        qkv_bias: Whether to include bias terms in query/key/value projections
    """

    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Initialize the causal attention module.

        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension for queries, keys, and values
            context_length: Maximum sequence length (determines mask size)
            dropout: Dropout rate for attention weights (0.0 to 1.0)
            qkv_bias: If True, adds learnable bias to query, key, value projections
        """
        super().__init__()
        self.d_out = d_out

        # Linear transformations for queries, keys, and values
        # Using nn.Linear for optimized initialization and optional bias
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout for regularization during training
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (moves with model to device)
        # Upper triangular matrix with 1s above diagonal for masking
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Apply causal self-attention to input sequences.

        The attention computation follows these steps:
        1. Project inputs to queries, keys, and values
        2. Compute attention scores via batched matrix multiplication
        3. Apply causal mask to prevent attending to future positions
        4. Normalize scores with scaled softmax
        5. Apply dropout for regularization
        6. Weight values by attention to get context vectors

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Context vectors of shape (batch_size, seq_len, d_out)
        """
        b, num_tokens, d_in = x.shape

        # Project inputs to queries, keys, and values
        keys = self.W_key(x)  # Shape: (batch, seq_len, d_out)
        queries = self.W_query(x)  # Shape: (batch, seq_len, d_out)
        values = self.W_value(x)  # Shape: (batch, seq_len, d_out)

        # Compute attention scores for all pairs of positions
        # Transpose keys for batched matrix multiplication
        attn_scores = queries @ keys.transpose(1, 2)  # Shape: (batch, seq_len, seq_len)

        # Apply causal mask: set scores to -inf where attending to future
        # This ensures softmax gives ~0 probability to masked positions
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Apply scaled softmax to get normalized attention weights
        # Scaling by 1/sqrt(d_k) prevents vanishing gradients with large d_out
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Apply dropout to attention weights for regularization
        # Note: dropout is only active during training
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values using attention weights
        context_vec = attn_weights @ values  # Shape: (batch, seq_len, d_out)
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
