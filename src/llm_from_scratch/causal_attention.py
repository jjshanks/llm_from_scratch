import torch
import torch.nn as nn


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
