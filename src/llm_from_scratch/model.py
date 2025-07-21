# Chapter 4: Implementing a GPT model from scratch to generate text
# This module implements the core GPT architecture components from the book

from llm_from_scratch.attention import MultiHeadAttention
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    """Placeholder GPT model architecture (Listing 4.1)

    This is a simplified version showing the overall structure of a GPT model
    before implementing the actual transformer blocks and layer normalization.
    Used to illustrate the high-level architecture.
    """

    def __init__(self, cfg):
        super().__init__()
        # Token embeddings: converts token IDs to dense vectors
        # Shape: (vocab_size, emb_dim) - e.g., (50257, 768) for GPT-2 small
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional embeddings: adds position information to tokens
        # Shape: (context_length, emb_dim) - e.g., (1024, 768)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout for regularization on embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of transformer blocks - the core of the GPT architecture
        # n_layers = 12 for GPT-2 small (124M parameters)
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization before output projection
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # Output projection layer: projects from embedding dimension to vocabulary size
        # Shape: (emb_dim, vocab_size) - e.g., (768, 50257)
        # bias=False follows modern LLM conventions
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx shape: (batch_size, seq_len) - token IDs
        batch_size, seq_len = in_idx.shape

        # Convert token IDs to embeddings
        # Shape: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # Create position embeddings for each position in sequence
        # Shape: (seq_len, emb_dim), broadcasted to (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine token and positional embeddings
        x = tok_embeds + pos_embeds

        # Apply dropout for regularization
        x = self.drop_emb(x)

        # Pass through transformer blocks
        x = self.trf_blocks(x)

        # Apply final layer normalization
        x = self.final_norm(x)

        # Project to vocabulary size for next token prediction
        # Output shape: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    """Placeholder transformer block

    This dummy implementation simply returns the input unchanged.
    Will be replaced by the actual TransformerBlock implementation.
    """

    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Pass through unchanged - placeholder behavior
        return x


class DummyLayerNorm(nn.Module):
    """Placeholder layer normalization

    This dummy implementation simply returns the input unchanged.
    Will be replaced by the actual LayerNorm implementation.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        # Pass through unchanged - placeholder behavior
        return x


class LayerNorm(nn.Module):
    """Layer Normalization (Listing 4.2)

    Normalizes activations to have mean 0 and variance 1, which helps
    stabilize training in deep networks. Applied before attention and
    feed-forward layers in transformer blocks.
    """

    def __init__(self, emb_dim):
        super().__init__()
        # Small epsilon to prevent division by zero
        self.eps = 1e-5

        # Learnable scale parameter (gamma) - initialized to 1
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # Learnable shift parameter (beta) - initialized to 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Calculate mean across embedding dimension
        # keepdim=True maintains the tensor shape for broadcasting
        mean = x.mean(dim=-1, keepdim=True)

        # Calculate variance (unbiased=False for compatibility with GPT-2)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: (x - mean) / sqrt(variance + epsilon)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift parameters
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function (Listing 4.3)

    A smooth activation function used in GPT models instead of ReLU.
    Approximates: GELU(x) = x * Φ(x) where Φ is the Gaussian CDF.
    This implementation uses the approximation from the original GPT-2.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        # This smooth activation allows small gradients for negative values
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Feed-forward network module (Listing 4.4)

    A two-layer neural network with GELU activation that processes
    each position independently. Expands dimensionality by 4x internally
    to allow the model to learn richer representations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # First layer: expand from emb_dim to 4*emb_dim
            # E.g., 768 -> 3072 for GPT-2 small
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # GELU activation for non-linearity
            GELU(),
            # Second layer: project back to emb_dim
            # E.g., 3072 -> 768 for GPT-2 small
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    """Example deep neural network to demonstrate shortcut connections (Listing 4.5)

    This class illustrates how shortcut (residual) connections help
    mitigate the vanishing gradient problem in deep networks.
    """

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # Process input through layer
            layer_output = layer(x)

            # Add shortcut connection if enabled and shapes match
            if self.use_shortcut and x.shape == layer_output.shape:
                # Residual connection: add input to output
                x = x + layer_output
            else:
                # No shortcut: just use layer output
                x = layer_output
        return x


class TransformerBlock(nn.Module):
    """Transformer block - the core building block of GPT (Listing 4.6)

    Combines multi-head attention with a feed-forward network, using
    layer normalization and residual connections. This block is repeated
    multiple times (e.g., 12 times for GPT-2 small) to form the full model.
    """

    def __init__(self, cfg):
        super().__init__()
        # Multi-head attention for capturing token relationships
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],  # e.g., 12 heads for GPT-2 small
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )  # Usually False for modern LLMs

        # Feed-forward network for position-wise transformations
        self.ff = FeedForward(cfg)

        # Layer normalization for attention and feed-forward sublayers
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout for residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # First sub-block: Multi-head attention with residual connection
        # Pre-LayerNorm: normalize before attention (modern approach)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Apply masked multi-head attention
        x = self.drop_shortcut(x)  # Dropout for regularization
        x = x + shortcut  # Add residual connection

        # Second sub-block: Feed-forward network with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # Apply feed-forward network
        x = self.drop_shortcut(x)  # Dropout for regularization
        x = x + shortcut  # Add residual connection
        return x


class GPTModel(nn.Module):
    """Complete GPT model architecture (Listing 4.7)

    This is the full GPT implementation combining all components:
    - Token and positional embeddings
    - Stack of transformer blocks
    - Final layer normalization
    - Output projection to vocabulary
    """

    def __init__(self, cfg):
        super().__init__()
        # Token embeddings: converts token IDs to dense vectors
        # Shape: (vocab_size, emb_dim) - e.g., (50257, 768) for GPT-2
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Positional embeddings: encodes position information
        # Shape: (context_length, emb_dim) - e.g., (1024, 768)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout for embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of transformer blocks (e.g., 12 for GPT-2 small)
        # This is where the main computation happens
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization before output projection
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output head: projects from embedding dimension to vocabulary size
        # No bias for compatibility with modern LLMs
        # Shape: (emb_dim, vocab_size) - e.g., (768, 50257)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx shape: (batch_size, seq_len) containing token IDs
        batch_size, seq_len = in_idx.shape

        # Get token embeddings
        # Shape: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # Create positional embeddings for sequence positions
        # torch.arange creates [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len, emb_dim), broadcasted to (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine token and positional information
        x = tok_embeds + pos_embeds

        # Apply embedding dropout for regularization
        x = self.drop_emb(x)

        # Process through all transformer blocks
        # Each block refines the representation using attention and FFN
        x = self.trf_blocks(x)

        # Final normalization for training stability
        x = self.final_norm(x)

        # Project to vocabulary size for next-token prediction
        # Output shape: (batch_size, seq_len, vocab_size)
        # Each position outputs scores for all possible next tokens
        logits = self.out_head(x)
        return logits
