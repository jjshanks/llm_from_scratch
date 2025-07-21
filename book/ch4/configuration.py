"""
Chapter 4 - GPT Model Configuration

This script defines the configuration parameters for building a GPT-2 style model.
The configuration shown here creates a 124-million parameter model, which is
small enough to train on consumer hardware while still being powerful.

Key concepts demonstrated:
- Model hyperparameters and their meanings
- How configuration drives model architecture
- Trade-offs between model size and computational requirements

The configuration parameters determine:
- Model capacity (how much the model can learn)
- Memory requirements (GPU RAM needed)
- Training time (larger models train slower)
- Inference speed (larger models generate text slower)

Usage: uv run python book/ch4/configuration.py
"""

# GPT Model Configuration for 124M parameter model
# This configuration defines a model similar in size to GPT-2 small
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Number of unique tokens the model can understand
    # (matches the GPT-2 tokenizer vocabulary)
    "context_length": 1024,  # Maximum number of tokens the model can process at once
    # (longer texts must be split into chunks)
    "emb_dim": 768,  # Size of token embeddings (higher = more expressive)
    # Each token becomes a 768-dimensional vector
    "n_heads": 12,  # Number of attention heads for multi-head attention
    # More heads = can attend to different aspects simultaneously
    "n_layers": 12,  # Number of transformer blocks stacked on top of each other
    # More layers = deeper model, more complex patterns
    "drop_rate": 0.1,  # Dropout rate (10% of connections randomly disabled)
    # Prevents overfitting during training
    "qkv_bias": False,  # Whether to use bias in query/key/value projections
    # Modern LLMs often skip this for efficiency
}


def print_config_details(config):
    """Print detailed explanation of each configuration parameter."""
    print("=== GPT Model Configuration Details ===\n")

    print(f"vocab_size: {config['vocab_size']:,}")
    print("  - Total number of unique tokens (words/subwords/characters)")
    print("  - GPT-2 uses byte-pair encoding with 50,257 tokens")
    print("  - Includes special tokens like <|endoftext|>\n")

    print(f"context_length: {config['context_length']:,}")
    print("  - Maximum sequence length the model can process")
    print("  - Texts longer than this must be split")
    print("  - Quadratic memory scaling with context length\n")

    print(f"emb_dim: {config['emb_dim']}")
    print("  - Dimensionality of token embeddings")
    print("  - Higher dimensions can capture more nuanced meanings")
    print("  - Must be divisible by n_heads for multi-head attention\n")

    print(f"n_heads: {config['n_heads']}")
    print("  - Number of parallel attention mechanisms")
    print("  - Each head learns different aspects of relationships")
    print(f"  - Each head uses {config['emb_dim'] // config['n_heads']} dimensions\n")

    print(f"n_layers: {config['n_layers']}")
    print("  - Number of transformer blocks in the model")
    print("  - Each layer refines the representation")
    print("  - Deeper models can learn more complex patterns\n")

    print(f"drop_rate: {config['drop_rate']}")
    print("  - Probability of dropping connections during training")
    print("  - Helps prevent overfitting")
    print("  - Set to 0.0 during inference\n")

    print(f"qkv_bias: {config['qkv_bias']}")
    print("  - Whether to include bias terms in attention projections")
    print("  - Many modern models skip bias for efficiency")
    print("  - Can slightly improve performance if True")


def estimate_model_size(config):
    """Estimate the model's parameter count and memory requirements."""
    print("\n=== Model Size Estimation ===\n")

    # Token and position embeddings
    token_embedding_params = config["vocab_size"] * config["emb_dim"]
    position_embedding_params = config["context_length"] * config["emb_dim"]

    # Transformer block parameters
    # Each block has: attention (Q,K,V,O projections) + FFN (2 linear layers)
    d_in = config["emb_dim"]
    d_out = config["emb_dim"]

    # Attention parameters per block
    qkv_params = 3 * d_in * d_out  # Q, K, V projections
    output_proj_params = d_out * d_in  # Output projection
    attention_params = qkv_params + output_proj_params

    # FFN parameters per block (4x expansion)
    ffn_hidden_dim = 4 * d_in
    ffn_params = d_in * ffn_hidden_dim + ffn_hidden_dim * d_in

    # Layer norm parameters (2 per block + 1 final)
    ln_params = 2 * d_in  # Scale and shift parameters
    ln_params_per_block = 2 * ln_params  # Two layer norms per block

    # Total per block
    params_per_block = attention_params + ffn_params + ln_params_per_block

    # All transformer blocks
    total_transformer_params = params_per_block * config["n_layers"]

    # Final layer norm and output projection
    final_ln_params = ln_params
    output_projection_params = config["emb_dim"] * config["vocab_size"]

    # Total parameters
    total_params = (
        token_embedding_params
        + position_embedding_params
        + total_transformer_params
        + final_ln_params
        + output_projection_params
    )

    print(f"Token embeddings:     {token_embedding_params:,} parameters")
    print(f"Position embeddings:  {position_embedding_params:,} parameters")
    print(f"Transformer blocks:   {total_transformer_params:,} parameters")
    print(f"  - Per block:        {params_per_block:,} parameters")
    print(f"  - Attention:        {attention_params:,} parameters")
    print(f"  - FFN:              {ffn_params:,} parameters")
    print(f"  - LayerNorm:        {ln_params_per_block:,} parameters")
    print(f"Final layer norm:     {final_ln_params:,} parameters")
    print(f"Output projection:    {output_projection_params:,} parameters")
    print(f"\nTotal parameters:     {total_params:,}")

    # Memory estimation (4 bytes per parameter for float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"\nMemory for parameters: {memory_mb:.1f} MB")
    print("\nNote: Training requires ~10-20x more memory for:")
    print("  - Gradients (same size as parameters)")
    print("  - Optimizer states (2x for Adam)")
    print("  - Activations for backpropagation")


def compare_model_sizes():
    """Compare different GPT model configurations."""
    print("\n=== Comparison with Other GPT Models ===\n")

    configs = {
        "GPT-2 Small (124M)": {"n_layers": 12, "emb_dim": 768, "n_heads": 12},
        "GPT-2 Medium (355M)": {"n_layers": 24, "emb_dim": 1024, "n_heads": 16},
        "GPT-2 Large (774M)": {"n_layers": 36, "emb_dim": 1280, "n_heads": 20},
        "GPT-2 XL (1.5B)": {"n_layers": 48, "emb_dim": 1600, "n_heads": 25},
    }

    print(f"{'Model':<20} {'Layers':<8} {'Dim':<8} {'Heads':<8}")
    print("-" * 44)
    for name, cfg in configs.items():
        print(
            f"{name:<20} {cfg['n_layers']:<8} {cfg['emb_dim']:<8} {cfg['n_heads']:<8}"
        )

    print("\nScaling observations:")
    print("- Larger models have more layers (depth)")
    print("- Larger models have wider embeddings (width)")
    print("- More heads allow finer-grained attention")
    print("- Parameter count scales roughly with depth × width²")


def create_gpt_configs():
    """
    Exercise 4.2 Solution: Create configurations for all GPT-2 model sizes.

    From Chapter 4: "Without making any code modifications besides updating the
    configuration file, use the GPTModel class to implement GPT-2 medium, large, and XL."
    """
    print("\n\n=== Exercise 4.2: GPT-2 Model Configurations ===\n")

    # GPT-2 Small (already defined as GPT_CONFIG_124M)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    # GPT-2 Medium
    GPT_CONFIG_355M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    # GPT-2 Large
    GPT_CONFIG_774M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    # GPT-2 XL
    GPT_CONFIG_1558M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    configs = {
        "GPT-2 Small (124M)": GPT_CONFIG_124M,
        "GPT-2 Medium (355M)": GPT_CONFIG_355M,
        "GPT-2 Large (774M)": GPT_CONFIG_774M,
        "GPT-2 XL (1.5B)": GPT_CONFIG_1558M,
    }

    print("Complete configurations for all GPT-2 sizes:")
    print("\nYou can use any of these configurations with the GPTModel class:")
    print("model = GPTModel(GPT_CONFIG_355M)  # For GPT-2 Medium")
    print("model = GPTModel(GPT_CONFIG_774M)   # For GPT-2 Large")
    print("model = GPTModel(GPT_CONFIG_1558M)  # For GPT-2 XL")

    return configs


def calculate_all_model_sizes():
    """Calculate detailed parameters and memory for all GPT-2 models."""
    print("\n\n=== Detailed Parameter Count for All GPT-2 Models ===\n")

    configs = create_gpt_configs()

    print(
        f"{'Model':<20} {'Total Params':>15} {'Memory (MB)':>12} {'Training Memory (GB)':>20}"
    )
    print("-" * 70)

    for name, config in configs.items():
        # Calculate parameters (same logic as estimate_model_size)
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]
        n_layers = config["n_layers"]

        # Embeddings
        token_emb = vocab_size * emb_dim
        pos_emb = context_length * emb_dim

        # Per transformer block
        qkv_params = 3 * emb_dim * emb_dim
        output_proj = emb_dim * emb_dim
        attn_params = qkv_params + output_proj

        ffn_hidden = 4 * emb_dim
        ffn_params = emb_dim * ffn_hidden + ffn_hidden * emb_dim

        ln_params = 2 * emb_dim  # scale and shift
        ln_per_block = 2 * ln_params  # two layer norms per block

        params_per_block = attn_params + ffn_params + ln_per_block
        all_blocks = params_per_block * n_layers

        # Final layer norm and output
        final_ln = ln_params
        output_layer = emb_dim * vocab_size

        # Total
        total_params = token_emb + pos_emb + all_blocks + final_ln + output_layer

        # Memory calculations
        param_memory_mb = (total_params * 4) / (1024 * 1024)

        # Training memory estimate (very rough)
        # Includes: params + gradients + optimizer states + activations
        training_memory_gb = (total_params * 4 * 15) / (1024 * 1024 * 1024)

        print(
            f"{name:<20} {total_params:>15,} {param_memory_mb:>11.1f} {training_memory_gb:>19.1f}"
        )

    print("\nNotes:")
    print("- Parameter memory assumes 32-bit floats (4 bytes each)")
    print("- Training memory is a rough estimate (15x parameter memory)")
    print("- Actual training memory depends on batch size and sequence length")
    print("- GPT-2 models use weight tying, reducing actual parameters by ~38M")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: GPT Model Configuration")
    print("=" * 60)

    # Display the configuration
    print("\nGPT_CONFIG_124M = {")
    for key, value in GPT_CONFIG_124M.items():
        print(f"    '{key}': {value},")
    print("}\n")

    # Detailed explanations
    print_config_details(GPT_CONFIG_124M)

    # Model size estimation
    estimate_model_size(GPT_CONFIG_124M)

    # Comparison with other models
    compare_model_sizes()

    # Exercise 4.2: All GPT-2 configurations
    create_gpt_configs()

    # Detailed size calculations for all models
    calculate_all_model_sizes()

    print("\n" + "=" * 60)
    print("This configuration will be used throughout Chapter 4")
    print("to build a complete GPT model from scratch!")
    print("=" * 60)
