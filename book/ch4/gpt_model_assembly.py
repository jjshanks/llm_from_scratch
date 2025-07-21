"""
Chapter 4 - Building the Complete GPT Model

This script demonstrates how to assemble a complete GPT model by combining
all the components we've built: embeddings, transformer blocks, and output
layers. It also shows how to analyze model size and parameter distribution.

Key concepts demonstrated:
- GPT model architecture overview
- Token and position embeddings
- Stacking transformer blocks
- Output projection layer
- Parameter counting and memory analysis
- Testing with DummyGPTModel
- Understanding model capacity

This brings together everything from the chapter into a working model
that can process text and generate predictions.

Usage: uv run python book/ch4/gpt_model_assembly.py
"""

import torch
import torch.nn as nn
import tiktoken
from llm_from_scratch.model import GPTModel, DummyGPTModel
from configuration import GPT_CONFIG_124M


def explain_gpt_architecture():
    """Explain the complete GPT architecture."""
    print("=== Building the Complete GPT Model ===\n")

    print("GPT Architecture:")
    print("1. Token embeddings (convert IDs to vectors)")
    print("2. Position embeddings (encode token positions)")
    print("3. Dropout (regularization)")
    print("4. 12x Transformer blocks")
    print("5. Final layer norm")
    print("6. Output projection (to vocabulary size)")

    print("\nData flow through the model:")
    print("Token IDs → Embeddings → Transformer Blocks → Logits")
    print("  [B, T]  →  [B, T, E]  →    [B, T, E]     → [B, T, V]")
    print("\nWhere:")
    print("  B = Batch size")
    print("  T = Sequence length (tokens)")
    print("  E = Embedding dimension")
    print("  V = Vocabulary size")


def visualize_model_architecture():
    """ASCII visualization of the complete GPT model."""
    print("\n\n=== GPT Model Architecture Visualization ===")

    print("\n    Input Token IDs [batch_size, seq_len]")
    print("            │")
    print("            ↓")
    print("    ┌───────────────┐")
    print("    │Token Embedding│ (vocab_size × emb_dim)")
    print("    └───────────────┘")
    print("            │")
    print("            ↓ [batch_size, seq_len, emb_dim]")
    print("            +")
    print("    ┌───────────────┐")
    print("    │Position Embed │ (max_length × emb_dim)")
    print("    └───────────────┘")
    print("            │")
    print("            ↓")
    print("    ┌───────────────┐")
    print("    │    Dropout    │")
    print("    └───────────────┘")
    print("            │")
    print("            ↓")
    print("    ╔═══════════════╗")
    print("    ║ Transformer   ║")
    print("    ║   Block 1     ║")
    print("    ╚═══════════════╝")
    print("            │")
    print("            ↓")
    print("    ╔═══════════════╗")
    print("    ║ Transformer   ║")
    print("    ║   Block 2     ║")
    print("    ╚═══════════════╝")
    print("            │")
    print("           ...")
    print("            │")
    print("            ↓")
    print("    ╔═══════════════╗")
    print("    ║ Transformer   ║")
    print("    ║   Block 12    ║")
    print("    ╚═══════════════╝")
    print("            │")
    print("            ↓")
    print("    ┌───────────────┐")
    print("    │ Final LayerNorm│")
    print("    └───────────────┘")
    print("            │")
    print("            ↓ [batch_size, seq_len, emb_dim]")
    print("    ┌───────────────┐")
    print("    │Output Projection│ (emb_dim × vocab_size)")
    print("    └───────────────┘")
    print("            │")
    print("            ↓")
    print("    Output Logits [batch_size, seq_len, vocab_size]")


def test_dummy_model():
    """Test the DummyGPTModel to understand data flow."""
    print("\n\n=== Testing DummyGPTModel (Placeholder Architecture) ===")

    print("\nThe DummyGPTModel uses placeholder components to show data flow")
    print("before implementing the real components.")

    # Create tokenizer and example batch
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    print(f"\nInput batch shape: {batch.shape}")
    print("(2 texts, 4 tokens each)")

    # Create and test dummy model
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    # Forward pass
    logits = model(batch)

    print(f"\nOutput shape: {logits.shape}")
    print("Interpretation: [batch_size=2, sequence_length=4, vocab_size=50257]")

    print("\nWhere does 50,257 come from?")
    print("- It's the GPT-2 tokenizer vocabulary size")
    print("- The tokenizer knows 50,257 different tokens")
    print("- So the model must predict which of these 50,257 tokens comes next")

    print("\nEach of the 4 tokens produces 50,257 scores (one per vocabulary word)")
    print("But wait - are these meaningful predictions? NO!")
    print("The model has RANDOM weights (not trained yet), so these are garbage values")

    return batch, tokenizer


def build_complete_model():
    """Build and analyze the complete GPT model."""
    print("\n\n=== Building Complete GPT Model ===")

    # Initialize the complete model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    print("\nModel created successfully!")
    print(
        f"Configuration: {GPT_CONFIG_124M['n_layers']} layers, "
        f"{GPT_CONFIG_124M['n_heads']} heads, "
        f"{GPT_CONFIG_124M['emb_dim']} embedding dim"
    )

    # Test with tokenized batch
    batch, tokenizer = test_dummy_model()

    print("\n\n=== Testing Complete Model ===")
    out = model(batch)
    print(f"\nInput batch (token IDs):\n{batch}")
    print(f"\nOutput shape: {out.shape}")
    print("Each position outputs 50,257 logits (unnormalized probabilities)")
    print("The highest logit indicates the predicted next token")

    return model


def analyze_model_size(model):
    """Analyze the model's parameter count and memory requirements."""
    print("\n\n=== Analyzing Model Size ===")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")

    print("\nWait, that's 163M, not 124M! Why?")

    # Analyze large layers
    print("\nLarge layers in the model:")
    print(f"Token embedding layer shape: {model.tok_emb.weight.shape}")
    print(f"Output layer shape: {model.out_head.weight.shape}")
    print("\nBoth are [50257, 768] = 38,597,376 parameters each!")
    print("These layers map between tokens and embeddings")

    # Explain weight tying
    print("\n--- Weight Tying ---")
    print("GPT-2 uses 'weight tying': sharing weights between embedding and output")
    print("Our implementation uses separate weights (often gives better results)")

    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(f"\nWith weight tying (like original GPT-2): {total_params_gpt2:,}")
    print("This matches the advertised 124M parameters!")

    # Memory analysis
    print("\n--- Memory Footprint ---")
    total_size_bytes = total_params * 4  # 32-bit floats
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size: {total_size_mb:.2f} MB")
    print("\nThis is just for storing parameters!")
    print("Training requires ~10-20x more memory for:")
    print("- Gradients (same size as parameters)")
    print("- Optimizer states (2x parameters for Adam)")
    print("- Activations (for backpropagation)")

    return total_params


def analyze_parameter_distribution(model):
    """Analyze how parameters are distributed across components."""
    print("\n\n=== Parameter Distribution Analysis ===")

    # Define component groups
    components = {
        "Token Embeddings": [model.tok_emb],
        "Position Embeddings": [model.pos_emb],
        "Transformer Blocks": model.trf_blocks,
        "Final LayerNorm": [model.final_norm],
        "Output Head": [model.out_head],
    }

    print(f"{'Component':<25} {'Parameters':>15} {'Percentage':>10}")
    print("-" * 52)

    total_params = sum(p.numel() for p in model.parameters())

    for name, modules in components.items():
        if isinstance(modules, nn.ModuleList):
            params = sum(p.numel() for module in modules for p in module.parameters())
        else:
            params = sum(p.numel() for module in modules for p in module.parameters())

        percentage = (params / total_params) * 100
        print(f"{name:<25} {params:>15,} {percentage:>9.1f}%")

    print("-" * 52)
    print(f"{'Total':<25} {total_params:>15,}")

    # Analyze transformer blocks in detail
    print("\n--- Transformer Block Breakdown ---")
    block = model.trf_blocks[0]  # Analyze first block

    block_components = {
        "Layer Norm 1": block.ln1,
        "Multi-Head Attention": block.attn,
        "Layer Norm 2": block.ln2,
        "Feed Forward Network": block.ff,
    }

    block_total = sum(p.numel() for p in block.parameters())
    print(f"\nEach transformer block has {block_total:,} parameters:")

    for name, module in block_components.items():
        params = sum(p.numel() for p in module.parameters())
        percentage = (params / block_total) * 100
        print(f"  {name:<25} {params:>10,} ({percentage:.1f}%)")


def exercise_4_1_parameter_comparison(model):
    """
    Exercise 4.1 Solution: Compare parameters in FFN vs Attention.

    From Chapter 4: "Calculate and compare the number of parameters that are
    contained in the feed forward module and those that are contained in the
    multi-head attention module."
    """
    print("\n\n=== Exercise 4.1: FFN vs Attention Parameter Comparison ===")

    # Get a transformer block
    block = model.trf_blocks[0]

    # Count parameters in attention
    attn_params = sum(p.numel() for p in block.attn.parameters())

    # Count parameters in FFN
    ffn_params = sum(p.numel() for p in block.ff.parameters())

    print("\nPer transformer block:")
    print(f"Multi-Head Attention: {attn_params:>10,} parameters")
    print(f"Feed Forward Network: {ffn_params:>10,} parameters")
    print(f"Ratio (FFN/Attention): {ffn_params/attn_params:>10.2f}x")

    print("\nDetailed breakdown:")

    # Attention details
    print("\nMulti-Head Attention:")
    print(f"  - Input dimension: {GPT_CONFIG_124M['emb_dim']}")
    print(f"  - Number of heads: {GPT_CONFIG_124M['n_heads']}")
    print(
        f"  - Q, K, V projections: 3 × {GPT_CONFIG_124M['emb_dim']} × {GPT_CONFIG_124M['emb_dim']} = {3 * GPT_CONFIG_124M['emb_dim'] * GPT_CONFIG_124M['emb_dim']:,}"
    )
    print(
        f"  - Output projection: {GPT_CONFIG_124M['emb_dim']} × {GPT_CONFIG_124M['emb_dim']} = {GPT_CONFIG_124M['emb_dim'] * GPT_CONFIG_124M['emb_dim']:,}"
    )
    print(f"  - Total: {attn_params:,}")

    # FFN details
    print("\nFeed Forward Network:")
    print(
        f"  - Input → Hidden: {GPT_CONFIG_124M['emb_dim']} × {4 * GPT_CONFIG_124M['emb_dim']} = {GPT_CONFIG_124M['emb_dim'] * 4 * GPT_CONFIG_124M['emb_dim']:,}"
    )
    print(
        f"  - Hidden → Output: {4 * GPT_CONFIG_124M['emb_dim']} × {GPT_CONFIG_124M['emb_dim']} = {4 * GPT_CONFIG_124M['emb_dim'] * GPT_CONFIG_124M['emb_dim']:,}"
    )
    print(
        f"  - Biases: {4 * GPT_CONFIG_124M['emb_dim']} + {GPT_CONFIG_124M['emb_dim']} = {5 * GPT_CONFIG_124M['emb_dim']:,}"
    )
    print(f"  - Total: {ffn_params:,}")

    print("\nKey insight:")
    print("• FFN has ~2x more parameters than attention")
    print("• This is because FFN expands to 4x dimension internally")
    print("• Most model parameters are in FFN layers!")


def exercise_4_2_all_model_configs():
    """
    Exercise 4.2 Complete Solution: Initialize all GPT-2 model sizes.

    From Chapter 4: "Use the GPTModel class to implement GPT-2 medium,
    large, and XL. As a bonus, calculate the total number of parameters
    in each GPT model."
    """
    print("\n\n=== Exercise 4.2: Initializing All GPT-2 Model Sizes ===")

    # Configuration for all models
    configs = {
        "GPT-2 Small (124M)": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False,
        },
        "GPT-2 Medium (355M)": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "drop_rate": 0.1,
            "qkv_bias": False,
        },
        "GPT-2 Large (774M)": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20,
            "n_layers": 36,
            "drop_rate": 0.1,
            "qkv_bias": False,
        },
        "GPT-2 XL (1.5B)": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1600,
            "n_heads": 25,
            "n_layers": 48,
            "drop_rate": 0.1,
            "qkv_bias": False,
        },
    }

    print("\nInitializing and analyzing all GPT-2 models:")
    print(f"{'Model':<20} {'Layers':<8} {'Dim':<8} {'Heads':<8} {'Parameters':>15}")
    print("-" * 60)

    for name, config in configs.items():
        # Initialize model (just for parameter counting)
        model = GPTModel(config)
        total_params = sum(p.numel() for p in model.parameters())

        print(
            f"{name:<20} {config['n_layers']:<8} {config['emb_dim']:<8} "
            f"{config['n_heads']:<8} {total_params:>15,}"
        )

    print("\nNote: These are the actual parameter counts including")
    print("separate embedding and output layers (no weight tying).")


def explain_embeddings():
    """Explain token and position embeddings in detail."""
    print("\n\n=== Understanding Embeddings ===")

    print("\n**Token Embeddings:**")
    print("- Converts token IDs to dense vectors")
    print("- Learned lookup table: [vocab_size, embedding_dim]")
    print("- Each token ID maps to a unique vector")
    print("- These vectors are learned during training")

    print("\n**Position Embeddings:**")
    print("- Encodes position information (1st word, 2nd word, etc.)")
    print("- Learned embeddings: [max_length, embedding_dim]")
    print("- Added to token embeddings")
    print("- Allows model to understand word order")

    print("\nWhy both?")
    print("- Token embedding: 'what' the token is")
    print("- Position embedding: 'where' the token is")
    print("- Combined: Complete representation of token + position")

    # Demonstrate embeddings
    model = GPTModel(GPT_CONFIG_124M)

    print("\n--- Embedding Shapes ---")
    print(f"Token embedding matrix: {model.tok_emb.weight.shape}")
    print(f"Position embedding matrix: {model.pos_emb.weight.shape}")

    # Example embedding lookup
    token_id = 1234
    position = 5

    with torch.no_grad():
        token_emb = model.tok_emb.weight[token_id]
        pos_emb = model.pos_emb.weight[position]
        combined = token_emb + pos_emb

    print("\nExample lookup:")
    print(f"Token ID {token_id} → embedding vector of size {token_emb.shape}")
    print(f"Position {position} → embedding vector of size {pos_emb.shape}")
    print(f"Combined → vector of size {combined.shape}")


def explain_output_projection():
    """Explain the output projection layer."""
    print("\n\n=== Output Projection Layer ===")

    print("\nThe final layer projects from embedding space to vocabulary:")
    print("- Input: [batch, seq_len, embedding_dim]")
    print("- Weight matrix: [embedding_dim, vocab_size]")
    print("- Output: [batch, seq_len, vocab_size]")

    print("\nThis produces 'logits' - unnormalized scores for each token")
    print("Higher logit = model thinks this token is more likely")

    print("\nTo get probabilities:")
    print("1. Apply softmax to logits")
    print("2. This gives probability distribution over vocabulary")
    print("3. Sum of all probabilities = 1.0")

    print("\nDuring generation:")
    print("1. Get logits for last position")
    print("2. Convert to probabilities")
    print("3. Sample or take argmax")
    print("4. Append chosen token and repeat")


def model_capacity_discussion():
    """Discuss model capacity and scaling."""
    print("\n\n=== Model Capacity and Scaling ===")

    print("\nWhat determines model capacity?")
    print("1. **Number of parameters** - more parameters = more patterns")
    print("2. **Layer depth** - deeper = more abstract representations")
    print("3. **Hidden size** - wider = more expressive power")
    print("4. **Attention heads** - more heads = more relationships")

    print("\n--- Scaling Laws ---")
    print("Research shows predictable relationships:")
    print("• 10x parameters → ~constant improvement in loss")
    print("• But 10x parameters → 10x memory and compute!")
    print("• Sweet spots depend on available resources")

    print("\n--- GPT Model Sizes ---")
    print("GPT-1:     117M parameters")
    print("GPT-2:     124M - 1.5B parameters")
    print("GPT-3:     175B parameters")
    print("GPT-4:     ~1.8T parameters (estimated)")

    print("\nOur 124M model is like GPT-2 small:")
    print("• Can learn language patterns")
    print("• Can generate coherent text")
    print("• Limited by capacity for complex reasoning")
    print("• Good for learning and experimentation!")


if __name__ == "__main__":
    # Turn off scientific notation for cleaner output
    torch.set_printoptions(sci_mode=False)

    print("=" * 60)
    print("Chapter 4: Building the Complete GPT Model")
    print("=" * 60)

    # Architecture overview
    explain_gpt_architecture()

    # Visualization
    visualize_model_architecture()

    # Build the model
    model = build_complete_model()

    # Analyze size
    total_params = analyze_model_size(model)

    # Parameter distribution
    analyze_parameter_distribution(model)

    # Exercise 4.1: FFN vs Attention comparison
    exercise_4_1_parameter_comparison(model)

    # Exercise 4.2: All model sizes
    exercise_4_2_all_model_configs()

    # Explain components
    explain_embeddings()
    explain_output_projection()

    # Capacity discussion
    model_capacity_discussion()

    print("\n" + "=" * 60)
    print("GPT model successfully assembled!")
    print("Ready for training (Chapter 5) and generation!")
    print("=" * 60)
