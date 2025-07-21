"""
Chapter 4 - Transformer Block Assembly

This script demonstrates how to assemble a complete transformer block by
combining all the components we've studied: multi-head attention, feed
forward networks, layer normalization, and shortcut connections.

Key concepts demonstrated:
- Transformer block architecture
- How components fit together
- Pre-norm vs post-norm design
- Shape preservation for stacking
- Component interaction and data flow

The transformer block is the fundamental building unit of GPT models.
By stacking multiple blocks, we create deep transformer architectures.

Usage: uv run python book/ch4/transformer_block.py
"""

import torch
from llm_from_scratch.model import (
    TransformerBlock,
)
from configuration import GPT_CONFIG_124M


def explain_transformer_block():
    """Explain the transformer block architecture."""
    print("=== Transformer Block Architecture ===\n")

    print("Each transformer block contains:")
    print("1. Multi-head attention (from Chapter 3)")
    print("2. Feed forward network")
    print("3. Two layer normalizations")
    print("4. Two shortcut connections")

    print("\nGPT stacks 12 of these blocks for the 124M model")

    print("\nWhy this specific arrangement?")
    print("• Attention: Gathers information from all positions")
    print("• FFN: Processes the gathered information")
    print("• LayerNorm: Keeps activations stable")
    print("• Shortcuts: Enable deep stacking")


def visualize_block_architecture():
    """ASCII visualization of transformer block."""
    print("\n\n=== Transformer Block Visualization ===")

    print("\n     Input (shape: [batch, seq_len, emb_dim])")
    print("       │")
    print("       ├──────────────────┐ (Shortcut 1)")
    print("       │                  │")
    print("       ↓                  │")
    print("  LayerNorm               │")
    print("       ↓                  │")
    print("  Multi-Head              │")
    print("  Attention               │")
    print("       ↓                  │")
    print("   Dropout                │")
    print("       ↓                  │")
    print("      (+)←────────────────┘")
    print("       │")
    print("       ├──────────────────┐ (Shortcut 2)")
    print("       │                  │")
    print("       ↓                  │")
    print("  LayerNorm               │")
    print("       ↓                  │")
    print(" Feed Forward             │")
    print("   Network                │")
    print("       ↓                  │")
    print("   Dropout                │")
    print("       ↓                  │")
    print("      (+)←────────────────┘")
    print("       │")
    print("       ↓")
    print("    Output (shape: [batch, seq_len, emb_dim])")

    print("\nKey insight: Input and output shapes are identical!")
    print("This allows stacking many blocks without shape mismatches")


def explain_component_roles():
    """Explain the role of each component in detail."""
    print("\n\n=== Component Roles in Detail ===")

    print("\n1. **First LayerNorm**")
    print("   - Normalizes input before attention")
    print("   - Stabilizes attention computation")
    print("   - Pre-norm design (modern standard)")

    print("\n2. **Multi-Head Attention**")
    print("   - Allows tokens to 'look at' each other")
    print("   - Multiple heads capture different relationships")
    print("   - Causal mask prevents looking at future tokens")

    print("\n3. **First Dropout**")
    print("   - Regularization after attention")
    print("   - Prevents attention from overfitting")
    print("   - Typically 10-20% dropout rate")

    print("\n4. **First Shortcut (+)**")
    print("   - Adds original input to attention output")
    print("   - Gradient highway for backpropagation")
    print("   - Enables learning residual functions")

    print("\n5. **Second LayerNorm**")
    print("   - Normalizes before FFN")
    print("   - Prepares stable input for FFN")

    print("\n6. **Feed Forward Network**")
    print("   - Processes each position independently")
    print("   - Expands to 4x dimension internally")
    print("   - Where most parameters live")

    print("\n7. **Second Dropout**")
    print("   - Regularization after FFN")
    print("   - Prevents FFN from memorizing")

    print("\n8. **Second Shortcut (+)**")
    print("   - Adds pre-FFN value to FFN output")
    print("   - Another gradient highway")


def demonstrate_transformer_block():
    """Demonstrate a transformer block in action."""
    print("\n\n=== Transformer Block Demonstration ===")

    # Create a single transformer block
    torch.manual_seed(123)
    block = TransformerBlock(GPT_CONFIG_124M)

    # Create sample input
    batch_size = 2
    seq_len = 4
    x = torch.rand(batch_size, seq_len, GPT_CONFIG_124M["emb_dim"])

    print(f"\nInput shape:  {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {GPT_CONFIG_124M['emb_dim']}")

    # Forward pass
    output = block(x)

    print(f"\nOutput shape: {output.shape}")
    print("\nShape preserved! This allows stacking multiple blocks")

    # Examine intermediate shapes
    print("\n--- Intermediate Tensor Shapes ---")
    print(
        f"After first LayerNorm:  [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )
    print(
        f"After attention:        [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )
    print(
        f"After first shortcut:   [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )
    print(
        f"After second LayerNorm: [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )
    print(
        f"After FFN:              [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )
    print(
        f"After second shortcut:  [{batch_size}, {seq_len}, {GPT_CONFIG_124M['emb_dim']}]"
    )


def analyze_parameter_distribution():
    """Analyze where parameters are distributed in the block."""
    print("\n\n=== Parameter Distribution ===")

    block = TransformerBlock(GPT_CONFIG_124M)

    # Count parameters by component
    components = {
        "norm1": block.norm1,
        "att": block.att,
        "norm2": block.norm2,
        "ff": block.ff,
    }

    total_params = 0
    print(f"{'Component':<20} {'Parameters':>12} {'Percentage':>10}")
    print("-" * 45)

    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name:<20} {params:>12,} {'':<10}")

    print("-" * 45)
    print(f"{'Total':<20} {total_params:>12,}")

    # Calculate percentages
    print("\nPercentage breakdown:")
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        percentage = (params / total_params) * 100
        print(f"{name:<20} {percentage:>6.1f}%")

    print("\nObservations:")
    print("• FFN has the most parameters (~67%)")
    print("• Attention has ~33% of parameters")
    print("• LayerNorm has very few parameters (just scale & shift)")


def explain_pre_norm_vs_post_norm():
    """Explain pre-normalization vs post-normalization designs."""
    print("\n\n=== Pre-Norm vs Post-Norm Design ===")

    print("\n**Post-Norm (Original Transformer):**")
    print("```")
    print("x → Attention → (+) → LayerNorm → FFN → (+) → LayerNorm → output")
    print("    ↑_____________|                  ↑_________|")
    print("```")
    print("• LayerNorm applied after the residual connection")
    print("• Can be unstable for very deep models")
    print("• Gradients can still explode before normalization")

    print("\n**Pre-Norm (Modern GPT):**")
    print("```")
    print("x → LayerNorm → Attention → (+) → LayerNorm → FFN → (+) → output")
    print("    ↑_______________________|       ↑________________|")
    print("```")
    print("• LayerNorm applied before the sub-layers")
    print("• More stable for deep models")
    print("• Better gradient flow")
    print("• Standard in GPT-2, GPT-3, and beyond")

    print("\nWhy Pre-Norm is Better:")
    print("1. Gradients flow through shortcuts untouched")
    print("2. LayerNorm prevents explosion before computation")
    print("3. Enables training of very deep models (100+ layers)")

    print("\n--- Historical Context ---")
    print("\nThe original Transformer paper (2017) used Post-Norm:")
    print("• Seemed natural to normalize after adding residual")
    print("• Worked well for 6-layer models")
    print("• Required careful learning rate warmup")

    print("\nResearchers discovered Pre-Norm advantages (2018-2019):")
    print("• Wang et al. 'Learning Deep Transformer Models for Machine Translation'")
    print("• Showed Pre-Norm enables training without warmup")
    print("• More stable gradients throughout training")

    print("\nWhy the field switched to Pre-Norm:")
    print("1. **Training stability**: Less sensitive to learning rate")
    print("2. **Deeper models**: Post-Norm struggles beyond 12 layers")
    print("3. **Faster convergence**: Pre-Norm often trains faster")
    print("4. **Gradient analysis**: Pre-Norm has better gradient properties")

    print("\nModern consensus (2020+):")
    print("• Pre-Norm is the de facto standard")
    print("• Used in GPT-2, GPT-3, T5, and most modern LLMs")
    print("• Post-Norm mainly of historical interest")
    print("• Some architectures (like BERT) still use Post-Norm variants")

    print("\nNote from Chapter 4:")
    print("'Older architectures, such as the original transformer model,")
    print("applied layer normalization after the self-attention and")
    print("feed forward networks instead, known as Post-LayerNorm,")
    print("which often leads to worse training dynamics.'")


def show_stacking_blocks():
    """Show how transformer blocks stack together."""
    print("\n\n=== Stacking Transformer Blocks ===")

    print("\nGPT-124M uses 12 transformer blocks:")

    print("\n  Token Embeddings")
    print("        ↓")
    print("  Position Embeddings")
    print("        ↓")
    print("     Dropout")
    print("        ↓")
    print(" ┌─ Block 1 ─┐")
    print(" │  LN→Attn  │")
    print(" │     (+)   │")
    print(" │  LN→FFN   │")
    print(" │     (+)   │")
    print(" └───────────┘")
    print("        ↓")
    print(" ┌─ Block 2 ─┐")
    print(" │  LN→Attn  │")
    print(" │     (+)   │")
    print(" │  LN→FFN   │")
    print(" │     (+)   │")
    print(" └───────────┘")
    print("        ↓")
    print("      ...")
    print("        ↓")
    print(" ┌─ Block 12 ─┐")
    print(" │  LN→Attn  │")
    print(" │     (+)   │")
    print(" │  LN→FFN   │")
    print(" │     (+)   │")
    print(" └────────────┘")
    print("        ↓")
    print("   Final LayerNorm")
    print("        ↓")
    print("  Output Projection")

    print("\nEach block refines the representation:")
    print("• Early blocks: Basic patterns (syntax, simple relationships)")
    print("• Middle blocks: Complex patterns (semantics, dependencies)")
    print("• Late blocks: Task-specific patterns (next word prediction)")


def implementation_example():
    """Show a simplified implementation of a transformer block."""
    print("\n\n=== Simplified Implementation ===")

    print("\n```python")
    print("class TransformerBlock(nn.Module):")
    print("    def __init__(self, cfg):")
    print("        super().__init__()")
    print("        self.ln1 = LayerNorm(cfg['emb_dim'])")
    print("        self.attn = MultiHeadAttention(")
    print("            d_in=cfg['emb_dim'],")
    print("            d_out=cfg['emb_dim'],")
    print("            context_length=cfg['context_length'],")
    print("            num_heads=cfg['n_heads'],")
    print("            dropout=cfg['drop_rate'],")
    print("            qkv_bias=cfg['qkv_bias']")
    print("        )")
    print("        self.ln2 = LayerNorm(cfg['emb_dim'])")
    print("        self.ff = FeedForward(cfg)")
    print("        self.drop_resid = nn.Dropout(cfg['drop_rate'])")
    print("")
    print("    def forward(self, x):")
    print("        # First sub-block: Attention with residual")
    print("        shortcut = x")
    print("        x = self.ln1(x)")
    print("        x = self.attn(x)")
    print("        x = self.drop_resid(x)")
    print("        x = x + shortcut  # First residual connection")
    print("")
    print("        # Second sub-block: FFN with residual")
    print("        shortcut = x")
    print("        x = self.ln2(x)")
    print("        x = self.ff(x)")
    print("        x = self.drop_resid(x)")
    print("        x = x + shortcut  # Second residual connection")
    print("")
    print("        return x")
    print("```")


def common_modifications():
    """Discuss common modifications to transformer blocks."""
    print("\n\n=== Common Modifications ===")

    print("\n1. **Parallel Attention and FFN (GPT-J style):**")
    print("   - Run attention and FFN in parallel")
    print("   - Slightly different computation order")
    print("   - Can be more efficient")

    print("\n2. **Gated Linear Units (GLU variants):**")
    print("   - Replace FFN with gated versions")
    print("   - SwiGLU, GeGLU popular in modern models")
    print("   - Often improves performance")

    print("\n3. **RMSNorm instead of LayerNorm:**")
    print("   - Simpler normalization (no mean centering)")
    print("   - Slightly faster")
    print("   - Used in LLaMA models")

    print("\n4. **Mixture of Experts (MoE):**")
    print("   - Multiple FFN experts")
    print("   - Route tokens to different experts")
    print("   - Increases capacity without full compute")

    print("\n5. **Cross-attention layers:**")
    print("   - For encoder-decoder models")
    print("   - Attend to encoder output")
    print("   - Not used in GPT (decoder-only)")


def debugging_tips():
    """Provide debugging tips for transformer blocks."""
    print("\n\n=== Debugging Tips ===")

    print("\n**Common Issues:**")

    print("\n1. Shape mismatches:")
    print("   - Check embedding dimensions match throughout")
    print("   - Ensure attention mask shape is correct")
    print("   - Verify batch dimensions are consistent")

    print("\n2. NaN or Inf values:")
    print("   - Often from exploding gradients")
    print("   - Check learning rate (too high?)")
    print("   - Verify LayerNorm is working")
    print("   - Add gradient clipping")

    print("\n3. No learning:")
    print("   - Check shortcuts are connected properly")
    print("   - Verify dropout isn't too high")
    print("   - Ensure attention mask isn't all zeros")

    print("\n4. Memory issues:")
    print("   - Reduce batch size")
    print("   - Use gradient checkpointing")
    print("   - Consider mixed precision training")

    print("\n**Debugging Tools:**")
    print("• Print shapes at each step")
    print("• Monitor gradient norms")
    print("• Visualize attention weights")
    print("• Check activation statistics")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Transformer Block Assembly")
    print("=" * 60)

    # Basic explanation
    explain_transformer_block()

    # Architecture visualization
    visualize_block_architecture()

    # Component roles
    explain_component_roles()

    # Demonstration
    demonstrate_transformer_block()

    # Parameter analysis
    analyze_parameter_distribution()

    # Pre-norm vs post-norm
    explain_pre_norm_vs_post_norm()

    # Stacking blocks
    show_stacking_blocks()

    # Implementation
    implementation_example()

    # Common modifications
    common_modifications()

    # Debugging tips
    debugging_tips()

    print("\n" + "=" * 60)
    print("Transformer blocks: The LEGO bricks of language models!")
    print("=" * 60)
