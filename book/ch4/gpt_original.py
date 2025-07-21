"""
Chapter 4 - Implementing a GPT Model from Scratch

This script demonstrates building a complete GPT (Generative Pre-trained Transformer)
architecture from the ground up, combining all the concepts from previous chapters.

Key concepts demonstrated:
- GPT model configuration and architecture
- Layer normalization for stable training
- GELU activation functions (smoother than ReLU)
- Feed forward networks within transformer blocks
- Shortcut (residual) connections to prevent vanishing gradients
- Complete transformer blocks combining attention and feed forward layers
- Full GPT model assembly with embedding and output layers
- Text generation process (converting model outputs to text)

The script builds towards a 124-million parameter GPT-2 model, showing:
1. How individual components work in isolation
2. How they combine into transformer blocks
3. How transformer blocks stack to form the complete model
4. How the model generates text token by token

Usage: uv run python book/ch4/gpt.py
"""

from llm_from_scratch.model import (
    GELU,
    DummyGPTModel,
    ExampleDeepNeuralNetwork,
    FeedForward,
    GPTModel,
    LayerNorm,
    TransformerBlock,
)
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt

# Turn off scientific notation for cleaner output
torch.set_printoptions(sci_mode=False)

# === Section 4.1: GPT Model Configuration ===
# This configuration defines a 124-million parameter GPT-2 model
# These settings determine the model's capacity and computational requirements

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

# === Section 4.1.1: Tokenizing Input Text ===
# Before feeding text to the model, we must convert it to numbers (token IDs)
# The tiktoken tokenizer uses byte-pair encoding (BPE) from Chapter 2

tokenizer = tiktoken.get_encoding("gpt2")

# Create a batch with two example sentences
# A "batch" is a group of inputs processed together (like a batch of cookies in an oven)
# Processing multiple texts at once is more efficient than one at a time
print("\n=== What is a Batch? ===")
print("A batch groups multiple inputs for efficient parallel processing")
print("Like baking 12 cookies at once instead of 1 at a time")
print("\nOur batch will contain 2 text samples:")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
print(f"Text 1: '{txt1}'")
print(f"Text 2: '{txt2}'")

# Encode each text into token IDs
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

# Stack into a 2D tensor where each row is one text sample
batch = torch.stack(batch, dim=0)

print("\n=== Tokenization Process ===")
print("Before feeding text to the model, we convert words to numbers (token IDs)")
print(f"'{txt1}' → {tokenizer.encode(txt1)}")
print(f"'{txt2}' → {tokenizer.encode(txt2)}")
print("\nNow we stack these into a 2D tensor (like a spreadsheet):")
print("- Each row = one text sample")
print("- Each column = token position (1st word, 2nd word, etc.)")
print("\nTokenized batch:")
print(batch)
print(f"\nBatch shape: {batch.shape}")
print("This means: 2 texts, with 4 tokens each")
print("The model will process both texts simultaneously!")

# === Section 4.1.2: Testing the Placeholder GPT Model ===
# The DummyGPTModel uses placeholder components to show the overall structure
# This helps understand data flow before implementing the real components

print("\n" + "=" * 60)
print("=== Testing DummyGPTModel (placeholder architecture) ===")

torch.manual_seed(123)  # For reproducible random initialization
model = DummyGPTModel(GPT_CONFIG_124M)

# Forward pass: convert token IDs to output predictions
logits = model(batch)

print("\nOutput shape:", logits.shape)
print("Interpretation: [batch_size=2, sequence_length=4, vocab_size=50257]")
print("\nWhere does 50,257 come from?")
print("- It's the GPT-2 tokenizer vocabulary size (see line 37: 'vocab_size': 50257)")
print("- The tokenizer knows 50,257 different tokens (words/subwords/characters)")
print("- So the model must predict which of these 50,257 tokens comes next")
print("\nEach of the 4 tokens produces 50,257 scores (one per vocabulary word)")
print("But wait - are these meaningful predictions? NO!")
print("The model has RANDOM weights (not trained yet), so these are garbage values")
print("\nRaw logits (first few values):")
print(logits[:, :, :5])  # Show first 5 vocab predictions only
print("\nThese random numbers will become meaningful predictions after training")

# === Section 4.2: Layer Normalization ===
# Layer normalization helps stabilize training in deep networks
# It normalizes activations to have mean=0 and variance=1

print("\n" + "=" * 60)
print("=== Understanding Layer Normalization ===")
print("\nWhat is Layer Normalization?")
print("- A technique that standardizes the inputs to each layer")
print("- For each sample, it makes the activations have mean=0 and variance=1")
print("- Like adjusting grades on a curve - everyone's scores get redistributed")
print("\nWhy normalize? Deep networks can suffer from:")
print("- Vanishing gradients: When values get very small (0.001 × 0.001 × ... → 0)")
print("- Exploding gradients: When values get very large (100 × 100 × ... → ∞)")
print("\nHow does normalization help?")
print("- Keeps values centered around 0 with variance 1")
print("- Prevents values from drifting too high or too low")
print("- Like a thermostat keeping temperature stable")

# Create example: a simple neural network layer
torch.manual_seed(123)
batch_example = torch.randn(2, 5)  # 2 samples, 5 features each

print("\nFirst, let's understand ReLU:")
print("- ReLU = Rectified Linear Unit")
print("- Simple rule: if input < 0, output = 0; otherwise output = input")
print("- Like a gate that only lets positive signals through")

layer = nn.Sequential(
    nn.Linear(5, 6),  # Transform 5 inputs to 6 outputs
    nn.ReLU(),  # ReLU activation (sets negative values to 0)
)
out = layer(batch_example)

print("\n--- Example: Why We Need Normalization ---")
print("\nStep 1: Layer output (after Linear + ReLU):")
print(out)
print("\nNotice: ReLU set all negative values to 0.0000")
print("The remaining values range from 0.2133 to 0.5198")

# Calculate statistics for each sample in the batch
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("\nStep 2: Check the current statistics")
print(f"Sample 1 - Mean: {mean[0].item():.4f}, Variance: {var[0].item():.4f}")
print(f"Sample 2 - Mean: {mean[1].item():.4f}, Variance: {var[1].item():.4f}")
print("\nWhat is variance?")
print("- Variance measures how spread out numbers are")
print("- Low variance (0.02) = numbers are close together")
print("- High variance (1.0) = numbers are spread apart")
print("- Like measuring if students scored 88-92 (low) or 50-100 (high)")
print("\nProblem: These values are all over the place!")
print("We want mean=0 and variance=1 for stable training")

# Apply normalization: (x - mean) / standard_deviation
print("\nStep 3: Apply normalization formula: (x - mean) / sqrt(variance)")
out_norm = (out - mean) / torch.sqrt(var)

print("\nNormalized values:")
print("Sample 1:", [f"{x:.3f}" for x in out_norm[0].tolist()])
print("Sample 2:", [f"{x:.3f}" for x in out_norm[1].tolist()])

# Verify the normalization worked
mean_norm = out_norm.mean(dim=-1, keepdim=True)
var_norm = out_norm.var(dim=-1, keepdim=True)

print("\nStep 4: Verify normalization worked")
print(f"Sample 1 - Mean: {mean_norm[0].item():.1e}, Variance: {var_norm[0].item():.4f}")
print(f"Sample 2 - Mean: {mean_norm[1].item():.1e}, Variance: {var_norm[1].item():.4f}")
print("\n✓ Success! Mean ≈ 0 and variance ≈ 1")
print("(The tiny mean values like 9.9e-09 are just computer rounding errors)")

# Removed redundant scientific notation section - already disabled at the top

# === Using the LayerNorm Module ===
# The LayerNorm class encapsulates normalization + learnable parameters
# It adds scale and shift parameters that the model can adjust during training

print("\n=== Using LayerNorm Module ===")
ln = LayerNorm(emb_dim=5)  # 5 = number of features to normalize
out_ln = ln(batch_example)

# Verify normalization
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("\nAfter LayerNorm module:")
print("Mean:\n", mean)
print("Variance:\n", var)
print("\nLayerNorm includes learnable scale/shift parameters")
print("This allows the model to adjust normalization if needed")

# === Section 4.3: GELU Activation Function ===
# GELU (Gaussian Error Linear Unit) is a smooth activation function
# Unlike ReLU's sharp cutoff at 0, GELU has a smooth transition

print("\n" + "=" * 60)
print("=== GELU vs ReLU Activation Functions ===")
print("\nActivation functions add non-linearity to neural networks")
print("Without them, stacking layers would just create one big linear transformation")

# Create both activation functions
gelu, relu = GELU(), nn.ReLU()

# Generate points to plot
x = torch.linspace(-3, 3, 100)  # 100 points from -3 to 3
y_gelu, y_relu = gelu(x), relu(x)

# Create comparison plots
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

print("\nKey differences:")
print("- ReLU: Sharp cutoff at 0 (if x<0, output=0)")
print("- GELU: Smooth curve, allows small negative outputs")
print("- GELU's smoothness can improve gradient flow during training")

# Uncomment to display plots:
# plt.tight_layout()
# plt.show()

# === Section 4.3.1: Feed Forward Network ===
# Each transformer block contains a feed forward network (FFN)
# The FFN processes each token independently (no cross-token interaction)

print("\n" + "=" * 60)
print("=== Feed Forward Network in Transformers ===")
print("\nStructure: Linear → GELU → Linear")
print("Expands dimension by 4x internally (768 → 3072 → 768)")
print("This expansion allows learning complex transformations")
print("\n--- What Does the FFN Actually Do? ---")
print("\nThink of it like a 'thinking space' for each token:")
print("1. First Linear (768 → 3072): Projects to higher dimension")
print("   - Like spreading data across more neurons to analyze")
print("   - Each of 3072 neurons looks for different patterns")
print("   - More dimensions = more capacity to detect features")
print("\n2. GELU Activation: Adds non-linearity")
print("   - Without this, two Linear layers = one Linear layer!")
print("   - Allows learning complex, non-linear patterns")
print("   - Smooth activation helps with gradient flow")
print("\n3. Second Linear (3072 → 768): Projects back down")
print("   - Combines the 3072 features into final representation")
print("   - Forces model to compress/summarize what it learned")
print("\n--- Mathematical View ---")
print("Input x (dim=768) goes through:")
print("  h = GELU(x @ W1 + b1)     # W1 is [768, 3072]")
print("  output = h @ W2 + b2      # W2 is [3072, 768]")
print("\n--- How Does Linear Layer Transform x → h? ---")
print("\nIt's NOT dividing or copying! Each h is a weighted combination:")
print("\nFor input x = [x₁, x₂, ..., x₇₆₈], each output neuron h_i:")
print("  h₁ = w₁,₁×x₁ + w₂,₁×x₂ + ... + w₇₆₈,₁×x₇₆₈ + b₁")
print("  h₂ = w₁,₂×x₁ + w₂,₂×x₂ + ... + w₇₆₈,₂×x₇₆₈ + b₂")
print("  ...")
print("  h₃₀₇₂ = w₁,₃₀₇₂×x₁ + w₂,₃₀₇₂×x₂ + ... + w₇₆₈,₃₀₇₂×x₇₆₈ + b₃₀₇₂")
print("\nEach h_i uses DIFFERENT weights - it's looking for different patterns!")
print("\n--- Concrete Example with Small Numbers ---")
print("Say x = [1, 2, 3] and we expand to 4 neurons:")
print("  h₁ = 0.1×1 + 0.5×2 + (-0.2)×3 = 0.5   (detects pattern A)")
print("  h₂ = -0.3×1 + 0.2×2 + 0.4×3 = 1.3    (detects pattern B)")
print("  h₃ = 0.8×1 + (-0.1)×2 + 0.1×3 = 0.9  (detects pattern C)")
print("  h₄ = 0.0×1 + 0.7×2 + 0.3×3 = 2.3     (detects pattern D)")
print("\nNotice: Each output is a UNIQUE combination of ALL inputs!")
print("The weights determine what patterns each neuron detects.")
print("\n--- What Makes Each Neuron Different? ---")
print("The WEIGHTS! Each of the 3072 neurons has its own set of 768 weights.")
print("That's 768 × 3072 = 2,359,296 trainable parameters in W1 alone!")
print("\nDuring training, each neuron learns to detect different features:")
print("• Neuron 1 might learn weights that activate for 'animal-like' patterns")
print("• Neuron 2 might learn weights that activate for 'verb-like' patterns")
print("• Neuron 3 might learn weights that activate for 'past-tense' patterns")
print("• ... and so on for all 3072 neurons")
print("\nIt's like having 3072 different 'detectors', each looking for")
print("something different in the 768-dimensional input vector!")
print("\n--- Why This Design? ---")
print("• Bottleneck architecture: expand then compress")
print("• Expansion gives model room to compute complex features")
print("• Compression forces it to extract only what's important")
print("• Applied to EACH token position independently")
print("\n--- Real World Analogy ---")
print("Like analyzing an image:")
print("1. Expand: Extract many features (edges, colors, textures...)")
print("2. Process: Identify patterns in those features")
print("3. Compress: Summarize into useful representation")
print("\nThe FFN does this for each token's representation!")
print("\n--- ASCII Visualization of FFN ---")
print("Token Input (768 dims)")
print("    │")
print("    │  [x₁, x₂, ..., x₇₆₈]")
print("    ↓")
print("Linear Layer (W1 @ x + b1)")
print("    │")
print("    │  Each h_i = weighted sum of ALL x values:")
print("    │  h₁ = Σ(w₁ᵢ × xᵢ) + b₁  ←─ unique weights")
print("    │  h₂ = Σ(w₂ᵢ × xᵢ) + b₂  ←─ different weights")
print("    │  h₃ = Σ(w₃ᵢ × xᵢ) + b₃  ←─ different weights")
print("    │  ...")
print("    │  h₃₀₇₂ = Σ(w₃₀₇₂ᵢ × xᵢ) + b₃₀₇₂")
print("    ↓")
print("GELU Activation")
print("    │")
print("    │  [GELU(h₁), GELU(h₂), ..., GELU(h₃₀₇₂)]")
print("    ↓")
print("Linear Layer (W2 @ h + b2)")
print("    │")
print("    │  [y₁, y₂, ..., y₇₆₈]")
print("    ↓")
print("Token Output (768 dims)")
print("\nKey insight: Even though input and output have same size,")
print("the intermediate expansion allows complex transformations!")

# Create FFN module
ffn = FeedForward(GPT_CONFIG_124M)

# Test with random input: [batch_size=2, seq_len=3, embed_dim=768]
x = torch.rand(2, 3, 768)
out = ffn(x)

print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print("\nNote: Input and output shapes match!")
print("This allows stacking many transformer blocks")
print("\n--- Example: What Patterns Might FFN Learn? ---")
print("\nImagine the token 'cat' with its 768-dim representation:")
print("• Some of the 3072 neurons might activate for:")
print("  - 'animal' patterns")
print("  - 'pet' patterns")
print("  - 'mammal' patterns")
print("  - 'noun' patterns")
print("  - 'living thing' patterns")
print("  - And many more subtle features...")
print("\nThe second Linear layer then combines these to create")
print("a refined representation that captures the essence of 'cat'")
print("in the context of the current sentence.")
print("\nIMPORTANT: FFN operates on EACH token independently!")
print("It doesn't look at other tokens - that's the attention's job.")

# === Section 4.4: Shortcut (Residual) Connections ===
# Shortcut connections help gradients flow through deep networks
# They add the input of a layer to its output: output = f(x) + x

print("\n" + "=" * 60)
print("=== Understanding Shortcut Connections ===")
print("\nProblem: In deep networks, gradients can vanish")
print("Solution: Shortcuts provide direct paths for gradients")
print("\n" + "--- ASCII Diagram: Regular Deep Network ---")
print("Input → Layer1 → Layer2 → Layer3 → Layer4 → Output")
print("         ↓         ↓         ↓         ↓")
print("      (grad₄)   (grad₃)   (grad₂)   (grad₁)")
print("\nGradient flow: grad₄ × grad₃ × grad₂ × grad₁")
print("If each gradient < 1, product → 0 (vanishing!)")
print("\n" + "--- ASCII Diagram: Network with Shortcut Connections ---")
print("Input ─┬→ Layer1 ─┬→ Layer2 ─┬→ Layer3 ─┬→ Layer4 → Output")
print("       │     ↓     │     ↓     │     ↓     │     ↓")
print("       │    (+)    │    (+)    │    (+)    │    (+)")
print("       └─────┴─────┴─────┴─────┴─────┴─────┴─────┘")
print("         (shortcut connections bypass layers)")
print("\nGradient flow: Direct paths prevent vanishing!")
print("Even if layer gradients are small, shortcuts carry signal")
print("\n" + "--- ASCII Diagram: Shortcut Connection Math ---")
print("     Input (x)")
print("        │")
print("        ├─────────┐")
print("        │         │")
print("        ↓         │")
print("    Layer(x)      │ (identity)")
print("        │         │")
print("        ↓         │")
print("       (+)←───────┘")
print("        │")
print("        ↓")
print("  Output = Layer(x) + x")
print("\nThis '+' operation is the key!")
print("It ensures gradients have multiple paths to flow back")

# Create a 5-layer network to demonstrate gradient flow
layer_sizes = [3, 3, 3, 3, 3, 1]  # 5 layers: 3→3→3→3→3→1
sample_input = torch.tensor([[1.0, 0.0, -1.0]])

# First, without shortcuts
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


def print_gradients(model, x):
    """Helper function to visualize gradient magnitudes in each layer."""
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.0]])  # Dummy target for loss calculation

    # Calculate loss (how wrong the prediction is)
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass: compute gradients
    loss.backward()

    # Print gradient magnitudes for each layer
    for name, param in model.named_parameters():
        if "weight" in name:
            # Mean absolute gradient shows how much the layer is learning
            print(f"{name} has gradient mean of {param.grad.abs().mean().item():.6f}")


print("\nGradients WITHOUT shortcut connections:")
print_gradients(model_without_shortcut, sample_input)
print("\nNotice: Gradients get smaller in earlier layers (vanishing gradient!)")

# Now with shortcuts
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print("\nGradients WITH shortcut connections:")
print_gradients(model_with_shortcut, sample_input)
print("\nNotice: Gradients remain stable across all layers!")
print("This enables training much deeper networks (like 12+ layer GPT)")
print("\n" + "--- ASCII Diagram: Shortcut in Transformer Block ---")
print("     Input")
print("       │")
print("       ├─────────────┐")
print("       ↓             │")
print("  LayerNorm1         │")
print("       ↓             │")
print(" Multi-Head          │")
print(" Attention           │")
print("       ↓             │")
print("    Dropout          │")
print("       ↓             │")
print("      (+)←───────────┘ (1st shortcut)")
print("       │")
print("       ├─────────────┐")
print("       ↓             │")
print("  LayerNorm2         │")
print("       ↓             │")
print("  FeedForward        │")
print("   Network           │")
print("       ↓             │")
print("    Dropout          │")
print("       ↓             │")
print("      (+)←───────────┘ (2nd shortcut)")
print("       │")
print("       ↓")
print("    Output")
print("\nEach transformer block has TWO shortcut connections!")
print("This double-shortcut pattern is key to training deep transformers")

# === Section 4.5: Transformer Block ===
# The transformer block is the core building block of GPT
# It combines multi-head attention with feed forward networks

print("\n" + "=" * 60)
print("=== Transformer Block Architecture ===")
print("\nEach transformer block contains:")
print("1. Multi-head attention (from Chapter 3)")
print("2. Feed forward network")
print("3. Two layer normalizations")
print("4. Two shortcut connections")
print("\nGPT stacks 12 of these blocks for the 124M model")

# Create a single transformer block
torch.manual_seed(123)
x = torch.rand(2, 4, 768)  # [batch=2, seq_len=4, embed_dim=768]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print("\nShape preserved! This allows stacking multiple blocks")

# === Section 4.6: Complete GPT Model ===
# Now we assemble all components into the full GPT architecture

print("\n" + "=" * 60)
print("=== Building the Complete GPT Model ===")
print("\nGPT Architecture:")
print("1. Token embeddings (convert IDs to vectors)")
print("2. Position embeddings (encode token positions)")
print("3. Dropout (regularization)")
print("4. 12x Transformer blocks")
print("5. Final layer norm")
print("6. Output projection (to vocabulary size)")

# Initialize the complete model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

# Forward pass with our tokenized batch
out = model(batch)
print("\nInput batch (token IDs):\n", batch)
print("\nOutput shape:", out.shape)
print("Interpretation: [batch=2, seq_len=4, vocab_size=50257]")
print("\nEach position outputs 50,257 logits (unnormalized probabilities)")
print("The highest logit indicates the predicted next token")

# === Section 4.6.1: Model Size Analysis ===
# Let's understand how many parameters (learnable weights) are in our model

print("\n" + "=" * 60)
print("=== Analyzing Model Size ===")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_params:,}")
print("\nWait, that's 163M, not 124M! Why?")

# The discrepancy comes from the embedding and output layers
print("\nLarge layers in the model:")
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
print("\nBoth are [50257, 768] = 38,597,376 parameters each!")
print("These layers map between tokens and embeddings")

# GPT-2 uses "weight tying": sharing weights between embedding and output
# Our implementation uses separate weights (often gives better results)
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"\nWith weight tying (like original GPT-2): {total_params_gpt2:,}")
print("This matches the advertised 124M parameters!")
print("\nWe keep them separate for better performance")

# Calculate memory requirements
# Each parameter is a 32-bit float (4 bytes)
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)

print("\nMemory footprint:")
print(f"Total size: {total_size_mb:.2f} MB")
print("\nThis is just for storing parameters!")
print("Training requires ~10-20x more memory for:")
print("- Gradients (same size as parameters)")
print("- Optimizer states (2x parameters for Adam)")
print("- Activations (for backpropagation)")

# === Section 4.7: Text Generation ===
# Now let's see how GPT generates text one token at a time

print("\n" + "=" * 60)
print("=== Text Generation Process ===")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text by repeatedly predicting the next token.

    Args:
        model: The GPT model
        idx: Starting token IDs [batch_size, sequence_length]
        max_new_tokens: How many new tokens to generate
        context_size: Maximum context length (truncate if needed)
    """
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions (no gradient needed for inference)
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last token's predictions
        logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Convert logits to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Pick the most likely next token (greedy decoding)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# === Section 4.7.1: Generating Text from a Prompt ===

print("\nStarting with a prompt:")
start_context = "Hello, I am"

# Convert text to token IDs
encoded = tokenizer.encode(start_context)
print(f"Text: '{start_context}'")
print(f"Token IDs: {encoded}")

# Convert to tensor and add batch dimension
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # [1, num_tokens]
print(f"Tensor shape: {encoded_tensor.shape}")

# Put model in evaluation mode (disables dropout)
model.eval()

print("\nGenerating 6 new tokens...")
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

print(f"\nGenerated token IDs: {out}")
print(f"Total length: {len(out[0])} tokens (4 original + 6 new)")

# Decode back to text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"\nGenerated text: '{decoded_text}'")

print("\n" + "=" * 60)
print("=== Why is the output gibberish? ===")
print("\nThe model has random weights - it hasn't been trained yet!")
print("Training is covered in Chapter 5, where we'll teach the model to:")
print("- Predict likely next tokens based on context")
print("- Generate coherent, meaningful text")
print("- Learn grammar, facts, and reasoning from training data")
print("\nFor now, we've successfully built the architecture!")
"""
expected output:

Tokenized batch (each number is a token ID):
tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])
Batch shape: torch.Size([2, 4]) (2 texts, 4 tokens each)

============================================================
=== Testing DummyGPTModel (placeholder architecture) ===

Output shape: torch.Size([2, 4, 50257])
Interpretation: [batch_size=2, sequence_length=4, vocab_size=50257]
Each of the 4 tokens produces 50,257 scores (one per vocabulary word)

Raw logits (first few values):
tensor([[[-1.2034,  0.3201, -0.7130,  0.8838, -0.2369],
         [-0.1192,  0.4539, -0.4432, -0.3348,  0.3880],
         [ 0.5307,  1.6720, -0.4695, -0.5703,  0.1557],
         [ 0.0139,  1.6755, -0.3388, -1.4925,  1.6891]],

        [[-1.0908,  0.1798, -0.9484,  0.4386, -0.3876],
         [-0.7860,  0.5581, -0.0610, -0.4965, -0.3014],
         [ 0.3567,  1.2698, -0.6398,  0.4755, -0.1853],
         [-0.2407, -0.7349, -0.5102, -0.7309,  1.2275]]],
       grad_fn=<SliceBackward0>)

============================================================
=== Understanding Layer Normalization ===

Why normalize? Deep networks can suffer from:
- Vanishing gradients (signals get too small)
- Exploding gradients (signals get too large)
Layer norm keeps values in a stable range.

Layer output before normalization:
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
Note: ReLU removed all negative values

Statistics before normalization:
Mean per sample:
 tensor([[0.1324],
        [0.2170]], grad_fn=<MeanBackward1>)
Variance per sample:
 tensor([[0.0231],
        [0.0398]], grad_fn=<VarBackward0>)

Note: Values are not standardized (meanΓëá0, varianceΓëá1)

Normalized layer outputs:
tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
       grad_fn=<DivBackward0>)

Statistics after normalization:
Mean per sample:
 tensor([[9.9341e-09],
        [1.9868e-08]], grad_fn=<MeanBackward1>)
Variance per sample:
 tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)

Success! Mean Γëê 0 and variance Γëê 1
(Small numerical errors like -5.96e-08 are normal)

With scientific notation disabled:
Mean:
 tensor([[    0.0000],
        [    0.0000]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)

=== Using LayerNorm Module ===

After LayerNorm module:
Mean:
 tensor([[    -0.0000],
        [     0.0000]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)

LayerNorm includes learnable scale/shift parameters
This allows the model to adjust normalization if needed

============================================================
=== GELU vs ReLU Activation Functions ===

Activation functions add non-linearity to neural networks
Without them, stacking layers would just create one big linear transformation

Key differences:
- ReLU: Sharp cutoff at 0 (if x<0, output=0)
- GELU: Smooth curve, allows small negative outputs
- GELU's smoothness can improve gradient flow during training

============================================================
=== Feed Forward Network in Transformers ===

Structure: Linear ΓåÆ GELU ΓåÆ Linear
Expands dimension by 4x internally (768 ΓåÆ 3072 ΓåÆ 768)
This expansion allows learning complex transformations

Input shape:  torch.Size([2, 3, 768])
Output shape: torch.Size([2, 3, 768])

Note: Input and output shapes match!
This allows stacking many transformer blocks

============================================================
=== Understanding Shortcut Connections ===

Problem: In deep networks, gradients can vanish
Solution: Shortcuts provide direct paths for gradients

Gradients WITHOUT shortcut connections:
layers.0.0.weight has gradient mean of 0.000202
layers.1.0.weight has gradient mean of 0.000120
layers.2.0.weight has gradient mean of 0.000715
layers.3.0.weight has gradient mean of 0.001399
layers.4.0.weight has gradient mean of 0.005050

Notice: Gradients get smaller in earlier layers (vanishing gradient!)

Gradients WITH shortcut connections:
layers.0.0.weight has gradient mean of 0.221698
layers.1.0.weight has gradient mean of 0.206941
layers.2.0.weight has gradient mean of 0.328970
layers.3.0.weight has gradient mean of 0.266573
layers.4.0.weight has gradient mean of 1.325854

Notice: Gradients remain stable across all layers!
This enables training much deeper networks (like 12+ layer GPT)

============================================================
=== Transformer Block Architecture ===

Each transformer block contains:
1. Multi-head attention (from Chapter 3)
2. Feed forward network
3. Two layer normalizations
4. Two shortcut connections

GPT stacks 12 of these blocks for the 124M model

Input shape:  torch.Size([2, 4, 768])
Output shape: torch.Size([2, 4, 768])

Shape preserved! This allows stacking multiple blocks

============================================================
=== Building the Complete GPT Model ===

GPT Architecture:
1. Token embeddings (convert IDs to vectors)
2. Position embeddings (encode token positions)
3. Dropout (regularization)
4. 12x Transformer blocks
5. Final layer norm
6. Output projection (to vocabulary size)

Input batch (token IDs):
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

Output shape: torch.Size([2, 4, 50257])
Interpretation: [batch=2, seq_len=4, vocab_size=50257]

Each position outputs 50,257 logits (unnormalized probabilities)
The highest logit indicates the predicted next token

============================================================
=== Analyzing Model Size ===

Total number of parameters: 163,009,536

Wait, that's 163M, not 124M! Why?

Large layers in the model:
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])

Both are [50257, 768] = 38,597,376 parameters each!
These layers map between tokens and embeddings

With weight tying (like original GPT-2): 124,412,160
This matches the advertised 124M parameters!

We keep them separate for better performance

Memory footprint:
Total size: 621.83 MB

This is just for storing parameters!
Training requires ~10-20x more memory for:
- Gradients (same size as parameters)
- Optimizer states (2x parameters for Adam)
- Activations (for backpropagation)

============================================================
=== Text Generation Process ===

Starting with a prompt:
Text: 'Hello, I am'
Token IDs: [15496, 11, 314, 716]
Tensor shape: torch.Size([1, 4])

Generating 6 new tokens...

Generated token IDs: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
Total length: 10 tokens (4 original + 6 new)

Generated text: 'Hello, I am Featureiman Byeswickattribute argue'

============================================================
=== Why is the output gibberish? ===

The model has random weights - it hasn't been trained yet!
Training is covered in Chapter 5, where we'll teach the model to:
- Predict likely next tokens based on context
- Generate coherent, meaningful text
- Learn grammar, facts, and reasoning from training data

For now, we've successfully built the architecture!
"""
