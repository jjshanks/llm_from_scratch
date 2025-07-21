# Chapter 4: Building a GPT Model from Scratch

This directory contains the implementation scripts for Chapter 4 of "Build a Large Language Model (From Scratch)" by Manning. This chapter demonstrates how to build a complete GPT (Generative Pre-trained Transformer) model by combining all the concepts from previous chapters.

## Overview

Chapter 4 takes you through building a 124-million parameter GPT-2 style model from the ground up. Each script focuses on a specific component or concept, making it easy to understand how modern language models work.

**Note**: The production-ready implementations of all these components are available in `src/llm_from_scratch/model.py`. The scripts in this directory are educational examples that demonstrate each concept step-by-step.

## Scripts

### Core Component Scripts

1. **`configuration.py`** - GPT model configuration and hyperparameters
   - Defines the 124M parameter model configuration
   - Explains each hyperparameter's role
   - Shows how configuration affects model size and capacity

2. **`tokenization_basics.py`** - Text tokenization and batch processing
   - Converting text to token IDs using tiktoken
   - Creating batches for efficient processing
   - Understanding batch dimensions and shapes

3. **`layer_normalization.py`** - Layer normalization for stable training
   - Why normalization is crucial for deep networks
   - Step-by-step normalization process
   - Comparison with batch normalization

4. **`activation_functions.py`** - GELU vs ReLU activation functions
   - Role of activation functions in neural networks
   - Visual and numerical comparison
   - Why transformers prefer GELU

5. **`feed_forward_network.py`** - Feed forward networks in transformers
   - FFN architecture (Linear → GELU → Linear)
   - 4x dimension expansion explained
   - Position-wise processing

6. **`shortcut_connections.py`** - Residual connections for gradient flow
   - Solving the vanishing gradient problem
   - Comparison of networks with/without shortcuts
   - Why deep networks need residual connections

### Model Assembly Scripts

7. **`transformer_block.py`** - Complete transformer block assembly
   - Combining attention, FFN, LayerNorm, and shortcuts
   - Pre-norm vs post-norm architectures
   - How blocks stack together

8. **`gpt_model_assembly.py`** - Building the complete GPT model
   - Full model architecture from embeddings to output
   - Parameter counting and memory analysis
   - Testing with DummyGPTModel

9. **`text_generation.py`** - Text generation process
   - Autoregressive generation explained
   - Different decoding strategies
   - Why untrained models produce gibberish

## Running the Scripts

Each script is self-contained and can be run independently:

```bash
# Run any script
uv run python book/ch4/configuration.py
uv run python book/ch4/tokenization_basics.py
# ... etc
```

## Recommended Learning Path

1. Start with `configuration.py` to understand model parameters
2. Learn tokenization with `tokenization_basics.py`
3. Study each component:
   - `layer_normalization.py`
   - `activation_functions.py`
   - `feed_forward_network.py`
   - `shortcut_connections.py`
4. See how components combine in `transformer_block.py`
5. Build the complete model with `gpt_model_assembly.py`
6. Understand generation with `text_generation.py`

## Key Concepts Covered

### Model Architecture
- Token and position embeddings
- Multi-head self-attention (from Chapter 3)
- Feed forward networks with GELU activation
- Layer normalization for stability
- Residual connections for gradient flow
- Stacking transformer blocks

### Technical Details
- 124M parameters (163M without weight tying)
- 12 transformer blocks
- 12 attention heads
- 768 embedding dimensions
- 3072 FFN hidden dimensions
- 50,257 vocabulary size (GPT-2 tokenizer)
- 1024 maximum context length

### Implementation Features
- Pre-normalization architecture (modern standard)
- Dropout for regularization
- Proper weight initialization
- Efficient tensor operations
- Clear educational comments

## Connection to Other Chapters

- **Chapter 2**: Uses tokenization concepts
- **Chapter 3**: Integrates multi-head attention
- **Chapter 5**: Will show how to train this model
- **Chapter 6**: Will demonstrate text generation with trained model

## Using the Production Implementation

After understanding the concepts from these educational scripts, you can use the production-ready implementations:

```python
from llm_from_scratch.model import GPTModel, GPTConfig

# Create a 124M parameter model
config = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False
)

model = GPTModel(config)
```

The production implementation in `src/llm_from_scratch/model.py` includes:
- `GPTConfig`: Configuration dataclass for model hyperparameters
- `LayerNorm`: Efficient layer normalization
- `GELU`: Gaussian Error Linear Unit activation
- `FeedForward`: Position-wise feed-forward network
- `TransformerBlock`: Complete transformer layer
- `GPTModel`: Full GPT architecture

## Model Capabilities

The 124M parameter model built here:
- Has similar capacity to GPT-2 small
- Can learn complex language patterns when trained
- Generates text autoregressively
- Suitable for educational purposes and experimentation

## Next Steps

After understanding the model architecture:
1. Chapter 5 will show how to pretrain the model
2. Learn about loss functions and optimization
3. See how the model learns from data
4. Generate meaningful text with trained weights

## Tips for Learning

- Run each script and read the output carefully
- Modify parameters to see their effects
- Use the ASCII visualizations to understand data flow
- Compare components to understand their roles
- Build mental models before diving into math

## Common Questions

**Q: Why 124M parameters?**
A: It's large enough to learn interesting patterns but small enough to train on consumer hardware.

**Q: Why is the untrained output gibberish?**
A: The model has random weights. Training (Chapter 5) teaches it language patterns.

**Q: What's the difference between this and GPT-2?**
A: Very similar architecture. Main differences are in training data and some implementation details.

**Q: Can I modify the architecture?**
A: Yes! Try changing the configuration values to create smaller/larger models.

## Troubleshooting

- **Out of memory**: Reduce batch size or model dimensions
- **Slow execution**: Normal for initial model creation; subsequent runs are faster
- **Import errors**: Ensure you've run `uv pip install -e '.[dev]'`
