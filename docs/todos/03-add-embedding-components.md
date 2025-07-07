# Add Embedding Components

**Priority:** Medium
**Status:** Pending
**Commit SHA:** 360fbdf2661ddbc00ce2487da7e6e9a8d6684253

## Description

Token embedding functionality currently only exists in the chapter examples (`book/ch2/token_embedding.py`). The main package should provide reusable embedding components that are essential for building LLMs, including token embeddings, positional embeddings, and combined embedding layers.

## Justification

- **Core Functionality**: Embeddings are fundamental to LLM architecture
- **Reusability**: Components will be needed for Chapter 3 and beyond
- **Encapsulation**: Proper abstraction of embedding logic
- **Extensibility**: Foundation for more advanced embedding techniques
- **Package Completeness**: Makes the package more self-contained

## Current State

Embedding functionality exists in `book/ch2/token_embedding.py:38-103` but is scattered across example code:

```python
# Token embeddings
vocab_size = 50257  # GPT-2 BPE tokenizer vocabulary size
output_dim = 256    # Embedding dimension
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Positional embeddings
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

# Combined embeddings
input_embeddings = token_embeddings + pos_embeddings
```

## Proposed Changes

1. **Create new module**: Add `src/llm_from_scratch/embedding.py`
2. **Component classes**: Implement `TokenEmbedding`, `PositionalEmbedding`, and `CombinedEmbedding`
3. **Configuration support**: Allow flexible embedding dimensions and vocabulary sizes
4. **PyTorch integration**: Proper `nn.Module` inheritance for seamless integration

## Task List

- [ ] Create `src/llm_from_scratch/embedding.py` module
- [ ] Implement `TokenEmbedding` class:
  - [ ] Inherit from `nn.Module`
  - [ ] Support configurable vocabulary size and embedding dimension
  - [ ] Add proper initialization methods
  - [ ] Include forward pass implementation
- [ ] Implement `PositionalEmbedding` class:
  - [ ] Support both learnable and fixed positional embeddings
  - [ ] Configurable maximum sequence length
  - [ ] Support different positional encoding schemes
- [ ] Implement `CombinedEmbedding` class:
  - [ ] Combine token and positional embeddings
  - [ ] Support different combination strategies (add, concat)
  - [ ] Handle dimension mismatches gracefully
- [ ] Add comprehensive type hints and docstrings
- [ ] Create factory functions for common configurations
- [ ] Update package `__init__.py` to export embedding classes
- [ ] Implement comprehensive tests
- [ ] Add usage examples and documentation

## Design Patterns Applied

- **Composition Pattern**: Combine token and positional embeddings
- **Strategy Pattern**: Support different embedding and combination strategies
- **Factory Pattern**: Create embeddings with common configurations
- **Template Method**: Base embedding class with customizable components

## Class Structure

```python
class TokenEmbedding(nn.Module):
    """Token embedding layer for converting token IDs to dense vectors."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        """Initialize token embedding layer."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""


class PositionalEmbedding(nn.Module):
    """Positional embedding layer for encoding sequence positions."""

    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        learned: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize positional embedding layer."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings for input sequence."""


class CombinedEmbedding(nn.Module):
    """Combined token and positional embedding layer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int,
        dropout: float = 0.0,
        combination: str = "add",  # "add" or "concat"
    ) -> None:
        """Initialize combined embedding layer."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate combined token and positional embeddings."""
```

## Testing Requirements

- [ ] Test `TokenEmbedding` with various vocabulary sizes
- [ ] Test `PositionalEmbedding` with different sequence lengths
- [ ] Test `CombinedEmbedding` with both addition and concatenation
- [ ] Test gradient flow through embedding layers
- [ ] Test with different batch sizes and sequence lengths
- [ ] Test edge cases (empty sequences, maximum length)
- [ ] Performance tests for large vocabularies
- [ ] Integration tests with tokenizers and datasets

## Breaking Changes

**None** - This is a new addition that doesn't affect existing code.

## Implementation Notes

1. Consider supporting both absolute and relative positional encodings
2. Add support for sinusoidal positional encodings (non-learnable)
3. Include proper weight initialization strategies
4. Consider adding embedding dropout for regularization
5. Support for different combination strategies beyond addition

## Factory Functions

```python
def create_gpt_embeddings(
    vocab_size: int,
    embed_dim: int,
    max_len: int,
    dropout: float = 0.0,
) -> CombinedEmbedding:
    """Create GPT-style embeddings with standard configuration."""

def create_bert_embeddings(
    vocab_size: int,
    embed_dim: int,
    max_len: int,
    dropout: float = 0.0,
) -> CombinedEmbedding:
    """Create BERT-style embeddings with standard configuration."""
```

## Success Criteria

- [ ] All embedding classes properly inherit from `nn.Module`
- [ ] Comprehensive test coverage for all components
- [ ] Clear documentation and usage examples
- [ ] Seamless integration with existing tokenizer and dataset classes
- [ ] Performance comparable to raw PyTorch implementations
- [ ] Support for common LLM embedding configurations
- [ ] Chapter examples updated to use package components
