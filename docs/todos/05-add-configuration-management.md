# Add Configuration Management

**Priority:** Low
**Status:** Pending
**Commit SHA:** 360fbdf2661ddbc00ce2487da7e6e9a8d6684253

## Description

Currently, configuration parameters are hardcoded throughout the codebase and chapter examples. The package should provide a centralized configuration management system using dataclasses to improve parameter management, reproducibility, and maintainability.

## Justification

- **Parameter Management**: Centralize configuration in one place
- **Reproducibility**: Enable saving and loading of configurations
- **Maintainability**: Reduce parameter duplication across files
- **Validation**: Ensure parameter values are valid
- **Documentation**: Self-documenting configuration with type hints
- **Flexibility**: Easy to extend with new parameters

## Current State

Parameters are scattered across files:

```python
# In book/ch2/dataloader.py
batch_size=4
max_length=256
stride=128

# In book/ch2/token_embedding.py
vocab_size = 50257
output_dim = 256
context_length = max_length

# In various examples
shuffle=True
drop_last=True
num_workers=0
```

## Proposed Changes

1. **Create configuration module**: Add `src/llm_from_scratch/config.py`
2. **Configuration dataclasses**: Define structured configuration classes
3. **Validation**: Add parameter validation and constraints
4. **Serialization**: Support for saving/loading configurations
5. **Factory methods**: Create common configurations easily

## Task List

- [ ] Create `src/llm_from_scratch/config.py` module
- [ ] Implement configuration dataclasses:
  - [ ] `TokenizerConfig` for tokenizer parameters
  - [ ] `DatasetConfig` for dataset and dataloader parameters
  - [ ] `EmbeddingConfig` for embedding layer parameters
  - [ ] `TrainingConfig` for training-related parameters
- [ ] Add validation methods for each configuration class
- [ ] Implement serialization methods (to/from JSON, YAML)
- [ ] Create factory methods for common configurations
- [ ] Add comprehensive type hints and documentation
- [ ] Update existing classes to accept configuration objects
- [ ] Create configuration validation tests
- [ ] Add example configuration files
- [ ] Update documentation with configuration examples

## Design Patterns Applied

- **Builder Pattern**: For complex configuration construction
- **Factory Pattern**: For creating common configurations
- **Strategy Pattern**: Different configuration strategies for different use cases
- **Validation Pattern**: Ensure configuration correctness

## Configuration Structure

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer settings."""

    tokenizer_type: str = "simple"  # "simple" or "bpe"
    encoding_name: str = "gpt2"  # For BPE tokenizers
    vocab_size: Optional[int] = None
    unknown_token: str = "<|unk|>"
    end_of_text_token: str = "<|endoftext|>"
    allowed_special: Optional[set] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tokenizer_type not in ["simple", "bpe"]:
            raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}")

    @classmethod
    def gpt2_config(cls) -> "TokenizerConfig":
        """Create GPT-2 style tokenizer configuration."""
        return cls(
            tokenizer_type="bpe",
            encoding_name="gpt2",
            vocab_size=50257,
            allowed_special={"<|endoftext|>"}
        )

@dataclass
class DatasetConfig:
    """Configuration for dataset and dataloader settings."""

    max_length: int = 256
    stride: int = 128
    batch_size: int = 4
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def gpt2_config(cls) -> "DatasetConfig":
        """Create GPT-2 style dataset configuration."""
        return cls(
            max_length=1024,
            stride=512,
            batch_size=8,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )

@dataclass
class EmbeddingConfig:
    """Configuration for embedding layers."""

    vocab_size: int = 50257
    embed_dim: int = 768
    max_len: int = 1024
    dropout: float = 0.1
    learned_positional: bool = True
    combination: str = "add"  # "add" or "concat"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.combination not in ["add", "concat"]:
            raise ValueError("combination must be 'add' or 'concat'")

@dataclass
class LLMConfig:
    """Complete LLM configuration combining all components."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LLMConfig":
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "tokenizer": self.tokenizer.__dict__,
            "dataset": self.dataset.__dict__,
            "embedding": self.embedding.__dict__
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create configuration from dictionary."""
        return cls(
            tokenizer=TokenizerConfig(**data["tokenizer"]),
            dataset=DatasetConfig(**data["dataset"]),
            embedding=EmbeddingConfig(**data["embedding"])
        )

    @classmethod
    def gpt2_config(cls) -> "LLMConfig":
        """Create GPT-2 style configuration."""
        return cls(
            tokenizer=TokenizerConfig.gpt2_config(),
            dataset=DatasetConfig.gpt2_config(),
            embedding=EmbeddingConfig(
                vocab_size=50257,
                embed_dim=768,
                max_len=1024,
                dropout=0.1
            )
        )
```

## Testing Requirements

- [ ] Test configuration validation and error handling
- [ ] Test serialization/deserialization (JSON, YAML)
- [ ] Test factory methods for common configurations
- [ ] Test configuration modification and validation
- [ ] Test integration with existing classes
- [ ] Test edge cases and invalid configurations
- [ ] Test performance impact of configuration usage

## Breaking Changes

**None** - This is a new addition that doesn't affect existing code. Existing classes will continue to work with individual parameters while also accepting configuration objects.

## Implementation Notes

1. Use dataclasses for clean, type-safe configuration
2. Add validation in `__post_init__` methods
3. Support both individual parameters and configuration objects
4. Consider using Pydantic for more advanced validation
5. Add support for environment variable overrides
6. Consider configuration inheritance for variations

## Usage Examples

```python
# Create configuration programmatically
config = LLMConfig.gpt2_config()

# Modify specific parameters
config.dataset.batch_size = 16
config.embedding.dropout = 0.2

# Save and load configuration
config.save("my_config.json")
loaded_config = LLMConfig.load("my_config.json")

# Use with existing classes
tokenizer = create_tokenizer(config.tokenizer)
dataset = GPTDatasetV1(text, tokenizer, config.dataset)
embeddings = CombinedEmbedding(config.embedding)
```

## Success Criteria

- [ ] Clean, type-safe configuration management
- [ ] Comprehensive validation and error handling
- [ ] Serialization support for reproducibility
- [ ] Integration with existing package components
- [ ] Factory methods for common configurations
- [ ] Clear documentation and examples
- [ ] No performance impact on existing code
- [ ] Support for configuration inheritance and modification
