# Enhance Package Structure

**Priority:** Low
**Status:** Pending
**Commit SHA:** 360fbdf2661ddbc00ce2487da7e6e9a8d6684253

## Description

The package currently has a minimal `__init__.py` file that doesn't expose the main components properly. The package structure should be enhanced to provide better discoverability, proper imports, version information, and a more professional package interface.

## Justification

- **Discoverability**: Make package components easily accessible
- **User Experience**: Provide intuitive import paths
- **Professional Polish**: Standard package structure expectations
- **Version Management**: Proper version tracking and display
- **Documentation**: Clear package-level documentation
- **Compatibility**: Follow Python packaging best practices

## Current State

The current `src/llm_from_scratch/__init__.py` is minimal:

```python
# llm_from_scratch package
```

Users must import components with full paths:
```python
from llm_from_scratch.tokenizer import SimpleTokenizerV1
from llm_from_scratch.dataset import GPTDatasetV1
```

## Proposed Changes

1. **Enhanced `__init__.py`**: Add proper imports and metadata
2. **Version management**: Add version tracking
3. **Public API**: Define clear public interface
4. **Documentation**: Add package-level docstrings
5. **Convenience imports**: Enable shorter import paths

## Task List

- [ ] Create `src/llm_from_scratch/_version.py` for version management
- [ ] Update `src/llm_from_scratch/__init__.py` with:
  - [ ] Package docstring with description and usage examples
  - [ ] Version information import
  - [ ] Public API imports
  - [ ] `__all__` definition for controlled exports
  - [ ] Author and license information
- [ ] Add convenience imports for all public components
- [ ] Create package-level constants and utilities
- [ ] Update `pyproject.toml` to use dynamic version from code
- [ ] Add proper logging configuration
- [ ] Create package-level exceptions
- [ ] Add backwards compatibility considerations
- [ ] Update documentation to reflect new import paths

## Design Patterns Applied

- **Facade Pattern**: Provide simplified interface to complex subsystems
- **Module Pattern**: Organize related functionality into logical modules
- **Namespace Pattern**: Clear separation of public and private APIs

## Enhanced Package Structure

```python
# src/llm_from_scratch/_version.py
__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# src/llm_from_scratch/__init__.py
"""
LLM from Scratch - Building blocks for creating large language models.

This package provides the fundamental components needed to build large language
models from scratch, including tokenizers, datasets, embeddings, and utilities.

Example usage:
    >>> from llm_from_scratch import SimpleTokenizerV1, GPTDatasetV1
    >>>
    >>> # Create tokenizer and dataset
    >>> tokenizer = SimpleTokenizerV1(dataset=text)
    >>> dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=128)
    >>>
    >>> # Create embeddings
    >>> from llm_from_scratch import CombinedEmbedding
    >>> embeddings = CombinedEmbedding(vocab_size=1000, embed_dim=512, max_len=256)

Components:
    - Tokenizers: SimpleTokenizerV1, BPETokenizer
    - Datasets: GPTDatasetV1
    - Embeddings: TokenEmbedding, PositionalEmbedding, CombinedEmbedding
    - Utilities: create_dataloader_v1, create_tokenizer
    - Configuration: TokenizerConfig, DatasetConfig, EmbeddingConfig, LLMConfig
"""

import logging
from typing import List

# Version information
from ._version import __version__, __version_info__

# Core components
from .tokenizer import SimpleTokenizerV1, BaseTokenizer
from .dataset import GPTDatasetV1

# Utilities (when implemented)
try:
    from .utils import create_dataloader_v1, create_tokenizer
except ImportError:
    # Graceful degradation if utils not yet implemented
    pass

# Embeddings (when implemented)
try:
    from .embedding import TokenEmbedding, PositionalEmbedding, CombinedEmbedding
except ImportError:
    # Graceful degradation if embeddings not yet implemented
    pass

# Configuration (when implemented)
try:
    from .config import TokenizerConfig, DatasetConfig, EmbeddingConfig, LLMConfig
except ImportError:
    # Graceful degradation if config not yet implemented
    pass

# BPE Tokenizer (when implemented)
try:
    from .tokenizer import BPETokenizer
except ImportError:
    # Graceful degradation if BPE tokenizer not yet implemented
    pass

# Package metadata
__author__ = "Manning Publications"
__email__ = "support@manning.com"
__license__ = "MIT"
__description__ = "Building blocks for creating large language models from scratch"
__url__ = "https://github.com/manning/llm-from-scratch"

# Define public API
__all__: List[str] = [
    # Version
    "__version__",
    "__version_info__",

    # Core components
    "SimpleTokenizerV1",
    "BaseTokenizer",
    "GPTDatasetV1",

    # Utilities (conditional)
    "create_dataloader_v1",
    "create_tokenizer",

    # Embeddings (conditional)
    "TokenEmbedding",
    "PositionalEmbedding",
    "CombinedEmbedding",

    # Configuration (conditional)
    "TokenizerConfig",
    "DatasetConfig",
    "EmbeddingConfig",
    "LLMConfig",

    # BPE Tokenizer (conditional)
    "BPETokenizer",
]

# Configure package logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


# Package-level exceptions
class LLMFromScratchError(Exception):
    """Base exception for llm_from_scratch package."""
    pass


class TokenizationError(LLMFromScratchError):
    """Exception raised for tokenization errors."""
    pass


class DatasetError(LLMFromScratchError):
    """Exception raised for dataset errors."""
    pass


class EmbeddingError(LLMFromScratchError):
    """Exception raised for embedding errors."""
    pass


class ConfigurationError(LLMFromScratchError):
    """Exception raised for configuration errors."""
    pass


# Package-level constants
DEFAULT_UNKNOWN_TOKEN = "<|unk|>"
DEFAULT_END_OF_TEXT_TOKEN = "<|endoftext|>"
DEFAULT_VOCAB_SIZE = 50257  # GPT-2 vocabulary size
DEFAULT_EMBED_DIM = 768     # Common embedding dimension
DEFAULT_MAX_LENGTH = 1024   # Common context length

# Convenience functions
def get_version() -> str:
    """Get package version string."""
    return __version__


def get_version_info() -> tuple:
    """Get package version tuple."""
    return __version_info__


def list_components() -> List[str]:
    """List all available components in the package."""
    return [item for item in __all__ if not item.startswith("_")]
```

## Testing Requirements

- [ ] Test package imports work correctly
- [ ] Test version information is accessible
- [ ] Test public API components are available
- [ ] Test graceful degradation for unimplemented components
- [ ] Test package-level exceptions
- [ ] Test convenience functions
- [ ] Test backwards compatibility
- [ ] Test import performance (shouldn't be slow)

## Breaking Changes

**None** - This enhancement maintains backwards compatibility while adding new convenience features.

## Implementation Notes

1. Use conditional imports to handle components that may not be implemented yet
2. Follow PEP 8 and Python packaging guidelines
3. Add proper logging configuration
4. Consider lazy loading for performance
5. Add package-level constants for common values
6. Ensure all imports are properly type-hinted

## Dynamic Version Management

Update `pyproject.toml` to use dynamic versioning:

```toml
[project]
name = "llm_from_scratch"
dynamic = ["version"]
description = "Building blocks for creating large language models from scratch"
authors = [
    {name = "Manning Publications", email = "support@manning.com"},
]
license = {text = "MIT"}

[tool.setuptools.dynamic]
version = {attr = "llm_from_scratch._version.__version__"}
```

## Usage Examples After Enhancement

```python
# Simple imports
from llm_from_scratch import SimpleTokenizerV1, GPTDatasetV1

# Version information
import llm_from_scratch
print(llm_from_scratch.__version__)

# List available components
print(llm_from_scratch.list_components())

# Exception handling
from llm_from_scratch import TokenizationError
try:
    tokenizer.encode(invalid_text)
except TokenizationError as e:
    print(f"Tokenization failed: {e}")
```

## Success Criteria

- [ ] Clean, professional package interface
- [ ] Proper version management
- [ ] Comprehensive public API exposure
- [ ] Graceful handling of unimplemented components
- [ ] Package-level documentation and examples
- [ ] Backwards compatibility maintained
- [ ] Follow Python packaging best practices
- [ ] Performance not impacted by imports
