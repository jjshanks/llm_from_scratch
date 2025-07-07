# Add BPE Tokenizer Wrapper

**Priority:** Medium
**Status:** Pending
**Commit SHA:** 360fbdf2661ddbc00ce2487da7e6e9a8d6684253

## Description

Currently, the main package only provides `SimpleTokenizerV1` (custom regex-based tokenizer), while BPE tokenization examples exist in `book/ch2/tiktoken_example.py`. The package should provide a consistent interface for both tokenization approaches through a BPE tokenizer wrapper class.

## Justification

- **Modern Standard**: BPE is the standard tokenization method for modern LLMs
- **Consistency**: Unified interface between custom and BPE tokenizers
- **Flexibility**: Allow users to choose appropriate tokenization strategy
- **Production Ready**: BPE tokenizers are what's used in real-world applications
- **Compatibility**: Maintain compatibility with existing tiktoken usage

## Current State

BPE tokenization is demonstrated in `book/ch2/tiktoken_example.py:24-38`:

```python
# Initialize the GPT-2 BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode text to token IDs
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# Decode token IDs back to text
strings = tokenizer.decode(integers)
```

The main package only has `SimpleTokenizerV1` which uses regex-based tokenization.

## Proposed Changes

1. **Create wrapper class**: Add `BPETokenizer` class that wraps tiktoken functionality
2. **Consistent interface**: Match the interface of `SimpleTokenizerV1`
3. **Special token handling**: Support special tokens like `<|endoftext|>` and `<|unk|>`
4. **Configuration options**: Support different BPE encodings (GPT-2, GPT-3, etc.)

## Task List

- [ ] Research tiktoken API to understand all available methods
- [ ] Create `BPETokenizer` class in `src/llm_from_scratch/tokenizer.py`:
  - [ ] Inherit from common base class or implement consistent interface
  - [ ] Support initialization with different encodings ("gpt2", "gpt-3.5-turbo", etc.)
  - [ ] Implement `encode()` method matching `SimpleTokenizerV1` interface
  - [ ] Implement `decode()` method matching `SimpleTokenizerV1` interface
  - [ ] Add `get_vocabulary()` method for consistency
  - [ ] Handle special tokens properly
- [ ] Create abstract base class for tokenizers:
  - [ ] Define common interface all tokenizers must implement
  - [ ] Add type hints and documentation
- [ ] Add comprehensive type hints and docstrings
- [ ] Create factory function for tokenizer selection
- [ ] Update package `__init__.py` to export new classes
- [ ] Implement comprehensive tests
- [ ] Update existing examples to demonstrate usage

## Design Patterns Applied

- **Adapter Pattern**: Wrap tiktoken API to match existing interface
- **Strategy Pattern**: Allow runtime selection of tokenization strategy
- **Factory Pattern**: Create appropriate tokenizer based on configuration
- **Template Method**: Common base class with customizable implementations

## Class Structure

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def get_vocabulary(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        pass


class BPETokenizer(BaseTokenizer):
    """BPE tokenizer wrapper for tiktoken."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        allowed_special: Optional[set] = None,
        disallowed_special: Optional[set] = None,
    ) -> None:
        """Initialize BPE tokenizer."""

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE tokenization."""

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode tokens using BPE tokenization."""

    def get_vocabulary(self) -> Dict[str, int]:
        """Get BPE vocabulary mapping."""

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""

    @property
    def END_OF_TEXT_TOKEN(self) -> str:
        """Get end of text token."""

    @property
    def END_OF_TEXT_INDEX(self) -> int:
        """Get end of text token index."""


def create_tokenizer(
    tokenizer_type: str = "bpe",
    encoding_name: str = "gpt2",
    dataset: Optional[str] = None,
    vocabulary: Optional[Dict[str, int]] = None,
    **kwargs
) -> BaseTokenizer:
    """Factory function to create appropriate tokenizer."""
```

## Testing Requirements

- [ ] Test `BPETokenizer` with different encodings (gpt2, gpt-3.5-turbo)
- [ ] Test encode/decode round-trip consistency
- [ ] Test special token handling
- [ ] Test vocabulary access and properties
- [ ] Test with various text inputs (ASCII, Unicode, empty strings)
- [ ] Test error handling for invalid inputs
- [ ] Compare output with direct tiktoken usage
- [ ] Test integration with existing dataset classes
- [ ] Performance comparison with SimpleTokenizerV1

## Breaking Changes

**None** - This is a new addition that doesn't affect existing code.

## Implementation Notes

1. Consider caching vocabulary lookup for performance
2. Handle tiktoken import gracefully (soft dependency)
3. Add proper error messages for missing tiktoken dependency
4. Consider supporting custom BPE models beyond tiktoken
5. Ensure thread safety for concurrent usage

## Integration Points

```python
# Usage examples that should work:
bpe_tokenizer = BPETokenizer("gpt2")
dataset = GPTDatasetV1(text, bpe_tokenizer, max_length=256, stride=128)

# Or with factory function:
tokenizer = create_tokenizer("bpe", encoding_name="gpt2")
tokenizer = create_tokenizer("simple", dataset=text)
```

## Success Criteria

- [ ] Consistent interface with `SimpleTokenizerV1`
- [ ] Proper special token handling
- [ ] Comprehensive test coverage
- [ ] Clear documentation and examples
- [ ] Integration with existing package components
- [ ] Performance comparable to direct tiktoken usage
- [ ] Graceful handling of missing dependencies
- [ ] Chapter examples updated to show both tokenizer types
