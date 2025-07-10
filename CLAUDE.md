# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides building blocks for creating large language models from scratch, based on the book "Build a Large Language Model (From Scratch)" by Manning. The codebase is structured as a Python package with a focus on tokenization functionality.

## Development Setup

```bash
# Install package in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e '.[dev]'
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tokenizer.py

# Run with verbose output
uv run pytest -v
```

## Architecture

The project follows a standard Python package structure:
- `src/llm_from_scratch/` - Main package containing core implementations
- `src/llm_from_scratch/tokenizer.py` - SimpleTokenizerV1 class with vocabulary building and encoding/decoding
- `src/llm_from_scratch/dataset.py` - GPTDatasetV1 class for creating training data with sliding window approach
- `book/ch2/` - Chapter 2 implementation examples and utilities
- `tests/` - pytest test suite
- `pyproject.toml` - Project configuration and dependencies

## Key Components

### SimpleTokenizerV1 Class
The main `SimpleTokenizerV1` class handles:
- Vocabulary building from text datasets
- Text preprocessing and tokenization
- Encoding text to integer sequences
- Decoding integer sequences back to text
- Special token handling (`<|endoftext|>` and `<|unk|>`)

### GPTDatasetV1 Class
The `GPTDatasetV1` PyTorch Dataset class provides:
- Sliding window approach for creating input-target pairs
- Configurable stride and context length
- Integration with PyTorch DataLoader for batch processing

## Task Completion

Before committing changes:
1. Run `uv run pytest` to ensure all tests pass
2. Verify package imports work: `from llm_from_scratch.tokenizer import SimpleTokenizerV1` and `from llm_from_scratch.dataset import GPTDatasetV1`
3. Ensure new dependencies are added to pyproject.toml

## Tech Stack

- Python 3.12 (3.13 not supported due to TensorFlow compatibility)
- PyTorch for deep learning components
- TikToken for tokenization utilities
- pytest for testing
- Additional ML libraries: matplotlib, tensorflow, numpy, pandas
