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
pytest

# Run specific test file
pytest tests/test_tokenizer.py

# Run with verbose output
pytest -v
```

## Architecture

The project follows a standard Python package structure:
- `src/llm_from_scratch/` - Main package containing core implementations
- `src/llm_from_scratch/tokenizer.py` - Tokenizer class with vocabulary building and encoding/decoding
- `tests/` - pytest test suite
- `pyproject.toml` - Project configuration and dependencies

## Key Components

### Tokenizer Class
The main `Tokenizer` class handles:
- Vocabulary building from text datasets
- Text preprocessing and tokenization
- Encoding text to integer sequences
- Decoding integer sequences back to text
- Special token handling (`<|endoftext|>` and `<|unk|>`)

## Task Completion

Before committing changes:
1. Run `pytest` to ensure all tests pass
2. Verify package imports work: `from llm_from_scratch.tokenizer import Tokenizer`
3. Ensure new dependencies are added to pyproject.toml

## Tech Stack

- Python 3.12 (3.13 not supported due to TensorFlow compatibility)
- PyTorch for deep learning components
- TikToken for tokenization utilities
- pytest for testing
- Additional ML libraries: matplotlib, tensorflow, numpy, pandas