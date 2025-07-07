# LLM From Scratch

This project provides the building blocks for creating large language models from scratch. It is based off the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

The codebase includes:
- **Tokenization**: SimpleTokenizerV1 class for text preprocessing and vocabulary building
- **Dataset Creation**: GPTDatasetV1 class for generating training data with sliding window approach
- **Chapter Examples**: Complete implementations from the book in the `book/ch2/` directory

## Setup

1. Install the package and its dependencies (via `pyproject.toml`):

   ```bash
   uv sync
   ```

2. For development with additional tools (pytest, black, ruff, pre-commit):

   ```bash
   uv sync --extra dev
   ```

## Usage

Once installed, you can import and use the tokenizer and dataset classes:

```python
from llm_from_scratch.tokenizer import SimpleTokenizerV1
from llm_from_scratch.dataset import GPTDatasetV1

# Tokenization
tokenizer = SimpleTokenizerV1(dataset="Hello, world!")
encoded = tokenizer.encode("Hello, world!")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)

# Dataset for training
dataset = GPTDatasetV1(txt="Hello, world!", tokenizer=tokenizer, max_length=4, stride=1)
print(len(dataset))
```

## Running Tests

This project includes pytest tests under `tests/`. To run them:

```bash
uv run pytest
```

## Development Tools

The project includes several development tools:

- **Testing**: `uv run pytest` or `uv run pytest -v` for verbose output
- **Code formatting**: `uv run black .` to format code
- **Linting**: `uv run ruff check .` to check for issues
- **Pre-commit hooks**: `uv run pre-commit install` to set up automatic checks

## Requirements

- Python 3.12 (3.13 not supported due to TensorFlow compatibility)
