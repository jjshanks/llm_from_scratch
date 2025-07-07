# LLM From Scratch

This project provides the building blocks for creating large language models from scratch. It is based off the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

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

Once installed, you can import and use the `Tokenizer` class:

```python
from llm_from_scratch.tokenizer import Tokenizer

tokenizer = Tokenizer(dataset="Hello, world!")
encoded = tokenizer.encode("Hello, world!")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
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
