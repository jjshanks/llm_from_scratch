# LLM From Scratch

This project provides the building blocks for creating large language models from scratch. It is based off the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

## Setup

 1. Install the package and its dependencies (via `pyproject.toml`):

    ```bash
    uv pip install -e .
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
uv pip install pytest
pytest
```
