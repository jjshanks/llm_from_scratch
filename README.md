# LLM From Scratch

This project provides the building blocks for creating large language models from scratch. It is based off the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

## Setup

 1. Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

 2. Install the package and its dependencies (via `pyproject.toml`):

    ```bash
    pip install -e .
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
pip install pytest
pytest
```
