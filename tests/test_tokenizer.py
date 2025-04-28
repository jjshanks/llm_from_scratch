import pytest

from llm_from_scratch.tokenizer import Tokenizer


def test_encode_decode_simple():
    text = "hello world"
    tokenizer = Tokenizer(dataset=text)
    encoded = tokenizer.encode(text)
    # Expect one index per token
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_punctuation_handling():
    text = "hello, world!"
    tokenizer = Tokenizer(dataset=text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_unknown_token_error():
    tokenizer = Tokenizer(vocabulary={"a": 0})
    with pytest.raises(ValueError) as exc:
        tokenizer.encode("b")
    assert "Unknown token" in str(exc.value)


def test_custom_vocab_initialization():
    vocab = {"a": 1, "b": 2}
    tokenizer = Tokenizer(vocabulary=vocab)
    assert tokenizer.get_vocabulary() == vocab
    encoded = tokenizer.encode("a b")
    assert encoded == [1, 2]
    decoded = tokenizer.decode([2, 1])
    assert decoded == "b a"
