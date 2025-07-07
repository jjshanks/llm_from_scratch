from llm_from_scratch.tokenizer import SimpleTokenizerV1


def test_encode_decode_simple():
    text = "hello world"
    tokenizer = SimpleTokenizerV1(dataset=text)
    encoded = tokenizer.encode(text)
    # Expect one index per token
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_punctuation_handling():
    text = "hello, world!"
    tokenizer = SimpleTokenizerV1(dataset=text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_unknown_token_mapping():
    # Unknown tokens should map to UNKNOWN_TOKEN_INDEX
    tokenizer = SimpleTokenizerV1(vocabulary={"a": 0})
    encoded = tokenizer.encode("b")
    assert encoded == [tokenizer.UNKNOWN_TOKEN_INDEX]
    # Decoding unknown token index yields UNKNOWN_TOKEN
    decoded = tokenizer.decode(encoded)
    assert decoded == tokenizer.UNKNOWN_TOKEN

    # Test encoding and decoding of the unknown token itself
    encoded_unk = tokenizer.encode(tokenizer.UNKNOWN_TOKEN)
    assert encoded_unk == [tokenizer.UNKNOWN_TOKEN_INDEX]
    decoded_unk = tokenizer.decode(encoded_unk)
    assert decoded_unk == tokenizer.UNKNOWN_TOKEN

    # Test encoding and decoding of the end-of-text token
    encoded_eot = tokenizer.encode(tokenizer.END_OF_TEXT_TOKEN)
    assert encoded_eot == [tokenizer.END_OF_TEXT_INDEX]
    decoded_eot = tokenizer.decode(encoded_eot)
    assert decoded_eot == tokenizer.END_OF_TEXT_TOKEN


def test_custom_vocab_initialization():
    vocab = {"a": 1, "b": 2}
    tokenizer = SimpleTokenizerV1(vocabulary=vocab)
    assert tokenizer.get_vocabulary() == vocab
    encoded = tokenizer.encode("a b")
    assert encoded == [1, 2]
    decoded = tokenizer.decode([2, 1])
    assert decoded == "b a"


def test_special_tokens_in_vocab():
    # Special tokens should be in vocabulary
    tokenizer = SimpleTokenizerV1(dataset="a")
    vocab = tokenizer.get_vocabulary()
    assert SimpleTokenizerV1.END_OF_TEXT_TOKEN in vocab
    assert SimpleTokenizerV1.UNKNOWN_TOKEN in vocab
    # Indices should match tokenizer attributes
    assert tokenizer.END_OF_TEXT_INDEX == vocab[SimpleTokenizerV1.END_OF_TEXT_TOKEN]
    assert tokenizer.UNKNOWN_TOKEN_INDEX == vocab[SimpleTokenizerV1.UNKNOWN_TOKEN]
