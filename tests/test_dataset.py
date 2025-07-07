import torch
from llm_from_scratch.dataset import GPTDatasetV1
from llm_from_scratch.tokenizer import SimpleTokenizerV1


def test_gpt_dataset_initialization():
    """Test GPTDatasetV1 initialization with basic parameters."""
    txt = "hello world test data sequence with enough tokens for samples"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length=4, stride=1)

    assert len(dataset) > 0
    assert hasattr(dataset, "input_ids")
    assert hasattr(dataset, "target_ids")
    assert len(dataset.input_ids) == len(dataset.target_ids)


def test_gpt_dataset_tensor_types():
    """Test that dataset returns proper tensor types."""
    txt = "hello world test data"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length=3, stride=1)

    input_ids, target_ids = dataset[0]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(target_ids, torch.Tensor)
    assert input_ids.shape == target_ids.shape


def test_gpt_dataset_sequence_length():
    """Test that sequences have correct length."""
    txt = "hello world test data sequence"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    max_length = 4
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=1)

    input_ids, target_ids = dataset[0]
    assert len(input_ids) == max_length
    assert len(target_ids) == max_length


def test_gpt_dataset_target_shift():
    """Test that target sequence is shifted by one position."""
    txt = "hello world test data sequence"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length=3, stride=1)

    input_ids, target_ids = dataset[0]
    # Target should be input shifted by one position
    token_ids = tokenizer.encode(txt)
    expected_input = token_ids[0:3]
    expected_target = token_ids[1:4]

    assert input_ids.tolist() == expected_input
    assert target_ids.tolist() == expected_target


def test_gpt_dataset_stride_functionality():
    """Test that stride parameter works correctly."""
    txt = "one two three four five six"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    max_length = 3
    stride = 2
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)

    token_ids = tokenizer.encode(txt)
    expected_samples = (len(token_ids) - max_length) // stride + 1

    # Should create samples with stride spacing
    assert len(dataset) <= expected_samples

    if len(dataset) >= 2:
        input1, _ = dataset[0]
        input2, _ = dataset[1]
        # Second sample should start 'stride' positions after first
        assert input2[0].item() == token_ids[stride]


def test_gpt_dataset_empty_handling():
    """Test dataset behavior with minimal text."""
    txt = "a b c"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length=2, stride=1)

    # Should handle short sequences gracefully
    assert len(dataset) >= 0
    if len(dataset) > 0:
        input_ids, target_ids = dataset[0]
        assert len(input_ids) == 2
        assert len(target_ids) == 2


def test_gpt_dataset_large_stride():
    """Test dataset with stride larger than sequence overlap."""
    txt = "one two three four five six seven eight"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    max_length = 3
    stride = 4
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)

    # Should still create valid samples
    assert len(dataset) >= 0
    if len(dataset) > 0:
        input_ids, target_ids = dataset[0]
        assert len(input_ids) == max_length
        assert len(target_ids) == max_length


def test_gpt_dataset_indexing():
    """Test dataset indexing and bounds."""
    txt = "hello world test data"
    tokenizer = SimpleTokenizerV1(dataset=txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length=3, stride=1)

    # Test valid indexing
    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)

    # Test bounds
    dataset_len = len(dataset)
    assert dataset_len == len(dataset.input_ids)
    assert dataset_len == len(dataset.target_ids)
