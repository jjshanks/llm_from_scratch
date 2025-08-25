"""
Utility functions for LLM training and data processing.

This module provides convenient functions for creating DataLoaders, managing tokenizers,
and other common operations used throughout the LLM training pipeline.
Includes both pretraining utilities and classification fine-tuning utilities.
"""

from typing import Any, Optional, Union, Dict, Tuple
import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader
import json
import os
import urllib.request
import numpy as np

from .dataset import GPTDatasetV1
from .tokenizer import SimpleTokenizerV1


def create_tokenizer(
    text: Optional[str] = None, use_bpe: bool = True, encoding_name: str = "gpt2"
) -> Union[SimpleTokenizerV1, Any]:
    """
    Create a tokenizer for text processing.

    Args:
        text: Raw text to build vocabulary from (required for SimpleTokenizerV1)
        use_bpe: Whether to use BPE tokenizer (tiktoken) or SimpleTokenizerV1
        encoding_name: Name of the BPE encoding to use (default: "gpt2")

    Returns:
        Tokenizer instance (either SimpleTokenizerV1 or tiktoken encoding)

    Raises:
        ValueError: If use_bpe=False but no text provided for SimpleTokenizerV1

    Examples:
        >>> # Create BPE tokenizer
        >>> tokenizer = create_tokenizer(use_bpe=True)

        >>> # Create simple tokenizer
        >>> tokenizer = create_tokenizer(text="sample text", use_bpe=False)
    """
    if use_bpe:
        return tiktoken.get_encoding(encoding_name)
    else:
        if text is None:
            raise ValueError("text parameter is required when use_bpe=False")
        return SimpleTokenizerV1(text)


def create_dataloader_v1(
    txt: str,
    tokenizer: Optional[Union[SimpleTokenizerV1, Any]] = None,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    use_bpe: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for GPT-style training data.

    This function creates a DataLoader that yields input-target pairs for training
    language models. It uses a sliding window approach to generate overlapping
    sequences from the input text.

    Args:
        txt: Raw text to tokenize and create samples from
        tokenizer: Custom tokenizer to use. If None, creates appropriate tokenizer
        batch_size: Number of samples per batch
        max_length: Maximum sequence length (context window)
        stride: Step size for sliding window (controls overlap)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of workers for data loading
        use_bpe: Whether to use BPE tokenizer when tokenizer is None

    Returns:
        DataLoader that yields (input_ids, target_ids) pairs

    Raises:
        ValueError: If max_length <= 0 or stride <= 0
        ValueError: If batch_size <= 0
        ValueError: If txt is empty

    Examples:
        >>> # Using default BPE tokenizer
        >>> dataloader = create_dataloader_v1(text)

        >>> # Using custom tokenizer
        >>> custom_tokenizer = SimpleTokenizerV1(text)
        >>> dataloader = create_dataloader_v1(text, tokenizer=custom_tokenizer)

        >>> # With specific parameters
        >>> dataloader = create_dataloader_v1(
        ...     text,
        ...     batch_size=8,
        ...     max_length=512,
        ...     stride=256
        ... )
    """
    # Parameter validation
    if not txt or not txt.strip():
        raise ValueError("txt parameter cannot be empty")

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(text=txt, use_bpe=use_bpe)

    # Create dataset using sliding window approach
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create DataLoader for efficient batching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def generate_text_simple(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Simple greedy text generation.

    Args:
        model: The GPT model.
        idx: Starting token IDs of shape (batch_size, sequence_length).
        max_new_tokens: The number of new tokens to generate.
        context_size: The maximum context length for the model.

    Returns:
        The generated token IDs of shape (batch_size, sequence_length + max_new_tokens).
    """
    model.eval()  # Set model to evaluation mode

    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions (no gradient needed for inference)
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on the last token's predictions
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Pick the most likely next token (greedy decoding)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate loss for a single batch.

    Args:
        input_batch: Input token IDs of shape (batch_size, sequence_length)
        target_batch: Target token IDs of shape (batch_size, sequence_length)
        model: The GPT model
        device: Device to run computation on

    Returns:
        Cross-entropy loss tensor
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over entire data loader.

    Args:
        data_loader: PyTorch DataLoader with (input, target) pairs
        model: The GPT model
        device: Device to run computation on
        num_batches: Optional limit on number of batches to evaluate

    Returns:
        Average cross-entropy loss as float
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on training and validation sets.

    Args:
        model: The GPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run computation on
        eval_iter: Number of batches to evaluate on each dataset

    Returns:
        Tuple of (train_loss, val_loss) as floats
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Return to training mode
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs tensor.

    Args:
        text: Raw text string to encode
        tokenizer: Tokenizer instance (usually tiktoken encoding)

    Returns:
        Token IDs tensor of shape (1, sequence_length)
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs tensor to text.

    Args:
        token_ids: Token IDs tensor of shape (1, sequence_length) or (sequence_length,)
        tokenizer: Tokenizer instance (usually tiktoken encoding)

    Returns:
        Decoded text string
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_with_temperature(
    model,
    tokenizer,
    device,
    start_context,
    max_new_tokens=25,
    temperature=1.0,
    top_k=None,
):
    """
    Generate text with advanced sampling techniques.

    Args:
        model: The GPT model
        tokenizer: Tokenizer instance (usually tiktoken encoding)
        device: Device to run computation on
        start_context: Starting text string
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more diverse)
        top_k: Limit to top-k tokens for sampling (None = no limit)

    Returns:
        Generated text string including the start_context
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = encoded

        for _ in range(max_new_tokens):
            # Crop to context size
            idx_cond = token_ids[:, -context_size:]

            # Get predictions
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last position

            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            # Apply temperature and sampling
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            token_ids = torch.cat((token_ids, idx_next), dim=1)

    model.train()
    return token_ids_to_text(token_ids, tokenizer)


# ============================================================================
# GPT-2 PRETRAINED WEIGHT UTILITIES
# ============================================================================
# Utilities for downloading and loading pretrained GPT-2 weights from OpenAI


def download_and_load_gpt2(
    model_size: str = "124M", models_dir: str = "gpt2"
) -> Tuple[Dict, Dict]:
    """
    Download and load GPT-2 model weights from OpenAI.

    Args:
        model_size: Size of GPT-2 model ("124M", "355M", "774M", or "1558M")
        models_dir: Directory to save model files

    Returns:
        Tuple of (settings_dict, params_dict) containing model configuration
        and pretrained weights

    Example:
        >>> settings, params = download_and_load_gpt2("124M", "gpt2")
        >>> print(f"Model has {settings['n_layer']} layers")
    """
    # Validate model size
    allowed_sizes = ["124M", "355M", "774M", "1558M"]
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    # Define file URLs and local paths
    base_url = f"https://openaipublic.blob.core.windows.net/gpt-2/models/{model_size}/"
    files_to_download = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    # Download files if they don't exist
    for filename in files_to_download:
        file_url = base_url + filename
        file_path = os.path.join(model_dir, filename)

        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                raise

    # Load hyperparameters
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        hparams = json.load(f)

    # Convert to our expected format
    settings = {
        "n_vocab": hparams["n_vocab"],
        "n_ctx": hparams["n_ctx"],
        "n_embd": hparams["n_embd"],
        "n_head": hparams["n_head"],
        "n_layer": hparams["n_layer"],
    }

    # Load TensorFlow checkpoint (requires tensorflow)
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow is required to load GPT-2 weights. Install with: pip install tensorflow>=2.15.0"
        )

    # Load parameters from checkpoint
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    params = load_gpt2_params_from_tf_ckpt(ckpt_path, settings)

    return settings, params


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Dict) -> Dict:
    """
    Load GPT-2 parameters from TensorFlow checkpoint.

    Args:
        ckpt_path: Path to TensorFlow checkpoint (without extension)
        settings: Model settings dictionary

    Returns:
        Dictionary containing model parameters
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        raise ImportError("TensorFlow is required to load checkpoints")

    params = {"blocks": {}}

    # Load checkpoint
    with tf.compat.v1.Session() as sess:
        # Load the checkpoint variables
        loader = tf.compat.v1.train.import_meta_graph(ckpt_path + ".meta")
        loader.restore(sess, ckpt_path)

        # Get all variables
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        for var in variables:
            name = var.name.rstrip(":0")  # Remove tensor suffix
            array = sess.run(var)

            # Parse variable names and organize into our structure
            if name.startswith("model/"):
                name = name[6:]  # Remove 'model/' prefix

            if name == "wpe":  # Position embedding
                params["wpe"] = array
            elif name == "wte":  # Token embedding
                params["wte"] = array
            elif name == "ln_f/g":  # Final layer norm scale
                params["g"] = array
            elif name == "ln_f/b":  # Final layer norm bias
                params["b"] = array
            elif name.startswith("h"):  # Transformer blocks
                # Parse block number and component
                parts = name.split("/")
                if len(parts) >= 2:
                    block_num = int(
                        parts[0][1:]
                    )  # Extract number from 'h0', 'h1', etc.

                    if block_num not in params["blocks"]:
                        params["blocks"][block_num] = {}

                    # Parse component type
                    component = "/".join(parts[1:])

                    if component.startswith("ln_1/"):
                        if "ln_1" not in params["blocks"][block_num]:
                            params["blocks"][block_num]["ln_1"] = {}
                        if component.endswith("g"):
                            params["blocks"][block_num]["ln_1"]["g"] = array
                        elif component.endswith("b"):
                            params["blocks"][block_num]["ln_1"]["b"] = array

                    elif component.startswith("ln_2/"):
                        if "ln_2" not in params["blocks"][block_num]:
                            params["blocks"][block_num]["ln_2"] = {}
                        if component.endswith("g"):
                            params["blocks"][block_num]["ln_2"]["g"] = array
                        elif component.endswith("b"):
                            params["blocks"][block_num]["ln_2"]["b"] = array

                    elif component.startswith("attn/"):
                        if "attn" not in params["blocks"][block_num]:
                            params["blocks"][block_num]["attn"] = {}

                        if "c_attn/w" in component:
                            if "c_attn" not in params["blocks"][block_num]["attn"]:
                                params["blocks"][block_num]["attn"]["c_attn"] = {}
                            params["blocks"][block_num]["attn"]["c_attn"]["w"] = array
                        elif "c_attn/b" in component:
                            if "c_attn" not in params["blocks"][block_num]["attn"]:
                                params["blocks"][block_num]["attn"]["c_attn"] = {}
                            params["blocks"][block_num]["attn"]["c_attn"]["b"] = array
                        elif "c_proj/w" in component:
                            if "c_proj" not in params["blocks"][block_num]["attn"]:
                                params["blocks"][block_num]["attn"]["c_proj"] = {}
                            params["blocks"][block_num]["attn"]["c_proj"]["w"] = array
                        elif "c_proj/b" in component:
                            if "c_proj" not in params["blocks"][block_num]["attn"]:
                                params["blocks"][block_num]["attn"]["c_proj"] = {}
                            params["blocks"][block_num]["attn"]["c_proj"]["b"] = array

                    elif component.startswith("mlp/"):
                        if "mlp" not in params["blocks"][block_num]:
                            params["blocks"][block_num]["mlp"] = {}

                        if "c_fc/w" in component:
                            if "c_fc" not in params["blocks"][block_num]["mlp"]:
                                params["blocks"][block_num]["mlp"]["c_fc"] = {}
                            params["blocks"][block_num]["mlp"]["c_fc"]["w"] = array
                        elif "c_fc/b" in component:
                            if "c_fc" not in params["blocks"][block_num]["mlp"]:
                                params["blocks"][block_num]["mlp"]["c_fc"] = {}
                            params["blocks"][block_num]["mlp"]["c_fc"]["b"] = array
                        elif "c_proj/w" in component:
                            if "c_proj" not in params["blocks"][block_num]["mlp"]:
                                params["blocks"][block_num]["mlp"]["c_proj"] = {}
                            params["blocks"][block_num]["mlp"]["c_proj"]["w"] = array
                        elif "c_proj/b" in component:
                            if "c_proj" not in params["blocks"][block_num]["mlp"]:
                                params["blocks"][block_num]["mlp"]["c_proj"] = {}
                            params["blocks"][block_num]["mlp"]["c_proj"]["b"] = array

    return params


def assign_check(left: torch.Tensor, right: np.ndarray) -> torch.nn.Parameter:
    """
    Check tensor shapes match and return right as Parameter.

    Args:
        left: Existing PyTorch tensor to check shape against
        right: NumPy array to convert to Parameter

    Returns:
        PyTorch Parameter with same data as right array

    Raises:
        ValueError: If shapes don't match
    """
    # Squeeze out any singleton dimensions from TensorFlow weights
    right_squeezed = np.squeeze(right)
    if left.shape != right_squeezed.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right_squeezed.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right_squeezed))


def load_weights_into_gpt(gpt_model, params: Dict):
    """
    Load pretrained GPT-2 weights into a GPTModel instance.

    This function carefully maps OpenAI's parameter structure to our
    GPTModel implementation, ensuring correct weight loading.

    Args:
        gpt_model: GPTModel instance to load weights into
        params: Parameters dictionary from download_and_load_gpt2

    Example:
        >>> settings, params = download_and_load_gpt2("124M")
        >>> model = GPTModel(config)
        >>> load_weights_into_gpt(model, params)
    """
    # Load embeddings
    gpt_model.pos_emb.weight = assign_check(gpt_model.pos_emb.weight, params["wpe"])
    gpt_model.tok_emb.weight = assign_check(gpt_model.tok_emb.weight, params["wte"])

    # Load transformer blocks
    for b in range(len(params["blocks"])):
        # Attention weights - split combined QKV weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt_model.trf_blocks[b].att.W_query.weight = assign_check(
            gpt_model.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt_model.trf_blocks[b].att.W_key.weight = assign_check(
            gpt_model.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt_model.trf_blocks[b].att.W_value.weight = assign_check(
            gpt_model.trf_blocks[b].att.W_value.weight, v_w.T
        )

        # Attention biases - split combined QKV biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt_model.trf_blocks[b].att.W_query.bias = assign_check(
            gpt_model.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt_model.trf_blocks[b].att.W_key.bias = assign_check(
            gpt_model.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt_model.trf_blocks[b].att.W_value.bias = assign_check(
            gpt_model.trf_blocks[b].att.W_value.bias, v_b
        )

        # Attention output projection
        gpt_model.trf_blocks[b].att.out_proj.weight = assign_check(
            gpt_model.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt_model.trf_blocks[b].att.out_proj.bias = assign_check(
            gpt_model.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # Feed-forward layers
        gpt_model.trf_blocks[b].ff.layers[0].weight = assign_check(
            gpt_model.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt_model.trf_blocks[b].ff.layers[0].bias = assign_check(
            gpt_model.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt_model.trf_blocks[b].ff.layers[2].weight = assign_check(
            gpt_model.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt_model.trf_blocks[b].ff.layers[2].bias = assign_check(
            gpt_model.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # Layer normalization
        gpt_model.trf_blocks[b].norm1.scale = assign_check(
            gpt_model.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt_model.trf_blocks[b].norm1.shift = assign_check(
            gpt_model.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt_model.trf_blocks[b].norm2.scale = assign_check(
            gpt_model.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt_model.trf_blocks[b].norm2.shift = assign_check(
            gpt_model.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    # Final layer normalization
    gpt_model.final_norm.scale = assign_check(gpt_model.final_norm.scale, params["g"])
    gpt_model.final_norm.shift = assign_check(gpt_model.final_norm.shift, params["b"])


def get_gpt2_model_configs() -> Dict[str, Dict]:
    """
    Get configuration dictionaries for different GPT-2 model sizes.

    Returns:
        Dictionary mapping model names to their configurations

    Example:
        >>> configs = get_gpt2_model_configs()
        >>> config_124m = configs["gpt2-small (124M)"]
    """
    return {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }


# ============================================================================
# CLASSIFICATION UTILITIES
# ============================================================================
# The following functions are specifically for fine-tuning models on
# classification tasks. They differ from pretraining utilities by focusing
# on discrete class prediction rather than next-token prediction.


def calc_classification_accuracy(data_loader, model, device, num_batches=None):
    """
    Calculate classification accuracy over a DataLoader.

    This function differs from language modeling evaluation by focusing on
    class prediction accuracy rather than token prediction perplexity.

    Args:
        data_loader: PyTorch DataLoader with (inputs, targets) where targets are class labels
        model: Classification model (should output logits for last token position)
        device: Device to run computation on
        num_batches: Limit evaluation to first N batches (None for all)

    Returns:
        accuracy: Accuracy as a float between 0 and 1

    Example:
        >>> accuracy = calc_classification_accuracy(val_loader, model, device)
        >>> print(f"Validation accuracy: {accuracy:.2%}")
    """
    model.eval()
    correct_predictions = 0
    num_examples = 0

    # Determine how many batches to evaluate
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Get model predictions - use last token position for classification
            logits = model(input_batch)[:, -1, :]  # [batch_size, num_classes]

            # Get predicted class
            predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size]

            # Update counters
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples if num_examples > 0 else 0.0


def calc_classification_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate classification loss for a single batch.

    This differs from language modeling loss by using only the last token
    position and comparing against discrete class labels.

    Args:
        input_batch: Input token IDs of shape [batch_size, seq_len]
        target_batch: Target class labels of shape [batch_size]
        model: Classification model
        device: Device to run computation on

    Returns:
        Cross-entropy loss for classification
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Get logits for last token position (classification prediction)
    logits = model(input_batch)[:, -1, :]  # [batch_size, num_classes]

    # Calculate cross-entropy loss against class labels
    loss = F.cross_entropy(logits, target_batch)

    return loss


def calc_classification_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average classification loss over a DataLoader.

    Args:
        data_loader: PyTorch DataLoader with (inputs, targets)
        model: Classification model
        device: Device to run computation on
        num_batches: Limit evaluation to first N batches (None for all)

    Returns:
        Average classification loss as float
    """
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.0

    # Determine how many batches to evaluate
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = calc_classification_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()

    return total_loss / num_batches


def evaluate_classification_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate classification model on training and validation sets.

    This provides both loss and accuracy metrics for classification tasks.

    Args:
        model: Classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run computation on
        eval_iter: Number of batches to evaluate on each dataset

    Returns:
        Dict with train_loss, val_loss, train_accuracy, val_accuracy
    """
    model.eval()

    with torch.no_grad():
        # Calculate losses
        train_loss = calc_classification_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_classification_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        # Calculate accuracies
        train_accuracy = calc_classification_accuracy(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_classification_accuracy(
            val_loader, model, device, num_batches=eval_iter
        )

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }


def classify_text_simple(
    text, model, tokenizer, device, max_length=120, pad_token_id=50256, class_names=None
):
    """
    Classify a single text using a trained classification model.

    This is a convenience function for inference on individual texts,
    handling tokenization, padding, and prediction in a single call.

    Args:
        text: Text string to classify
        model: Trained classification model
        tokenizer: Tokenizer for encoding text (e.g., tiktoken)
        device: Device to run inference on
        max_length: Maximum sequence length for padding/truncation
        pad_token_id: Token ID for padding (default: <|endoftext|>)
        class_names: List of class names for human-readable output

    Returns:
        Dict with predicted_class, confidence, probabilities, and class_names

    Example:
        >>> result = classify_text_simple(
        ...     "Win £1000 now!", model, tokenizer, device,
        ...     class_names=["not spam", "spam"]
        ... )
        >>> print(f"Prediction: {result['predicted_label']} ({result['confidence']:.3f})")
    """
    if class_names is None:
        class_names = ["class_0", "class_1"]  # Generic names

    model.eval()

    # Tokenize and preprocess
    token_ids = tokenizer.encode(text)

    # Apply length constraints
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]  # Truncate

    # Pad to max_length
    token_ids += [pad_token_id] * (max_length - len(token_ids))

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Use last token
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

    return {
        "text": text,
        "predicted_class": predicted_class,
        "predicted_label": (
            class_names[predicted_class]
            if predicted_class < len(class_names)
            else f"class_{predicted_class}"
        ),
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().tolist(),
        "class_names": class_names,
    }
