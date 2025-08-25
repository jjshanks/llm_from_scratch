"""
Classification utilities and models for fine-tuning LLMs.

This module provides the core components needed for text classification tasks,
including specialized dataset classes, model wrappers, and training utilities.
These are used throughout Chapter 6 for spam detection fine-tuning.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def get_last_token_positions(input_batch, pad_token_id=50256):
    """
    Find the position of the last non-padding token for each sequence in a batch.

    This is crucial for classification tasks where we need to use the last actual
    content token rather than a padding token for making predictions.

    Args:
        input_batch: Input tensor [batch_size, seq_len] with token IDs
        pad_token_id: Token ID used for padding (default: 50256 = <|endoftext|>)

    Returns:
        positions: Tensor [batch_size] with position indices of last non-padding tokens
    """
    batch_size, seq_len = input_batch.shape
    positions = torch.zeros(batch_size, dtype=torch.long, device=input_batch.device)

    for i in range(batch_size):
        # Find all non-padding positions for this sequence
        non_pad_mask = input_batch[i] != pad_token_id
        non_pad_positions = non_pad_mask.nonzero(as_tuple=True)[0]

        if len(non_pad_positions) > 0:
            # Use the last non-padding position
            positions[i] = non_pad_positions[-1]
        else:
            # Fallback: if entire sequence is padding (shouldn't happen), use last position
            positions[i] = seq_len - 1

    return positions


def extract_last_token_logits(model_output, input_batch, pad_token_id=50256):
    """
    Extract logits from the last non-padding token position for each sequence.

    Args:
        model_output: Model output tensor [batch_size, seq_len, hidden_size or num_classes]
        input_batch: Input token IDs [batch_size, seq_len]
        pad_token_id: Token ID used for padding

    Returns:
        logits: Tensor [batch_size, num_classes] with logits from last content tokens
    """
    batch_size = model_output.shape[0]
    last_positions = get_last_token_positions(input_batch, pad_token_id)

    # Extract logits at the last non-padding position for each sequence
    logits = torch.stack(
        [model_output[i, last_positions[i], :] for i in range(batch_size)]
    )

    return logits


class SpamDataset(Dataset):
    """
    PyTorch Dataset for spam classification with text padding.

    This dataset handles variable-length text messages by tokenizing them
    and applying consistent padding/truncation strategies for batch processing.
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer,
        max_length: Optional[int] = None,
        pad_token_id: int = 50256,
    ):
        """
        Initialize the spam classification dataset.

        Args:
            csv_file: Path to CSV file with 'Text' and 'Label' columns
            tokenizer: Tokenizer for encoding text (e.g., tiktoken)
            max_length: Maximum sequence length (None for auto-detect from 95th percentile)
            pad_token_id: Token ID to use for padding (default: <|endoftext|>)
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

        # Validate data format
        if not all(col in self.data.columns for col in ["Text", "Label"]):
            raise ValueError("CSV must contain 'Text' and 'Label' columns")

        # Tokenize all texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Determine optimal max_length
        if max_length is None:
            self.max_length = self._find_optimal_length()
        else:
            self.max_length = max_length

        # Apply padding and truncation
        self._process_sequences()

    def _find_optimal_length(self) -> int:
        """Find optimal sequence length covering 95% of data."""
        lengths = [len(tokens) for tokens in self.encoded_texts]
        return int(np.percentile(lengths, 95))

    def _process_sequences(self):
        """Apply truncation and padding to all sequences."""
        processed_texts = []

        for tokens in self.encoded_texts:
            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]

            # Pad if too short
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            processed_texts.append(tokens)

        self.encoded_texts = processed_texts

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example from the dataset.

        Args:
            idx: Index of the example

        Returns:
            tuple: (input_tokens, label) where:
                - input_tokens: tensor of token IDs [max_length]
                - label: tensor with class label [1]
        """
        encoded_text = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]

        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


class ClassificationGPT(torch.nn.Module):
    """
    GPT model wrapper for classification tasks.

    This class wraps a pretrained GPT model and adds a classification head,
    with utilities for freezing layers and configuring for fine-tuning.
    """

    def __init__(self, base_model, num_classes: int = 2, freeze_base: bool = True):
        """
        Initialize classification wrapper around GPT model.

        Args:
            base_model: The base GPT model (e.g., GPTModel instance)
            num_classes: Number of output classes
            freeze_base: Whether to freeze the base model parameters
        """
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        # Get embedding dimension from the base model
        self.emb_dim = base_model.out_head.in_features

        # Replace the output head with classification head
        self.base_model.out_head = torch.nn.Linear(self.emb_dim, num_classes)

        # Optionally freeze base model parameters
        if freeze_base:
            self.freeze_base_layers()

    def freeze_base_layers(self, unfreeze_last_n_blocks: int = 1):
        """
        Freeze base model parameters except for the last N transformer blocks.

        Args:
            unfreeze_last_n_blocks: Number of last transformer blocks to keep trainable
        """
        # Freeze all parameters first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        if hasattr(self.base_model, "trf_blocks"):
            for block in self.base_model.trf_blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze final layer norm
        if hasattr(self.base_model, "final_norm"):
            for param in self.base_model.final_norm.parameters():
                param.requires_grad = True

        # Classification head is trainable by default
        for param in self.base_model.out_head.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input token tensor [batch_size, seq_len]

        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Get full model output [batch_size, seq_len, num_classes]
        output = self.base_model(x)

        # For classification, use the last non-padding token position
        # This ensures we're using actual content, not padding tokens
        last_token_logits = extract_last_token_logits(output, x)

        return last_token_logits

    def get_trainable_params(self):
        """Get list of trainable parameters for optimizer."""
        return [p for p in self.base_model.parameters() if p.requires_grad]

    def count_parameters(self) -> dict:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": (trainable / total) * 100,
        }


def calc_accuracy_loader(data_loader, model, device, num_batches: Optional[int] = None):
    """
    Calculate classification accuracy over a DataLoader.

    Args:
        data_loader: PyTorch DataLoader with (inputs, targets)
        model: Classification model
        device: Device to run computation on
        num_batches: Limit evaluation to first N batches (None for all)

    Returns:
        accuracy: Accuracy as a float between 0 and 1
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

            # Get model predictions - use last non-padding token for classification
            model_output = model(input_batch)  # [batch_size, seq_len, num_classes]
            logits = extract_last_token_logits(
                model_output, input_batch
            )  # [batch_size, num_classes]

            # Get predicted class
            predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size]

            # Update counters
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples if num_examples > 0 else 0.0


def calc_loss_batch(input_batch, target_batch, model, device, loss_fn=None):
    """
    Calculate loss for a single batch.

    Args:
        input_batch: Input tokens [batch_size, seq_len]
        target_batch: Target labels [batch_size]
        model: Classification model
        device: Device to run computation on
        loss_fn: Loss function (default: CrossEntropyLoss)

    Returns:
        loss: Scalar loss value
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Get model predictions - use last non-padding token for classification
    model_output = model(input_batch)  # [batch_size, seq_len, num_classes]
    logits = extract_last_token_logits(
        model_output, input_batch
    )  # [batch_size, num_classes]

    # Calculate loss
    loss = loss_fn(logits, target_batch)

    return loss


def calc_loss_loader(
    data_loader, model, device, num_batches: Optional[int] = None, loss_fn=None
):
    """
    Calculate average loss over a DataLoader.

    Args:
        data_loader: PyTorch DataLoader with (inputs, targets)
        model: Classification model
        device: Device to run computation on
        num_batches: Limit evaluation to first N batches (None for all)
        loss_fn: Loss function (default: CrossEntropyLoss)

    Returns:
        avg_loss: Average loss across batches
    """
    if len(data_loader) == 0:
        return float("nan")

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0

    # Determine how many batches to evaluate
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = calc_loss_batch(input_batch, target_batch, model, device, loss_fn)
            total_loss += loss.item()

    return total_loss / num_batches


def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int = 5,
    eval_freq: int = 50,
    eval_iter: int = 5,
    loss_fn=None,
):
    """
    Simple training loop for classification fine-tuning.

    Args:
        model: Classification model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (e.g., AdamW)
        device: Device to train on
        num_epochs: Number of training epochs
        eval_freq: Evaluate every N steps
        eval_iter: Number of batches to use for evaluation
        loss_fn: Loss function (default: CrossEntropyLoss)

    Returns:
        dict: Training history with losses and accuracies
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Calculate loss
            loss = calc_loss_batch(input_batch, target_batch, model, device, loss_fn)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update counters
            examples_seen += input_batch.shape[0]
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                # Calculate losses
                train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn
                )
                val_loss = calc_loss_loader(
                    val_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        print(
            f"Training accuracy: {train_accuracy*100:.2f}% | "
            f"Validation accuracy: {val_accuracy*100:.2f}%"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "examples_seen": examples_seen,
        "steps": global_step + 1,
    }


def classify_text(
    text: str,
    model,
    tokenizer,
    device,
    max_length: int = 120,
    pad_token_id: int = 50256,
    class_names: list = None,
):
    """
    Classify a single text using the trained model.

    Args:
        text: Text to classify
        model: Trained classification model
        tokenizer: Tokenizer for encoding text
        device: Device to run inference on
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        class_names: List of class names for output (e.g., ["not spam", "spam"])

    Returns:
        dict: Classification results with prediction, confidence, and probabilities
    """
    if class_names is None:
        class_names = ["not spam", "spam"]

    model.eval()

    # Tokenize and pad
    token_ids = tokenizer.encode(text)

    # Truncate if too long
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]

    # Pad if too short
    token_ids += [pad_token_id] * (max_length - len(token_ids))

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        if hasattr(model, "forward"):
            # Handle ClassificationGPT wrapper
            logits = model(input_tensor)
        else:
            # Handle raw GPT model - use last token
            logits = model(input_tensor)[:, -1, :]

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

    return {
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
