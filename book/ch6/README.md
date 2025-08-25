# Chapter 6: Fine-tuning for Classification

This directory contains the implementation scripts for Chapter 6 of "Build a Large Language Model (From Scratch)" by Manning, focusing on fine-tuning pretrained LLMs for classification tasks.

## Overview

Chapter 6 demonstrates how to adapt a pretrained GPT model for text classification tasks. The concrete example used throughout is spam detection - classifying text messages as "spam" or "not spam". This chapter bridges the gap between general-purpose language modeling and task-specific applications.

## Key Concepts Covered

### Fine-tuning Approaches
- **Classification fine-tuning**: Modifying the model to predict specific class labels
- **Instruction fine-tuning**: Teaching models to follow natural language instructions
- Trade-offs between flexibility and specialization

### Model Modification
- Replacing vocabulary-sized output layer with classification head
- Freezing vs unfreezing pretrained layers
- Parameter-efficient fine-tuning strategies
- Understanding which layers to modify

### Dataset Preparation
- Downloading and preprocessing classification datasets
- Handling class imbalance through sampling strategies
- Creating balanced train/validation/test splits
- Text tokenization and padding for classification

### Training Process
- Adapting pretraining loops for classification objectives
- Cross-entropy loss for discrete classification
- Tracking accuracy alongside loss metrics
- Preventing overfitting with proper evaluation
- Parameter-efficient approach: Only ~7.7% of parameters trainable (last transformer block + classification head)
- Production features: Early stopping, gradient clipping, learning rate scheduling
- Comprehensive evaluation: Accuracy, precision, recall, F1-score, confusion matrices
- Training visualization and model checkpointing for reproducibility

### Evaluation Metrics
- Classification accuracy, precision, and recall
- Confusion matrices and error analysis
- Performance comparison before and after fine-tuning
- Practical deployment considerations

## Scripts

### Educational Implementation Files

- **`spam_dataset_preparation.py`** - Downloads SMS Spam Collection dataset, explores class distributions, creates balanced datasets, and demonstrates train/validation/test splitting strategies
- **`classification_dataset.py`** - Implements PyTorch Dataset and DataLoader for classification, handles text tokenization and padding, and shows batch processing for variable-length sequences
- **`model_modification.py`** - Loads pretrained GPT model, replaces output layer with classification head, demonstrates parameter freezing/unfreezing, and analyzes architectural changes
- **`fine_tuning_strategies.py`** - Compares different fine-tuning approaches (last layer only, last N layers, full fine-tuning), shows training efficiency trade-offs, and provides practical recommendations
- **`classification_training.py`** - Implements production-ready training loop with comprehensive features:
  - Default hyperparameters: 5 epochs, batch_size=8, learning_rate=5e-5
  - Parameter-efficient training: Only last transformer block + classification head (~7.7% of parameters)
  - AdamW optimizer with weight decay (0.01) and cosine annealing scheduler
  - Early stopping (patience=3) and gradient clipping (max_norm=1.0)
  - Model checkpointing, training visualization, and inference demonstration
- **`evaluation_metrics.py`** - Comprehensive classification evaluation beyond accuracy, implements confusion matrices and per-class analysis, and provides error analysis tools
- **`spam_classifier_demo.py`** - End-to-end demonstration from data preparation to deployment, loads pretrained weights and fine-tunes on spam dataset, and provides interactive classification interface
- **`instruction_vs_classification.py`** - Compares instruction fine-tuning vs classification fine-tuning approaches, demonstrates same task with different methods, and discusses use case recommendations

## Key Learning Objectives

### 1. Understanding Fine-tuning Types
Learn when to use classification fine-tuning vs instruction fine-tuning based on:
- Task specificity and requirements
- Available computational resources
- Dataset size and format
- Deployment constraints

### 2. Model Architecture Adaptation
Understand how to:
- Replace language modeling heads with classification layers
- Choose appropriate layers to freeze/unfreeze
- Balance training efficiency with performance
- Preserve pretrained knowledge while learning new tasks

### 3. Dataset Handling for Classification
Master techniques for:
- Preprocessing text classification datasets
- Handling class imbalance effectively
- Creating appropriate data splits
- Tokenization strategies for classification

### 4. Training and Evaluation
Develop skills in:
- Adapting training loops for classification objectives
- Using appropriate loss functions and metrics
- Monitoring training progress and preventing overfitting
- Comprehensive model evaluation and error analysis

## Prerequisites

Before running the fine-tuning scripts, you must:

1. **Install dependencies**: Ensure all dependencies from `pyproject.toml` are installed
2. **Prepare datasets**: Run the dataset preparation script first to download and process the SMS Spam Collection dataset:
   ```bash
   uv run python book/ch6/spam_dataset_preparation.py
   ```
   This creates the required `train.csv`, `validation.csv`, and `test.csv` files.

3. **Hardware requirements**: GPU recommended for training (CPU training will be slow)
4. **Memory requirements**: ~2GB GPU memory for GPT-2 124M training

## Generated Outputs

After running the training scripts, you'll find these generated files:
- `best_spam_classifier.pth` - Saved model checkpoint with best validation accuracy
- `book/ch6/classification_training_progress.png` - Training progress visualization plots
- `train.csv`, `validation.csv`, `test.csv` - Processed datasets (created by spam_dataset_preparation.py)

## Usage Examples

### Basic Fine-tuning Pipeline
```bash
# Load and prepare spam dataset
uv run python book/ch6/spam_dataset_preparation.py

# Create classification datasets and dataloaders
uv run python book/ch6/classification_dataset.py

# Modify pretrained model for classification
uv run python book/ch6/model_modification.py

# Train the classification model
uv run python book/ch6/classification_training.py

# Evaluate model performance
uv run python book/ch6/evaluation_metrics.py
```

### Complete End-to-End Demo
```bash
# Run complete spam classifier demo
uv run python book/ch6/spam_classifier_demo.py
```

### Compare Fine-tuning Strategies
```bash
# Compare different fine-tuning approaches
uv run python book/ch6/fine_tuning_strategies.py

# Compare instruction vs classification fine-tuning
uv run python book/ch6/instruction_vs_classification.py
```

## Dependencies

All required dependencies are already specified in `pyproject.toml`:
- PyTorch - for model training and fine-tuning
- tiktoken - for GPT-2 compatible tokenization
- pandas - for dataset handling and analysis
- urllib, zipfile - for dataset downloading
- matplotlib - for training visualization

## Dataset

The chapter uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository:
- 5,572 SMS messages labeled as "ham" (legitimate) or "spam"
- Real-world text classification challenge
- Demonstrates handling of class imbalance (747 spam, 4,825 ham messages)
- Public domain dataset suitable for educational use

## Model Requirements

The fine-tuning examples use the GPT-2 small (124M parameter) model configuration:
- Loads pretrained weights directly from OpenAI's GPT-2 models
- Architecture: 12 layers, 12 attention heads, 768 embedding dimensions
- Can be trained on modest hardware (GPU recommended but CPU works)
- Parameter-efficient training: Only ~9.5M parameters trainable out of 124M total
- ~500MB model weights + ~2GB GPU memory during training

## Integration with Previous Chapters

Chapter 6 builds directly on previous implementations:
- **Chapter 2**: Uses tokenization infrastructure for text preprocessing
- **Chapter 3**: Leverages attention mechanisms in the pretrained model
- **Chapter 4**: Modifies the complete GPT architecture for classification
- **Chapter 5**: Adapts pretraining loops and weight loading utilities

## Advanced Topics

### Parameter-Efficient Fine-tuning
While the chapter focuses on standard fine-tuning, the implementations provide a foundation for exploring:
- LoRA (Low-Rank Adaptation)
- Prompt tuning and prefix tuning
- Adapter layers

### Multi-class Classification
The spam detection example generalizes to:
- Sentiment analysis (positive/negative/neutral)
- Topic classification (multiple categories)
- Intent detection for chatbots

### Transfer Learning Insights
The chapter demonstrates key transfer learning principles:
- Feature extraction vs fine-tuning trade-offs
- Layer-wise learning rate strategies
- Catastrophic forgetting prevention

## Next Steps

After completing Chapter 6, you'll have:
- A working text classifier built from a pretrained LLM
- Understanding of fine-tuning strategies and trade-offs
- Tools for evaluating classification model performance
- Foundation for building specialized NLP applications

The skills developed here apply to many real-world scenarios:
- Content moderation systems
- Document classification
- Customer support automation
- Sentiment analysis applications

## Exercises and Extensions

The chapter includes several exercises to deepen understanding:
1. **Dataset Analysis**: Explore class distributions and text characteristics
2. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
3. **Architecture Variants**: Try different numbers of unfrozen layers
4. **Multi-class Extension**: Adapt the system for 3+ categories
5. **Deployment Considerations**: Optimize for inference speed and memory

These implementations provide both educational value and practical utility for building production text classification systems.
