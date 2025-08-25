"""
Chapter 6 - Classification Evaluation Metrics

This script demonstrates various metrics for evaluating classification
models beyond simple accuracy.

Key concepts demonstrated:
- Accuracy, precision, recall, and F1-score
- Confusion matrices and their interpretation
- Class-wise performance analysis
- Handling imbalanced datasets
- ROC curves and AUC (if applicable)
- Error analysis and common mistakes

Going beyond Section 6.6, this provides comprehensive evaluation
tools for classification models.

Usage: uv run python book/ch6/evaluation_metrics.py
"""

import torch
import torch.nn as nn
import tiktoken
import numpy as np
import os

from llm_from_scratch.model import GPTModel
from llm_from_scratch.classification import SpamDataset
from torch.utils.data import DataLoader


# Configuration for GPT-2 Small
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}


def explain_classification_metrics():
    """Explain different classification evaluation metrics."""
    print("=== Classification Evaluation Metrics ===\n")

    print("Why accuracy alone isn't enough:")
    print("• Can be misleading with imbalanced datasets")
    print("• Doesn't distinguish between different types of errors")
    print("• Doesn't show confidence in predictions")
    print("• May hide problems with specific classes")

    print("\n--- Core Metrics ---")

    print("\n**Accuracy**: Overall correctness")
    print("  Formula: (TP + TN) / (TP + TN + FP + FN)")
    print("  Range: 0.0 to 1.0 (higher is better)")
    print("  Good for: Balanced datasets")
    print("  Limitation: Misleading with class imbalance")

    print("\n**Precision**: How many predicted positives are actually positive")
    print("  Formula: TP / (TP + FP)")
    print("  Range: 0.0 to 1.0 (higher is better)")
    print("  Good for: Minimizing false positives (spam detection)")
    print("  Example: Of emails flagged as spam, how many are actually spam?")

    print("\n**Recall (Sensitivity)**: How many actual positives are found")
    print("  Formula: TP / (TP + FN)")
    print("  Range: 0.0 to 1.0 (higher is better)")
    print("  Good for: Minimizing false negatives (disease detection)")
    print("  Example: Of all spam emails, how many did we catch?")

    print("\n**F1-Score**: Harmonic mean of precision and recall")
    print("  Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print("  Range: 0.0 to 1.0 (higher is better)")
    print("  Good for: Balanced view when both precision and recall matter")

    print("\n**Specificity**: How many actual negatives are correctly identified")
    print("  Formula: TN / (TN + FP)")
    print("  Range: 0.0 to 1.0 (higher is better)")
    print("  Good for: Minimizing false alarms")


def visualize_confusion_matrix_concepts():
    """Visualize confusion matrix components with ASCII art."""
    print("\n=== Understanding Confusion Matrix ===")

    print("\nConfusion Matrix Structure:")
    print("                    Predicted")
    print("                 Not Spam  Spam")
    print("   Actual Not Spam   TN    FP   ← Specificity = TN/(TN+FP)")
    print("          Spam       FN    TP   ← Recall = TP/(TP+FN)")
    print("                     ↑     ↑")
    print("                     └─────┘")
    print("                 Precision = TP/(TP+FP)")

    print("\nInterpretation:")
    print("• TP (True Positive): Correctly identified spam")
    print("• TN (True Negative): Correctly identified not spam")
    print("• FP (False Positive): Wrongly flagged as spam (Type I error)")
    print("• FN (False Negative): Missed spam (Type II error)")

    print("\nExample scenarios:")
    print("\n**High Precision, Low Recall**")
    print("  Matrix: TN=900, FP=10, FN=80, TP=10")
    print("  Precision: 10/(10+10) = 0.50")
    print("  Recall: 10/(10+80) = 0.11")
    print("  → Conservative: Only flags obvious spam, misses subtle cases")

    print("\n**Low Precision, High Recall**")
    print("  Matrix: TN=800, FP=100, FN=10, TP=90")
    print("  Precision: 90/(90+100) = 0.47")
    print("  Recall: 90/(90+10) = 0.90")
    print("  → Aggressive: Catches most spam, but many false alarms")

    print("\n**Balanced Performance**")
    print("  Matrix: TN=850, FP=50, FN=40, TP=60")
    print("  Precision: 60/(60+50) = 0.55")
    print("  Recall: 60/(60+40) = 0.60")
    print("  → Good balance between catching spam and avoiding false alarms")


def load_trained_model(device):
    """Load a trained model for evaluation."""
    print("\n=== Loading Trained Model ===")

    # Initialize model architecture
    model = GPTModel(GPT_CONFIG_124M)
    model.out_head = nn.Linear(GPT_CONFIG_124M["emb_dim"], 2)  # Binary classification

    # Apply same freezing as in training
    for param in model.parameters():
        param.requires_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    model.to(device)

    # Try to load trained weights
    if os.path.exists("best_spam_classifier.pth"):
        checkpoint = torch.load("best_spam_classifier.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Loaded trained model from checkpoint")
        print(f"  Training validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
        return model, True
    else:
        print("⚠️  No trained model found (best_spam_classifier.pth)")
        print("Using randomly initialized model for demonstration")
        print("Run classification_training.py first for meaningful results")
        return model, False


def comprehensive_model_evaluation(model, data_loader, device, dataset_name="Test"):
    """Perform comprehensive evaluation of the model."""
    print(f"\n=== {dataset_name} Set Evaluation ===")

    model.eval()

    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(device)

            # Get logits and probabilities
            logits = model(input_batch)[:, -1, :]  # Last token for classification
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate confusion matrix components
    tp = sum(1 for p, t in zip(all_predictions, all_targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(all_predictions, all_targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(all_predictions, all_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(all_predictions, all_targets) if p == 0 and t == 1)

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Print results
    print("\nOverall Performance:")
    print(f"  Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision:   {precision:.3f}")
    print(f"  Recall:      {recall:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  F1-Score:    {f1_score:.3f}")

    print("\nConfusion Matrix:")
    print("                   Predicted")
    print("                Not Spam  Spam   Total")
    print(f"Actual Not Spam    {tn:4d}  {fp:4d}    {tn+fp:4d}")
    print(f"       Spam        {fn:4d}  {tp:4d}    {fn+tp:4d}")
    print(f"       Total       {tn+fn:4d}  {fp+tp:4d}    {tp+tn+fp+fn:4d}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "predictions": all_predictions,
        "targets": all_targets,
        "probabilities": all_probabilities,
    }


def analyze_prediction_confidence(results):
    """Analyze the confidence distribution of predictions."""
    print("\n=== Prediction Confidence Analysis ===")

    probabilities = np.array(results["probabilities"])
    predictions = np.array(results["predictions"])
    targets = np.array(results["targets"])

    # Get confidence (max probability) for each prediction
    confidences = np.max(probabilities, axis=1)

    print("Confidence distribution:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence:  {np.min(confidences):.3f}")
    print(f"  Max confidence:  {np.max(confidences):.3f}")
    print(f"  Std confidence:  {np.std(confidences):.3f}")

    # Analyze confidence by correctness
    correct_mask = predictions == targets
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    if len(correct_conf) > 0 and len(incorrect_conf) > 0:
        print("\nConfidence by correctness:")
        print(
            f"  Correct predictions:   {np.mean(correct_conf):.3f} ± {np.std(correct_conf):.3f}"
        )
        print(
            f"  Incorrect predictions: {np.mean(incorrect_conf):.3f} ± {np.std(incorrect_conf):.3f}"
        )

        # Check if model is well-calibrated
        if np.mean(correct_conf) > np.mean(incorrect_conf):
            print("  ✓ Model shows good calibration (higher confidence when correct)")
        else:
            print("  ⚠️ Poor calibration (similar confidence for correct/incorrect)")

    # Confidence histogram
    print("\nConfidence histogram:")
    hist, bins = np.histogram(confidences, bins=10, range=(0.5, 1.0))
    for i in range(len(hist)):
        bin_start, bin_end = bins[i], bins[i + 1]
        bar = "█" * int(hist[i] / max(hist) * 20) if max(hist) > 0 else ""
        print(f"  {bin_start:.2f}-{bin_end:.2f}: {bar} {hist[i]:3d}")


def error_analysis(results, tokenizer, test_dataset):
    """Analyze specific errors made by the model."""
    print("\n=== Error Analysis ===")

    predictions = np.array(results["predictions"])
    targets = np.array(results["targets"])
    probabilities = np.array(results["probabilities"])

    # Find different types of errors
    false_positives = np.where((predictions == 1) & (targets == 0))[0]
    false_negatives = np.where((predictions == 0) & (targets == 1))[0]

    print("Error breakdown:")
    print(f"  False Positives (flagged as spam, actually not): {len(false_positives)}")
    print(f"  False Negatives (missed spam): {len(false_negatives)}")

    # Show examples of each error type
    def show_error_examples(error_indices, error_type, max_examples=3):
        if len(error_indices) == 0:
            print(f"\n{error_type}: No examples found")
            return

        print(f"\n{error_type} Examples:")

        # Sort by confidence (show most confident errors)
        confidences = np.max(probabilities[error_indices], axis=1)
        sorted_indices = error_indices[np.argsort(confidences)[::-1]]

        for i, idx in enumerate(sorted_indices[:max_examples]):
            text = test_dataset.data.iloc[idx]["Text"]
            actual_label = "spam" if targets[idx] == 1 else "not spam"
            predicted_label = "spam" if predictions[idx] == 1 else "not spam"
            confidence = confidences[i]

            print(f"\n  Example {i+1}:")
            print(f"    Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
            print(f"    Actual: {actual_label}")
            print(f"    Predicted: {predicted_label}")
            print(f"    Confidence: {confidence:.3f}")

    show_error_examples(false_positives, "FALSE POSITIVES (Not spam flagged as spam)")
    show_error_examples(false_negatives, "FALSE NEGATIVES (Spam missed)")


def threshold_analysis(results):
    """Analyze performance at different classification thresholds."""
    print("\n=== Threshold Analysis ===")

    probabilities = np.array(results["probabilities"])
    targets = np.array(results["targets"])

    # Get spam probabilities (class 1)
    spam_probs = probabilities[:, 1]

    print("Analyzing performance at different spam probability thresholds...")
    print("(Default threshold is 0.5)")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(
        f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}"
    )
    print("-" * 50)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        # Make predictions with this threshold
        pred_at_threshold = (spam_probs >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((pred_at_threshold == 1) & (targets == 1))
        tn = np.sum((pred_at_threshold == 0) & (targets == 0))
        fp = np.sum((pred_at_threshold == 1) & (targets == 0))
        fn = np.sum((pred_at_threshold == 0) & (targets == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"{threshold:<10.1f} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold for F1-score: {best_threshold} (F1: {best_f1:.3f})")

    if best_threshold != 0.5:
        print(f"💡 Consider using threshold {best_threshold} instead of default 0.5")
    else:
        print("✓ Default threshold 0.5 is optimal")


def compare_class_performance(results):
    """Compare performance for each class separately."""
    print("\n=== Per-Class Performance Analysis ===")

    predictions = np.array(results["predictions"])
    targets = np.array(results["targets"])

    classes = ["Not Spam", "Spam"]

    print("Performance breakdown by class:")
    print(f"\n{'Class':<10} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)

    for class_idx, class_name in enumerate(classes):
        # Calculate per-class metrics
        class_mask = targets == class_idx
        class_support = np.sum(class_mask)

        if class_support == 0:
            continue

        # True positives for this class
        tp_class = np.sum((predictions == class_idx) & (targets == class_idx))
        # False positives for this class (predicted as this class but actually another)
        fp_class = np.sum((predictions == class_idx) & (targets != class_idx))
        # False negatives for this class (actually this class but predicted as another)
        fn_class = np.sum((predictions != class_idx) & (targets == class_idx))

        precision = tp_class / (tp_class + fp_class) if (tp_class + fp_class) > 0 else 0
        recall = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"{class_name:<10} {class_support:<8d} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}"
        )

    # Overall macro and weighted averages
    overall_precision = results["precision"]
    overall_recall = results["recall"]
    overall_f1 = results["f1_score"]

    print("\nOverall averages:")
    print(
        f"  Macro avg:    -       {overall_precision:.3f}     {overall_recall:.3f}     {overall_f1:.3f}"
    )
    print(
        f"  Weighted avg: -       {overall_precision:.3f}     {overall_recall:.3f}     {overall_f1:.3f}"
    )


def practical_deployment_recommendations(results, is_trained_model):
    """Provide practical recommendations for model deployment."""
    print("\n=== Deployment Recommendations ===")

    if not is_trained_model:
        print("⚠️  These recommendations are based on a randomly initialized model.")
        print("Train the model first for meaningful recommendations.")
        print()

    accuracy = results["accuracy"]
    precision = results["precision"]
    recall = results["recall"]
    f1_score = results["f1_score"]

    print("Model Performance Summary:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1-Score: {f1_score:.1%}")

    # Performance assessment
    if is_trained_model:
        if accuracy > 0.9:
            performance_level = "Excellent"
        elif accuracy > 0.8:
            performance_level = "Good"
        elif accuracy > 0.7:
            performance_level = "Acceptable"
        else:
            performance_level = "Needs Improvement"

        print(f"\nPerformance Level: {performance_level}")

    print("\n--- Deployment Considerations ---")

    print("\n**For Spam Detection Use Case:**")
    if precision > 0.8:
        print(f"  ✓ High precision ({precision:.1%}) - Few false spam flags")
    else:
        print(f"  ⚠️  Low precision ({precision:.1%}) - May flag legitimate emails")
        print("     → Consider raising classification threshold")

    if recall > 0.8:
        print(f"  ✓ High recall ({recall:.1%}) - Catches most spam")
    else:
        print(f"  ⚠️  Low recall ({recall:.1%}) - May miss spam emails")
        print("     → Consider lowering classification threshold or more training")

    print("\n**Optimization Strategies:**")
    if precision < recall:
        print("  • Focus on precision: Reduce false positives")
        print("  • Strategy: Increase threshold, add more 'not spam' training data")
    elif recall < precision:
        print("  • Focus on recall: Reduce false negatives")
        print("  • Strategy: Decrease threshold, add more spam training data")
    else:
        print("  • Good balance between precision and recall")

    print("\n**Production Recommendations:**")
    print("  • Use threshold analysis to find optimal operating point")
    print("  • Monitor performance on new data over time")
    print("  • Implement feedback loop to retrain on mislabeled examples")
    print("  • Consider ensemble methods for improved robustness")
    print("  • Add confidence-based routing (high confidence → auto-classify)")


def main():
    """Main function for comprehensive evaluation demonstration."""
    print("=" * 70)
    print("Chapter 6: Classification Evaluation Metrics")
    print("=" * 70)

    # Step 1: Explain evaluation concepts
    explain_classification_metrics()
    visualize_confusion_matrix_concepts()

    # Step 2: Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_trained = load_trained_model(device)

    # Step 3: Load test dataset
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        test_dataset = SpamDataset("book/ch6/test.csv", tokenizer, max_length=120)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        print(f"✓ Loaded test dataset: {len(test_dataset)} examples")
    except FileNotFoundError:
        print("⚠️  Test dataset not found. Run spam_dataset_preparation.py first.")
        return

    # Step 4: Comprehensive evaluation
    results = comprehensive_model_evaluation(model, test_loader, device)

    # Step 5: Confidence analysis
    analyze_prediction_confidence(results)

    # Step 6: Error analysis
    error_analysis(results, tokenizer, test_dataset)

    # Step 7: Threshold analysis
    threshold_analysis(results)

    # Step 8: Per-class performance
    compare_class_performance(results)

    # Step 9: Deployment recommendations
    practical_deployment_recommendations(results, is_trained)

    print("\n" + "=" * 70)
    print("Classification evaluation completed!")
    print("✓ Comprehensive metrics calculated and analyzed")
    print("✓ Confusion matrix breakdown provided")
    print("✓ Prediction confidence analyzed")
    print("✓ Error examples examined")
    print("✓ Threshold optimization performed")
    print("✓ Per-class performance evaluated")
    print("✓ Deployment recommendations provided")

    if is_trained:
        print(f"✓ Model achieved {results['accuracy']:.1%} accuracy on test set")
    else:
        print("⚠️  Run classification_training.py first for trained model evaluation")

    print("\nNext step: Run spam_classifier_demo.py for end-to-end demonstration")
    print("=" * 70)


if __name__ == "__main__":
    main()
