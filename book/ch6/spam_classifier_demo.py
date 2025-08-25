"""
Chapter 6 - Complete Spam Classifier Demo

This script ties everything together into a working spam classifier,
from data preparation through deployment.

Key concepts demonstrated:
- End-to-end pipeline from raw text to predictions
- Loading saved fine-tuned models
- Classifying new, unseen messages
- Confidence scores and thresholds
- Practical deployment considerations
- Interactive classification demo

This follows Section 6.8, showing how to use the fine-tuned model
in practice for classifying new messages.

Usage: uv run python book/ch6/spam_classifier_demo.py
"""

import torch
import torch.nn as nn
import tiktoken
import os

from llm_from_scratch.model import GPTModel
from llm_from_scratch.utils import classify_text_simple


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


class SpamClassifier:
    """
    Complete spam classification system.

    This class encapsulates the entire spam detection pipeline,
    providing a clean interface for classifying text messages.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize the spam classifier.

        Args:
            model_path: Path to saved model weights
            device: Device to run inference on
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_length = 120
        self.pad_token_id = 50256
        self.class_names = ["not spam", "spam"]
        self.model = None
        self.is_trained = False

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load and setup the classification model."""
        # Initialize model architecture
        self.model = GPTModel(GPT_CONFIG_124M)
        self.model.out_head = nn.Linear(GPT_CONFIG_124M["emb_dim"], 2)

        # Apply same layer freezing as used in training
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in self.model.final_norm.parameters():
            param.requires_grad = True
        for param in self.model.out_head.parameters():
            param.requires_grad = True

        self.model.to(self.device)

        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.is_trained = True
            print(f"✓ Loaded trained model from {model_path}")
        elif os.path.exists("best_spam_classifier.pth"):
            checkpoint = torch.load(
                "best_spam_classifier.pth", map_location=self.device
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.is_trained = True
            print("✓ Loaded trained model from best_spam_classifier.pth")
        else:
            print("⚠️  No trained model found - using random weights")
            print("Run classification_training.py first for meaningful results")

    def classify(self, text, return_probabilities=False):
        """
        Classify a single text message.

        Args:
            text: Text message to classify
            return_probabilities: Whether to return class probabilities

        Returns:
            dict with classification results
        """
        result = classify_text_simple(
            text,
            self.model,
            self.tokenizer,
            self.device,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            class_names=self.class_names,
        )

        if not return_probabilities:
            # Remove probabilities for simpler output
            result = {k: v for k, v in result.items() if k != "probabilities"}

        return result

    def classify_batch(self, texts):
        """
        Classify multiple texts at once.

        Args:
            texts: List of text messages

        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results

    def get_confidence_explanation(self, confidence):
        """Get human-readable explanation of confidence level."""
        if confidence > 0.9:
            return "Very high confidence"
        elif confidence > 0.8:
            return "High confidence"
        elif confidence > 0.7:
            return "Moderate confidence"
        elif confidence > 0.6:
            return "Low confidence"
        else:
            return "Very low confidence"


def demonstrate_classifier_capabilities():
    """Demonstrate the complete spam classifier capabilities."""
    print("=== Spam Classifier Capabilities Demo ===\n")

    print("This demo shows a complete end-to-end spam classification system:")
    print("• Loads a trained GPT model fine-tuned for spam detection")
    print("• Classifies text messages with confidence scores")
    print("• Provides explanations and recommendations")
    print("• Suitable for production deployment")

    print("\nInitializing classifier...")
    classifier = SpamClassifier()

    print("✓ Classifier ready")
    print(f"  Device: {classifier.device}")
    print(f"  Model trained: {classifier.is_trained}")
    print(f"  Max sequence length: {classifier.max_length} tokens")

    return classifier


def test_example_messages(classifier):
    """Test the classifier on a variety of example messages."""
    print("\n=== Testing Example Messages ===")

    # Curated test messages covering different scenarios
    test_messages = [
        # Clear not spam examples
        ("Hey, are we still meeting for lunch tomorrow?", "not spam"),
        ("Thanks for the meeting notes. I'll review them tonight.", "not spam"),
        ("Can you pick up milk on your way home?", "not spam"),
        ("Happy birthday! Hope you have a great day.", "not spam"),
        # Clear spam examples
        ("URGENT! You've won £5000! Call now to claim your prize!", "spam"),
        ("FREE MONEY! Limited time offer! Click here now!", "spam"),
        ("WINNER! Claim your cash prize by calling 0800-CASH-NOW!", "spam"),
        ("Congratulations! You have been selected for a special offer!", "spam"),
        # Borderline cases
        ("Sale: 50% off all items this weekend only!", "borderline"),
        ("Your account will be closed unless you verify immediately", "borderline"),
        ("Reminder: Your subscription expires tomorrow", "borderline"),
        ("You have 1 new voicemail. Press 1 to listen.", "borderline"),
    ]

    print(f"Testing {len(test_messages)} messages...")

    correct_predictions = 0
    total_clear_cases = 0

    for i, (message, expected_type) in enumerate(test_messages, 1):
        result = classifier.classify(message, return_probabilities=True)

        prediction = result["predicted_label"]
        confidence = result["confidence"]
        spam_prob = result["probabilities"][1]

        # Check correctness for clear cases
        is_correct = "?"
        if expected_type in ["not spam", "spam"]:
            is_correct = "✓" if prediction == expected_type else "✗"
            total_clear_cases += 1
            if prediction == expected_type:
                correct_predictions += 1

        confidence_desc = classifier.get_confidence_explanation(confidence)

        print(f"\n{i:2d}. \"{message[:60]}{'...' if len(message) > 60 else ''}\"")
        print(
            f"     Prediction: {prediction.upper()} ({confidence:.3f} - {confidence_desc})"
        )
        print(f"     Spam probability: {spam_prob:.3f}")
        print(f"     Expected: {expected_type} {is_correct}")

    if total_clear_cases > 0:
        accuracy = correct_predictions / total_clear_cases
        print(
            f"\nAccuracy on clear cases: {accuracy:.1%} ({correct_predictions}/{total_clear_cases})"
        )

        if not classifier.is_trained:
            print("⚠️  Model is not trained - random performance expected")
        elif accuracy > 0.8:
            print("✓ Good performance on test cases")
        else:
            print("⚠️  Consider retraining or adjusting threshold")


def interactive_classifier_demo(classifier):
    """Interactive demo allowing user to input their own messages."""
    print("\n=== Interactive Spam Classifier Demo ===")

    print("Type messages to classify them as spam or not spam.")
    print("Type 'quit' to exit, 'help' for commands.")

    if not classifier.is_trained:
        print("\n⚠️  WARNING: Model is not trained - results will be random!")
        print(
            "Run 'python book/ch6/classification_training.py' first for meaningful results."
        )

    print()

    while True:
        try:
            # Get user input
            message = input("Enter message to classify: ").strip()

            if not message:
                continue
            elif message.lower() == "quit":
                print("Goodbye!")
                break
            elif message.lower() == "help":
                print("\nCommands:")
                print("  quit  - Exit the demo")
                print("  help  - Show this help")
                print("  <message> - Classify the message")
                continue

            # Classify the message
            result = classifier.classify(message, return_probabilities=True)

            prediction = result["predicted_label"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            confidence_desc = classifier.get_confidence_explanation(confidence)

            # Display results with nice formatting
            print()
            print("─" * 50)
            print(f'📧 Message: "{message}"')
            print(f"🤖 Prediction: {prediction.upper()}")
            print(f"📊 Confidence: {confidence:.3f} ({confidence_desc})")
            print("📈 Probabilities:")
            print(f"   Not Spam: {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)")
            print(f"   Spam:     {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)")

            # Provide recommendations
            if confidence < 0.7:
                print("💡 Low confidence - consider manual review")
            elif prediction == "spam" and confidence > 0.9:
                print("🚫 High confidence spam - recommend blocking")
            elif prediction == "not spam" and confidence > 0.9:
                print("✅ High confidence legitimate - safe to deliver")

            print("─" * 50)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_classification_demo(classifier):
    """Demonstrate batch classification capabilities."""
    print("\n=== Batch Classification Demo ===")

    # Simulate a batch of messages that might arrive in a real system
    message_batch = [
        "Meeting rescheduled to 3pm tomorrow",
        "WIN BIG! Call now for your FREE prize!",
        "Grocery list: milk, bread, eggs",
        "URGENT: Account suspended! Click to verify",
        "Thanks for dinner, had a great time!",
        "Exclusive offer just for you! Act now!",
        "Can you send me the report when ready?",
        "CONGRATULATIONS! You're our lucky winner!",
    ]

    print(f"Processing batch of {len(message_batch)} messages...")

    # Classify all messages
    results = classifier.classify_batch(message_batch)

    # Display results in a table format
    print("\nBatch Classification Results:")
    print(f"{'#':<2} {'Message':<40} {'Prediction':<10} {'Confidence':<10}")
    print("─" * 65)

    spam_count = 0
    high_confidence_count = 0

    for i, (message, result) in enumerate(zip(message_batch, results), 1):
        prediction = result["predicted_label"]
        confidence = result["confidence"]

        # Truncate long messages
        display_msg = message[:37] + "..." if len(message) > 40 else message

        print(f"{i:<2} {display_msg:<40} {prediction:<10} {confidence:.3f}")

        if prediction == "spam":
            spam_count += 1
        if confidence > 0.8:
            high_confidence_count += 1

    print("─" * 65)
    print(f"Summary: {spam_count}/{len(message_batch)} classified as spam")
    print(
        f"         {high_confidence_count}/{len(message_batch)} high confidence predictions"
    )

    # Production recommendations
    print("\nProduction insights:")
    spam_rate = spam_count / len(message_batch)
    if spam_rate > 0.5:
        print(f"⚠️  High spam rate ({spam_rate:.1%}) - check message sources")
    else:
        print(f"✓ Normal spam rate ({spam_rate:.1%})")

    confidence_rate = high_confidence_count / len(message_batch)
    if confidence_rate > 0.8:
        print(f"✓ Most predictions high confidence ({confidence_rate:.1%})")
    else:
        print(
            f"⚠️  Many low confidence predictions ({confidence_rate:.1%}) - consider retraining"
        )


def performance_analysis(classifier):
    """Analyze the performance characteristics of the classifier."""
    print("\n=== Performance Analysis ===")

    print("Analyzing classifier performance characteristics...")

    # Test inference speed
    import time

    test_message = "This is a test message to measure inference speed."

    # Warm up
    for _ in range(5):
        classifier.classify(test_message)

    # Measure inference time
    num_tests = 50
    start_time = time.time()

    for _ in range(num_tests):
        classifier.classify(test_message)

    end_time = time.time()
    avg_time = (end_time - start_time) / num_tests
    throughput = 1.0 / avg_time

    print("\nInference Performance:")
    print(f"  Average time per message: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} messages/second")

    # Memory usage estimation
    param_count = sum(p.numel() for p in classifier.model.parameters())
    trainable_params = sum(
        p.numel() for p in classifier.model.parameters() if p.requires_grad
    )

    # Rough memory estimate (4 bytes per float32 parameter)
    model_memory_mb = (param_count * 4) / (1024 * 1024)

    print("\nModel Characteristics:")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model memory (approx): {model_memory_mb:.1f} MB")
    print(f"  Device: {classifier.device}")

    # Production readiness assessment
    print("\nProduction Readiness:")
    if avg_time < 0.1:  # 100ms
        print(f"  ✓ Fast inference ({avg_time*1000:.1f}ms) - suitable for real-time")
    elif avg_time < 1.0:  # 1 second
        print(
            f"  ✓ Moderate inference ({avg_time*1000:.1f}ms) - suitable for batch processing"
        )
    else:
        print(f"  ⚠️  Slow inference ({avg_time:.2f}s) - consider optimization")

    if model_memory_mb < 1000:  # 1GB
        print(f"  ✓ Reasonable memory usage ({model_memory_mb:.1f}MB)")
    else:
        print(
            f"  ⚠️  High memory usage ({model_memory_mb:.1f}MB) - may need larger instances"
        )


def deployment_guidelines():
    """Provide practical deployment guidelines."""
    print("\n=== Deployment Guidelines ===")

    print("Guidelines for deploying this spam classifier in production:")

    print("\n**1. Model Loading and Caching**")
    print("   • Load model once at startup, reuse for all requests")
    print("   • Cache tokenizer to avoid reinitialization")
    print("   • Consider model quantization for reduced memory usage")

    print("\n**2. Input Processing**")
    print("   • Validate input text (length limits, encoding)")
    print("   • Sanitize inputs to prevent injection attacks")
    print("   • Handle empty or malformed messages gracefully")

    print("\n**3. Confidence Thresholds**")
    print("   • Use different thresholds for different use cases:")
    print("     - High precision: threshold > 0.8 (fewer false positives)")
    print("     - High recall: threshold > 0.3 (catch more spam)")
    print("     - Balanced: threshold > 0.5 (default)")

    print("\n**4. Monitoring and Maintenance**")
    print("   • Log predictions and confidence scores")
    print("   • Monitor false positive/negative rates")
    print("   • Implement feedback mechanism for model improvement")
    print("   • Plan for periodic retraining with new data")

    print("\n**5. Scalability Considerations**")
    print("   • Use GPU for higher throughput if available")
    print("   • Implement batch processing for efficiency")
    print("   • Consider async processing for non-critical paths")
    print("   • Load balance across multiple model instances")

    print("\n**6. Fallback Strategies**")
    print("   • Have rule-based backup system for model failures")
    print("   • Route low-confidence predictions to human review")
    print("   • Implement circuit breakers for error handling")

    print("\n**Example Integration Code:**")
    print("```python")
    print("class ProductionSpamFilter:")
    print("    def __init__(self):")
    print("        self.classifier = SpamClassifier('best_model.pth')")
    print("        self.spam_threshold = 0.7  # Adjustable")
    print("    ")
    print("    def filter_message(self, message):")
    print("        result = self.classifier.classify(message)")
    print("        is_spam = (result['predicted_class'] == 1 and ")
    print("                  result['confidence'] > self.spam_threshold)")
    print("        return {'is_spam': is_spam, 'confidence': result['confidence']}")
    print("```")


def main():
    """Main function for the complete spam classifier demo."""
    print("=" * 70)
    print("Chapter 6: Complete Spam Classifier Demo")
    print("=" * 70)

    # Step 1: Initialize and demonstrate classifier
    classifier = demonstrate_classifier_capabilities()

    # Step 2: Test on example messages
    test_example_messages(classifier)

    # Step 3: Demonstrate batch processing
    batch_classification_demo(classifier)

    # Step 4: Performance analysis
    performance_analysis(classifier)

    # Step 5: Deployment guidelines
    deployment_guidelines()

    # Step 6: Interactive demo (optional)
    print("\n" + "=" * 70)
    print("Demo completed! Summary of capabilities:")
    print("✓ End-to-end spam classification pipeline")
    print("✓ Single message and batch processing")
    print("✓ Confidence scores and explanations")
    print("✓ Performance analysis and optimization")
    print("✓ Production deployment guidelines")

    if classifier.is_trained:
        print("✓ Using trained model - results are meaningful")
    else:
        print("⚠️  Using untrained model - run classification_training.py first")

    # Offer interactive mode
    print("\nWould you like to try the interactive classifier? (y/n)")
    try:
        response = input().strip().lower()
        if response in ["y", "yes"]:
            interactive_classifier_demo(classifier)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")

    print("\nNext step: Run instruction_vs_classification.py to compare approaches")
    print("=" * 70)


if __name__ == "__main__":
    main()
