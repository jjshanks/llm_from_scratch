"""
Chapter 6 - Instruction vs Classification Fine-tuning

This script compares the two main fine-tuning approaches: instruction
fine-tuning and classification fine-tuning.

Key concepts demonstrated:
- Architectural differences between approaches
- Dataset format requirements
- Use case comparisons
- Performance and efficiency trade-offs
- When to use each approach
- Limitations and capabilities

Based on Section 6.1, this provides a practical comparison to help
choose the right fine-tuning approach.

Usage: uv run python book/ch6/instruction_vs_classification.py
"""

import torch
import torch.nn as nn
import tiktoken
from llm_from_scratch.model import GPTModel
from llm_from_scratch.utils import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


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


def explain_fine_tuning_paradigms():
    """Explain the two main fine-tuning approaches."""
    print("=== Fine-tuning Paradigms Overview ===\n")

    print("Two main approaches to fine-tune language models:")

    print("\n**1. Classification Fine-tuning**")
    print("• Goal: Train model to output specific class labels")
    print("• Architecture: Replace vocab head with classification head")
    print("• Training: Supervised learning on labeled examples")
    print("• Output: Discrete class predictions (e.g., spam/not spam)")
    print("• Use case: When you have specific, well-defined categories")

    print("\n**2. Instruction Fine-tuning**")
    print("• Goal: Train model to follow natural language instructions")
    print("• Architecture: Keep original language modeling head")
    print("• Training: Generate appropriate responses to instructions")
    print("• Output: Natural language text responses")
    print("• Use case: When you want flexible, conversational AI")

    print("\nKey distinction:")
    print("• Classification: Specialized for one specific task")
    print("• Instruction: Generalist that can handle various tasks")


def compare_architectures():
    """Compare the architectural differences between the approaches."""
    print("\n=== Architectural Comparison ===")

    print("\nCLASSIFICATION FINE-TUNING:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│                Input Text                           │")
    print("│              [batch, seq_len]                       │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│              GPT Backbone                           │")
    print("│           (mostly frozen)                           │")
    print("│         [batch, seq_len, emb_dim]                   │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("                  ▼ (take last token)")
    print("┌─────────────────────────────────────────────────────┐")
    print("│          Classification Head                        │")
    print("│           [emb_dim] → [num_classes]                 │")
    print("│            Output: Class logits                     │")
    print("└─────────────────────────────────────────────────────┘")

    print("\nINSTRUCTION FINE-TUNING:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│            Instruction + Context                    │")
    print("│              [batch, seq_len]                       │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("┌─────────────────▼───────────────────────────────────┐")
    print("│              GPT Backbone                           │")
    print("│           (partially frozen)                        │")
    print("│         [batch, seq_len, emb_dim]                   │")
    print("└─────────────────┬───────────────────────────────────┘")
    print("                  │")
    print("                  ▼ (all positions)")
    print("┌─────────────────────────────────────────────────────┐")
    print("│          Language Modeling Head                     │")
    print("│          [emb_dim] → [vocab_size]                   │")
    print("│         Output: Next token logits                   │")
    print("└─────────────────────────────────────────────────────┘")


def demonstrate_classification_approach():
    """Demonstrate classification fine-tuning approach."""
    print("\n=== Classification Approach Demonstration ===")

    print("Setting up classification model...")

    # Initialize model for classification
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.out_head = nn.Linear(GPT_CONFIG_124M["emb_dim"], 2)  # Binary classification

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("✓ Classification model ready")
    print(f"  Output layer: {GPT_CONFIG_124M['emb_dim']} → 2 classes")

    # Example classification usage
    test_messages = [
        "Hey, can we meet for coffee tomorrow?",
        "URGENT! You've won £1000! Call now!",
        "Thanks for the presentation today.",
    ]

    print("\nClassification predictions (random weights):")

    model.eval()
    with torch.no_grad():
        for i, message in enumerate(test_messages, 1):
            # Tokenize and pad
            tokens = tokenizer.encode(message)
            if len(tokens) > 50:
                tokens = tokens[:50]

            input_tensor = torch.tensor(tokens, device=device).unsqueeze(0)

            # Get classification prediction
            logits = model(input_tensor)[:, -1, :]  # Last token only
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, prediction].item()

            class_name = "not spam" if prediction == 0 else "spam"

            print(f'{i}. "{message}"')
            print(f"   → {class_name.upper()} (confidence: {confidence:.3f})")

    print("\nClassification characteristics:")
    print("• Input: Raw text message")
    print("• Output: Single class label per message")
    print("• Training: Supervised learning on labeled data")
    print("• Scalability: Fast inference, specialized performance")

    return model


def demonstrate_instruction_approach():
    """Demonstrate instruction fine-tuning approach."""
    print("\n=== Instruction Approach Demonstration ===")

    print("Setting up instruction model...")

    # Initialize model for instruction following
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    # Keep original vocabulary-sized output layer

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("✓ Instruction model ready")
    print(
        f"  Output layer: {GPT_CONFIG_124M['emb_dim']} → {GPT_CONFIG_124M['vocab_size']} tokens"
    )

    # Example instruction usage
    instruction_prompts = [
        "Is the following text spam? Answer with 'yes' or 'no': 'Hey, can we meet for coffee tomorrow?'",
        "Is the following text spam? Answer with 'yes' or 'no': 'URGENT! You've won £1000! Call now!'",
        "Is the following text spam? Answer with 'yes' or 'no': 'Thanks for the presentation today.'",
    ]

    print("\nInstruction-based predictions (random weights):")

    model.eval()
    for i, prompt in enumerate(instruction_prompts, 1):
        # Generate response to instruction
        input_ids = text_to_token_ids(prompt, tokenizer).to(device)

        # Generate a short response
        with torch.no_grad():
            response_ids = generate_text_simple(
                model,
                input_ids,
                max_new_tokens=5,
                context_size=GPT_CONFIG_124M["context_length"],
            )

        # Extract just the generated part
        generated_part = response_ids[0, input_ids.shape[1] :]
        response = token_ids_to_text(generated_part.unsqueeze(0), tokenizer)

        # Extract the original message for display
        message_start = prompt.find("'") + 1
        message_end = prompt.rfind("'")
        message = prompt[message_start:message_end]

        print(f'{i}. "{message}"')
        print('   Instruction: "Is this spam? Answer yes/no"')
        print(f'   → Response: "{response.strip()}"')

    print("\nInstruction characteristics:")
    print("• Input: Instruction + context in natural language")
    print("• Output: Generated text response")
    print("• Training: Learn to follow diverse instructions")
    print("• Scalability: Flexible but slower, general-purpose")

    return model


def compare_dataset_formats():
    """Compare the dataset formats for each approach."""
    print("\n=== Dataset Format Comparison ===")

    print("CLASSIFICATION DATASET FORMAT:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│  Text                          │  Label             │")
    print("├─────────────────────────────────────────────────────┤")
    print('│  "Hey, let\'s meet for coffee"   │  0 (not spam)      │')
    print('│  "Win £1000! Call now!"         │  1 (spam)          │')
    print('│  "Thanks for the notes"         │  0 (not spam)      │')
    print('│  "URGENT! Act now!"             │  1 (spam)          │')
    print("└─────────────────────────────────────────────────────┘")
    print("• Format: (input_text, class_label)")
    print("• Simple tabular structure")
    print("• Easy to collect and annotate")
    print("• Minimal preprocessing needed")

    print("\nINSTRUCTION DATASET FORMAT:")
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│  Instruction + Input                    │  Expected Output              │")
    print("├─────────────────────────────────────────────────────────────────────────┤")
    print(
        '│  "Is this spam? \'Hey, let\'s meet\'"      │  "No, this is not spam."     │'
    )
    print('│  "Is this spam? \'Win £1000!\'"          │  "Yes, this is spam."        │')
    print('│  "Classify: \'Thanks for notes\'"        │  "Not spam"                  │')
    print('│  "Spam detection: \'URGENT!\'"           │  "This appears to be spam."  │')
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print("• Format: (instruction + context, natural_language_response)")
    print("• More complex conversational structure")
    print("• Requires careful prompt engineering")
    print("• Higher annotation cost")

    print("\nDataset creation effort:")
    print("• Classification: Collect texts + assign labels (moderate effort)")
    print("• Instruction: Create diverse instructions + write responses (high effort)")


def analyze_performance_characteristics():
    """Analyze and compare performance characteristics."""
    print("\n=== Performance Analysis ===")

    print("INFERENCE SPEED:")
    print("┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│      Aspect         │   Classification    │    Instruction      │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Forward pass        │ Single pass         │ Multiple passes     │")
    print("│ Output processing   │ Argmax (fast)       │ Text generation     │")
    print("│ Tokens generated    │ 0 (classification)  │ 5-50+ tokens        │")
    print("│ Typical speed       │ 10-100ms           │ 100-1000ms         │")
    print("│ Batch processing    │ Highly efficient    │ Less efficient      │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")

    print("\nMEMORY REQUIREMENTS:")
    print("┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│      Aspect         │   Classification    │    Instruction      │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Model parameters    │ Reduced (new head)  │ Full model          │")
    print("│ Output layer size   │ Small (num_classes) │ Large (vocab_size)  │")
    print("│ Generation cache    │ Not needed          │ Required for speed  │")
    print("│ Memory efficiency   │ High                │ Moderate            │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")

    print("\nTRAINING REQUIREMENTS:")
    print("┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│      Aspect         │   Classification    │    Instruction      │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Dataset size        │ 1K-10K examples     │ 10K-100K+ examples  │")
    print("│ Data complexity     │ Simple labels       │ Rich responses      │")
    print("│ Training time       │ Hours to days       │ Days to weeks       │")
    print("│ Compute resources   │ Moderate            │ High               │")
    print("│ Overfitting risk    │ Higher (small head) │ Lower (full model)  │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")


def discuss_use_case_scenarios():
    """Discuss when to use each approach."""
    print("\n=== Use Case Decision Guide ===")

    scenarios = [
        {
            "scenario": "Email Spam Detection",
            "classification": "✓ IDEAL",
            "instruction": "⚠️ OVERKILL",
            "reason": "Binary classification, speed critical, simple task",
        },
        {
            "scenario": "Content Moderation",
            "classification": "✓ GOOD",
            "instruction": "⚠️ COMPLEX",
            "reason": "Multiple categories, consistent rules needed",
        },
        {
            "scenario": "Sentiment Analysis",
            "classification": "✓ EFFICIENT",
            "instruction": "⚠️ SLOW",
            "reason": "Clear categories (pos/neg/neutral), high throughput",
        },
        {
            "scenario": "General Purpose Chatbot",
            "classification": "✗ INSUFFICIENT",
            "instruction": "✓ PERFECT",
            "reason": "Need flexible responses, various tasks",
        },
        {
            "scenario": "Document Q&A Assistant",
            "classification": "✗ INAPPROPRIATE",
            "instruction": "✓ NECESSARY",
            "reason": "Open-ended responses, context understanding",
        },
        {
            "scenario": "Code Generation Helper",
            "classification": "✗ WRONG TOOL",
            "instruction": "✓ ESSENTIAL",
            "reason": "Creative generation, following complex instructions",
        },
        {
            "scenario": "Simple Intent Classification",
            "classification": "✓ OPTIMAL",
            "instruction": "⚠️ WASTEFUL",
            "reason": "Few predefined intents, speed matters",
        },
        {
            "scenario": "Medical Diagnosis Support",
            "classification": "✓ APPROPRIATE",
            "instruction": "⚠️ RISKY",
            "reason": "Specific categories, consistency critical",
        },
    ]

    print("Decision matrix for common use cases:")
    print(
        "\n┌─────────────────────────┬─────────────────┬─────────────────┬──────────────────────────┐"
    )
    print(
        "│      Use Case           │ Classification  │  Instruction    │        Reasoning         │"
    )
    print(
        "├─────────────────────────┼─────────────────┼─────────────────┼──────────────────────────┤"
    )

    for scenario_data in scenarios:
        scenario = scenario_data["scenario"]
        classification = scenario_data["classification"]
        instruction = scenario_data["instruction"]
        reason = scenario_data["reason"]

        print(
            f"│ {scenario:<23} │ {classification:<15} │ {instruction:<15} │ {reason:<24} │"
        )

    print(
        "└─────────────────────────┴─────────────────┴─────────────────┴──────────────────────────┘"
    )

    print("\nLegend:")
    print("  ✓ IDEAL/PERFECT - Best approach for this use case")
    print("  ✓ GOOD/EFFICIENT - Works well, recommended")
    print("  ⚠️ OVERKILL/COMPLEX - Works but unnecessarily complicated")
    print("  ⚠️ SLOW/WASTEFUL - Works but inefficient")
    print("  ✗ INSUFFICIENT/WRONG - Not suitable for this use case")


def provide_selection_framework():
    """Provide a decision framework for choosing approaches."""
    print("\n=== Selection Framework ===")

    print("Use this flowchart to choose the right approach:")

    print("\n                    Start Here")
    print("                        │")
    print("                        ▼")
    print("              ┌─────────────────────┐")
    print("              │ Do you need the     │")
    print("              │ model to generate   │")
    print("              │ flexible text       │")
    print("              │ responses?          │")
    print("              └─────────┬───────────┘")
    print("                        │")
    print("            ┌───────────┴───────────┐")
    print("           YES                     NO")
    print("            │                       │")
    print("            ▼                       ▼")
    print("    ┌───────────────┐      ┌────────────────┐")
    print("    │ INSTRUCTION   │      │ Are there      │")
    print("    │ FINE-TUNING   │      │ predefined     │")
    print("    │               │      │ categories?    │")
    print("    │ Use when:     │      └────────┬───────┘")
    print("    │ • Chat bots   │               │")
    print("    │ • Q&A systems │   ┌───────────┴───────────┐")
    print("    │ • Assistants  │  YES                     NO")
    print("    │ • Creative AI │   │                       │")
    print("    └───────────────┘   ▼                       ▼")
    print("                ┌──────────────┐        ┌──────────────┐")
    print("                │CLASSIFICATION│        │ Consider     │")
    print("                │ FINE-TUNING  │        │ unsupervised │")
    print("                │              │        │ or clustering│")
    print("                │ Use when:    │        │ approaches   │")
    print("                │ • Spam       │        └──────────────┘")
    print("                │ • Sentiment  │")
    print("                │ • Categories │")
    print("                │ • Intent     │")
    print("                └──────────────┘")

    print("\nAdditional considerations:")

    print("\n**Choose Classification if:**")
    print("• You have well-defined categories")
    print("• Speed is critical (real-time systems)")
    print("• You need consistent, predictable outputs")
    print("• Dataset is smaller (1K-10K examples)")
    print("• Task is narrow and specific")

    print("\n**Choose Instruction if:**")
    print("• You need flexible, natural responses")
    print("• Task variety is high")
    print("• User interaction is conversational")
    print("• You have large, diverse datasets (10K+ examples)")
    print("• Creative or generative outputs are needed")

    print("\n**Hybrid Approach:**")
    print("• Use classification for routing/filtering")
    print("• Use instruction fine-tuning for response generation")
    print("• Example: Classify intent first, then generate appropriate response")


def demonstrate_hybrid_approach():
    """Demonstrate a hybrid approach combining both methods."""
    print("\n=== Hybrid Approach Demonstration ===")

    print("Real-world systems often combine both approaches:")

    print("\nExample: Smart Email Assistant")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                    Incoming Email                       │")
    print("└─────────────────┬───────────────────────────────────────┘")
    print("                  │")
    print("                  ▼")
    print("         ┌─────────────────┐")
    print("         │ STEP 1:         │")
    print("         │ Classification  │")
    print("         │ Fine-tuned GPT  │")
    print("         │                 │")
    print("         │ Categories:     │")
    print("         │ • Spam          │")
    print("         │ • Important     │")
    print("         │ • Newsletter    │")
    print("         │ • Personal      │")
    print("         └─────────┬───────┘")
    print("                   │")
    print("                   ▼")
    print("          ┌─────────────────┐")
    print("          │ STEP 2:         │")
    print("          │ Route based on  │")
    print("          │ classification  │")
    print("          └─────────┬───────┘")
    print("                    │")
    print("        ┌───────────┼───────────┐")
    print("        │           │           │")
    print("        ▼           ▼           ▼")
    print("   ┌─────────┐ ┌─────────┐ ┌─────────┐")
    print("   │  Spam   │ │Personal │ │Important│")
    print("   │ → Block │ │→ STEP 3 │ │→ STEP 3 │")
    print("   └─────────┘ └─────────┘ └─────────┘")
    print("                    │           │")
    print("                    ▼           ▼")
    print("            ┌─────────────────────────┐")
    print("            │ STEP 3:                 │")
    print("            │ Instruction Fine-tuned  │")
    print("            │ Response Generator      │")
    print("            │                         │")
    print("            │ Generate:               │")
    print("            │ • Smart replies         │")
    print("            │ • Summaries            │")
    print("            │ • Action suggestions   │")
    print("            └─────────────────────────┘")

    print("\nBenefits of hybrid approach:")
    print("• Fast filtering with classification (low latency)")
    print("• Rich responses with instruction tuning (high quality)")
    print("• Resource efficiency (only generate when needed)")
    print("• Scalable to different use cases")

    print("\nImplementation strategy:")
    print("1. Train classification model for routing/filtering")
    print("2. Train instruction model for response generation")
    print("3. Use classification confidence to decide when to generate")
    print("4. Cache common responses for efficiency")


def main():
    """Main function comparing instruction vs classification fine-tuning."""
    print("=" * 70)
    print("Chapter 6: Instruction vs Classification Fine-tuning")
    print("=" * 70)

    # Step 1: Explain the paradigms
    explain_fine_tuning_paradigms()

    # Step 2: Compare architectures
    compare_architectures()

    # Step 3: Demonstrate classification approach
    demonstrate_classification_approach()

    # Step 4: Demonstrate instruction approach
    demonstrate_instruction_approach()

    # Step 5: Compare dataset formats
    compare_dataset_formats()

    # Step 6: Analyze performance
    analyze_performance_characteristics()

    # Step 7: Discuss use cases
    discuss_use_case_scenarios()

    # Step 8: Provide selection framework
    provide_selection_framework()

    # Step 9: Show hybrid approach
    demonstrate_hybrid_approach()

    print("\n" + "=" * 70)
    print("Instruction vs Classification comparison completed!")
    print("✓ Both approaches explained and demonstrated")
    print("✓ Architectural differences highlighted")
    print("✓ Performance characteristics analyzed")
    print("✓ Use case decision matrix provided")
    print("✓ Selection framework established")
    print("✓ Hybrid approach strategies discussed")
    print("\n🎯 Key takeaway:")
    print("   • Classification: Fast, specialized, limited scope")
    print("   • Instruction: Flexible, general-purpose, higher cost")
    print("   • Hybrid: Best of both worlds for complex systems")
    print("\nThis completes Chapter 6: Fine-tuning for Classification!")
    print("=" * 70)


if __name__ == "__main__":
    main()
