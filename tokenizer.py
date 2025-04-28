import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_from_scratch.tokenizer import Tokenizer

if __name__ == "__main__":
    # Load dataset from package root ‘data’ directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    file_path = os.path.join(os.path.dirname(base_dir), "data", "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(dataset=text)
    vocab = tokenizer.get_vocabulary()
    print(f"Loaded dataset from {file_path}")
    print(f"Vocabulary size: {len(vocab)}")
    sample = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
    encoded = tokenizer.encode(sample)
    print(f"Encoded text: {encoded}")
    print(f"Decoded text: {tokenizer.decode(encoded)}")
