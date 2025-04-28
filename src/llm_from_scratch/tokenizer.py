import re
from typing import Dict, List, Optional, Sequence

class Tokenizer:
    """
    Tokenizer class for encoding and decoding text.
    Initialize with either a dataset or a pre-existing vocabulary.
    """
    def __init__(self, dataset: Optional[str] = None, vocabulary: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize the tokenizer with either:
        - dataset: a text string to build the vocabulary from
        - vocabulary: a pre-existing mapping of tokens to integer indices
        """
        if dataset is None and vocabulary is None:
            raise ValueError("Either dataset or vocabulary must be provided.")
        self.dataset: Optional[str] = dataset
        if vocabulary is None:
            # Build vocabulary from the provided dataset string
            self.vocabulary: Dict[str, int] = self._build_vocabulary(dataset)  # type: ignore
        else:
            # Use the provided token-to-index mapping
            self.vocabulary: Dict[str, int] = vocabulary
        # Map from token string to integer index
        self.str_to_int: Dict[str, int] = self.vocabulary
        # Map from integer index to token string
        self.int_to_str: Dict[int, str] = {i: s for s, i in self.vocabulary.items()}

    def _build_vocabulary(self, dataset: str) -> Dict[str, int]:
        """
        Builds a vocabulary from the dataset.
        The vocabulary is a mapping from tokens to unique integers.
        """
        words = self._preprocess_text(dataset)
        sorted_words = sorted(set(words))
        vocab = {token: integer for integer, token in enumerate(sorted_words)}
        return vocab

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesses the input text by removing unwanted characters and splitting into tokens.
        """
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item.strip() for item in words if item.strip()]
        return words

    def _int_to_str(self, index: int) -> str:
        """
        Converts an integer index to its corresponding token.
        """
        if index < 0 or index >= len(self.vocabulary):
            raise ValueError(f"Index {index} out of range.")
        return list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)]

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Returns the vocabulary mapping tokens to integer indices.
        """
        return self.vocabulary

    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text to a sequence of token indices.
        """
        tokens = self._preprocess_text(text)
        encoded: List[int] = []
        for token in tokens:
            if token not in self.vocabulary:
                raise ValueError(f"Unknown token: '{token}'")
            encoded.append(self.vocabulary[token])
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decodes a sequence of integer indices back to text.
        """
        text = " ".join([self.int_to_str[i] for i in tokens])
        text = re.sub(r'\s+([,.?\!"()\'])', r'\1', text)
        return text
