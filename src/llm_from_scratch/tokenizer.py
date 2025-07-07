import re
from typing import Dict, List, Optional, Sequence


class SimpleTokenizerV1:
    """
    A simple tokenizer for text preprocessing and encoding/decoding.

    This tokenizer supports building vocabularies from text datasets and provides
    methods to encode text into integer sequences and decode them back to text.
    Includes special token handling for end-of-text and unknown tokens.

    Args:
        dataset: Optional text string to build vocabulary from
        vocabulary: Optional pre-existing token-to-index mapping

    Raises:
        ValueError: If neither dataset nor vocabulary is provided
    """

    # Special tokens
    END_OF_TEXT_TOKEN: str = "<|endoftext|>"
    UNKNOWN_TOKEN: str = "<|unk|>"

    def __init__(
        self, dataset: Optional[str] = None, vocabulary: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Initialize the tokenizer with either a dataset or pre-existing vocabulary.

        Args:
            dataset: Text string to build vocabulary from. If provided, vocabulary
                    will be built by preprocessing the text and creating token mappings.
            vocabulary: Pre-existing token-to-index mapping. If provided, this will
                       be used directly as the vocabulary.

        Raises:
            ValueError: If neither dataset nor vocabulary is provided.
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
        # Add special tokens to vocabulary with unique indices
        # Use max value + 1 to avoid collisions with existing indices
        next_index = max(self.vocabulary.values()) + 1 if self.vocabulary else 0
        for token in (self.END_OF_TEXT_TOKEN, self.UNKNOWN_TOKEN):
            if token not in self.vocabulary:
                self.vocabulary[token] = next_index
                next_index += 1
        # Map from token string to integer index
        self.str_to_int: Dict[str, int] = self.vocabulary
        # Map from integer index to token string
        self.int_to_str: Dict[int, str] = {i: s for s, i in self.vocabulary.items()}
        # Set special token indices
        self.END_OF_TEXT_INDEX: int = self.str_to_int[self.END_OF_TEXT_TOKEN]
        self.UNKNOWN_TOKEN_INDEX: int = self.str_to_int[self.UNKNOWN_TOKEN]

    def _build_vocabulary(self, dataset: str) -> Dict[str, int]:
        """
        Build vocabulary from text dataset by preprocessing and tokenizing.

        Args:
            dataset: Raw text string to build vocabulary from

        Returns:
            Dictionary mapping tokens to unique integer indices, sorted alphabetically
        """
        words = self._preprocess_text(dataset)
        sorted_words = sorted(set(words))
        vocab = {token: integer for integer, token in enumerate(sorted_words)}
        return vocab

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing on punctuation and whitespace.

        Uses regex to split on common punctuation marks and whitespace while
        preserving the punctuation as separate tokens. Strips whitespace and
        removes empty tokens.

        Args:
            text: Raw input text to preprocess

        Returns:
            List of preprocessed tokens
        """
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item.strip() for item in words if item.strip()]
        return words

    def _int_to_str(self, index: int) -> str:
        """
        Convert integer index to corresponding token string.

        Args:
            index: Integer index to convert

        Returns:
            Token string corresponding to the index

        Raises:
            ValueError: If index is out of vocabulary range
        """
        if index < 0 or index >= len(self.vocabulary):
            raise ValueError(f"Index {index} out of range.")
        return list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)]

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get the complete vocabulary mapping.

        Returns:
            Dictionary mapping token strings to integer indices
        """
        return self.vocabulary

    def encode(self, text: str) -> List[int]:
        """
        Encode text string into sequence of integer token indices.

        Preprocesses the input text and maps each token to its corresponding
        integer index in the vocabulary. Unknown tokens are mapped to the
        unknown token index.

        Args:
            text: Input text to encode

        Returns:
            List of integer indices representing the encoded text
        """
        tokens = self._preprocess_text(text)
        encoded: List[int] = []
        for token in tokens:
            # Map unknown tokens to UNKNOWN_TOKEN_INDEX
            idx = self.vocabulary.get(token, self.UNKNOWN_TOKEN_INDEX)
            encoded.append(idx)
        return encoded

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode sequence of integer indices back to text string.

        Converts each integer index to its corresponding token and joins them
        with spaces. Applies post-processing to remove extra spaces before
        punctuation marks for better readability.

        Args:
            tokens: Sequence of integer indices to decode

        Returns:
            Decoded text string with proper punctuation spacing
        """
        text = " ".join([self.int_to_str[i] for i in tokens])
        text = re.sub(r'\s+([,.?\!"()\'])', r"\1", text)
        return text
