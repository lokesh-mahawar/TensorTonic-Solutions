from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        # Add special tokens
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }

        # Collect unique words
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        # Add words in sorted order
        for word in sorted(unique_words):
            self.word_to_id[word] = len(self.word_to_id)

        # Reverse mapping
        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use <UNK> for unknown words.
        """
        words = text.lower().split()

        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in words
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = [
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        ]

        return " ".join(words)