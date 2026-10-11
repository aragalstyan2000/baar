"""Runtime access to the trained word2vec vectors.

The vectors are trained offline (see ``src/train_word2vec.py``) and exported to
plain numpy arrays by ``src/export_word2vec.py``. This module only *loads* those
arrays, so the serving process needs nothing more than numpy -- no torch, no
transformers.

It is used for out-of-ladder guesses: words a player types that are not part of
the ranking vocabulary but that the vectors still know (typically inflected forms).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import W2V_VECTORS_FILE, W2V_WORDS_FILE


class WordVectors:
    """Full word2vec table with fast word -> normalized vector lookup."""

    def __init__(self, words: list[str], matrix: np.ndarray):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.word_to_index = {w: i for i, w in enumerate(words)}

    @classmethod
    def load(cls) -> "WordVectors":
        if not W2V_VECTORS_FILE.exists():
            raise FileNotFoundError(
                f"{W2V_VECTORS_FILE} not found. Run `python src/export_word2vec.py` "
                "with the training venv first."
            )
        matrix = np.load(W2V_VECTORS_FILE)
        words = [
            w.strip()
            for w in W2V_WORDS_FILE.read_text(encoding="utf-8").splitlines()
            if w.strip()
        ]
        return cls(words, matrix)

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_index

    def get(self, word: str) -> np.ndarray | None:
        """Return the normalized vector for ``word`` or ``None`` if unknown."""
        idx = self.word_to_index.get(word)
        if idx is None:
            return None
        return self.matrix[idx]


@lru_cache(maxsize=1)
def get_vectors() -> "WordVectors":
    return WordVectors.load()
