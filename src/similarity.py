"""Vocabulary index and similarity ranking for the Contexto game.

Given the secret word, every vocabulary word is ranked by cosine similarity to
it. Rank 1 is the secret word itself; rank 2 is the closest word, and so on.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import EMBEDDINGS_FILE, EMBEDDINGS_WORDS_FILE


class VocabularyIndex:
    """Holds the word list and its (normalized) embedding matrix."""

    def __init__(self, words: list[str], matrix: np.ndarray):
        if len(words) != matrix.shape[0]:
            raise ValueError("words and matrix size mismatch")
        self.words = words
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.word_to_index = {w: i for i, w in enumerate(words)}

    @classmethod
    def load(cls) -> "VocabularyIndex":
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"{EMBEDDINGS_FILE} not found. Run `python src/export_word2vec.py` "
                "with the training venv first (see src/precompute.py)."
            )
        matrix = np.load(EMBEDDINGS_FILE)
        words = [
            w.strip()
            for w in EMBEDDINGS_WORDS_FILE.read_text(encoding="utf-8").splitlines()
            if w.strip()
        ]
        return cls(words, matrix)

    def __len__(self) -> int:
        return len(self.words)

    def contains(self, word: str) -> bool:
        return word in self.word_to_index

    @lru_cache(maxsize=256)
    def ranking_for(self, secret: str) -> np.ndarray:
        """Return an int array `ranks` where `ranks[i]` is the 1-based rank of
        vocabulary word `i` relative to ``secret`` (rank 1 == secret word)."""
        idx = self.word_to_index[secret]
        sims = self.matrix @ self.matrix[idx]
        # argsort descending; most similar (the secret itself) first.
        order = np.argsort(-sims, kind="stable")
        ranks = np.empty(len(order), dtype=np.int32)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
        return ranks

    @lru_cache(maxsize=256)
    def _ordered_indices(self, secret: str) -> np.ndarray:
        idx = self.word_to_index[secret]
        sims = self.matrix @ self.matrix[idx]
        return np.argsort(-sims, kind="stable")

    def rank_of(self, secret: str, guess: str) -> int:
        """1-based rank of ``guess`` relative to ``secret``."""
        ranks = self.ranking_for(secret)
        return int(ranks[self.word_to_index[guess]])

    def similarity(self, secret: str, guess: str) -> float:
        a = self.matrix[self.word_to_index[secret]]
        b = self.matrix[self.word_to_index[guess]]
        return float(a @ b)

    def word_at_rank(self, secret: str, rank: int) -> str:
        """Return the vocabulary word occupying the given 1-based rank."""
        order = self._ordered_indices(secret)
        rank = max(1, min(rank, len(order)))
        return self.words[int(order[rank - 1])]

    @lru_cache(maxsize=256)
    def _sims_to(self, secret: str) -> np.ndarray:
        idx = self.word_to_index[secret]
        return self.matrix @ self.matrix[idx]

    def external_rank(self, secret: str, embedding: np.ndarray) -> tuple[int, float]:
        """Rank an arbitrary (normalized) embedding against ``secret``.

        Returns ``(rank, similarity)``. Used for guesses that are not part of the
        precomputed vocabulary.
        """
        secret_vec = self.matrix[self.word_to_index[secret]]
        sim = float(secret_vec @ embedding.astype(np.float32))
        sims = self._sims_to(secret)
        # Number of vocabulary words strictly closer to the secret, +1.
        rank = int(np.count_nonzero(sims > sim)) + 1
        return rank, sim

    def nearest(self, secret: str, top: int = 10) -> list[tuple[str, int]]:
        """Return the ``top`` closest words (excluding the secret itself)."""
        order = self._ordered_indices(secret)
        out: list[tuple[str, int]] = []
        for rank, i in enumerate(order[1 : top + 1], start=2):
            out.append((self.words[int(i)], rank))
        return out
