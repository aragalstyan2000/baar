"""Game logic: daily secret selection and guess handling."""

from __future__ import annotations

import datetime as _dt
import unicodedata
from dataclasses import dataclass, field

from .config import ANSWERS_FILE, EPOCH
from .similarity import VocabularyIndex


@dataclass
class GuessResult:
    word: str
    rank: int | None
    similarity: float | None
    found: bool          # word could be ranked (in vocab or encodable)
    win: bool
    in_vocabulary: bool
    message: str = ""


@dataclass
class GameEngine:
    index: VocabularyIndex
    answers: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "GameEngine":
        index = VocabularyIndex.load()
        answers = [
            w.strip()
            for w in ANSWERS_FILE.read_text(encoding="utf-8").splitlines()
            if w.strip()
        ]
        # Only keep answers that exist in the embedding vocabulary.
        answers = [w for w in answers if index.contains(w)]
        if not answers:
            raise RuntimeError("No valid answer words found in the vocabulary.")
        return cls(index=index, answers=answers)

    # --- daily game numbering -------------------------------------------------
    @staticmethod
    def today_number(today: _dt.date | None = None) -> int:
        today = today or _dt.date.today()
        return (today - EPOCH).days

    def secret_for_number(self, game_number: int) -> str:
        return self.answers[game_number % len(self.answers)]

    def date_for_number(self, game_number: int) -> _dt.date:
        return EPOCH + _dt.timedelta(days=game_number)

    # --- guessing -------------------------------------------------------------
    @staticmethod
    def normalize(word: str) -> str:
        return unicodedata.normalize("NFC", word).strip().lower()

    def guess(self, game_number: int, raw_word: str) -> GuessResult:
        word = self.normalize(raw_word)
        secret = self.secret_for_number(game_number)

        if not word:
            return GuessResult(word, None, None, False, False, False, "Empty guess.")

        if word == secret:
            sim = 1.0
            return GuessResult(word, 1, sim, True, True, True, "Correct!")

        if self.index.contains(word):
            rank = self.index.rank_of(secret, word)
            sim = self.index.similarity(secret, word)
            return GuessResult(word, rank, sim, True, rank == 1, True)

        # Not part of the ranking ladder. If word2vec still knows the word
        # (typically an inflected form) we rank it; otherwise we reject it,
        # exactly like Contexto's "I don't know this word".
        from . import embeddings as emb

        vec = emb.get_vectors().get(word)
        if vec is not None:
            rank, sim = self.index.external_rank(secret, vec)
            return GuessResult(
                word, rank, sim, True, rank == 1, False,
            )
        return GuessResult(
            word, None, None, False, False, False,
            "Unknown word — not in the dictionary.",
        )

    # --- helpers --------------------------------------------------------------
    def giveup(self, game_number: int) -> str:
        return self.secret_for_number(game_number)

    def hint(self, game_number: int, best_rank: int | None) -> GuessResult:
        """Reveal a word roughly halfway between the best known rank and the top."""
        secret = self.secret_for_number(game_number)
        if best_rank is None or best_rank <= 2:
            target_rank = 2
        else:
            target_rank = max(2, best_rank // 2)
        word = self.index.word_at_rank(secret, target_rank)
        sim = self.index.similarity(secret, word)
        return GuessResult(word, target_rank, sim, True, False, True, "Hint")

    def nearest(self, game_number: int, top: int = 10) -> list[tuple[str, int]]:
        secret = self.secret_for_number(game_number)
        return self.index.nearest(secret, top=top)
