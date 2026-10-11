"""Lightweight smoke tests for the ranking engine and game logic.

These tests require the ranking artifacts (run `python src/export_word2vec.py`
with the training venv). They only guess in-vocabulary words.
"""

from __future__ import annotations

import pytest

from src.config import EMBEDDINGS_FILE
from src.game import GameEngine

pytestmark = pytest.mark.skipif(
    not EMBEDDINGS_FILE.exists(), reason="embeddings not precomputed yet"
)


@pytest.fixture(scope="module")
def engine() -> GameEngine:
    return GameEngine.load()


def test_secret_has_rank_one(engine: GameEngine) -> None:
    secret = engine.secret_for_number(0)
    result = engine.guess(0, secret)
    assert result.win is True
    assert result.rank == 1


def test_ranks_are_unique_and_complete(engine: GameEngine) -> None:
    secret = engine.secret_for_number(3)
    ranks = engine.index.ranking_for(secret)
    assert ranks.min() == 1
    assert ranks.max() == len(engine.index)
    assert len(set(ranks.tolist())) == len(ranks)


def test_hint_is_closer_than_best(engine: GameEngine) -> None:
    hint = engine.hint(5, best_rank=1000)
    assert hint.rank is not None and hint.rank < 1000


def test_nearest_excludes_secret(engine: GameEngine) -> None:
    secret = engine.secret_for_number(7)
    nearest = engine.nearest(7, top=5)
    words = [w for w, _ in nearest]
    assert secret not in words
    assert all(r >= 2 for _, r in nearest)
