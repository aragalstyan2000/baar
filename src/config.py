"""Central configuration and paths for the Armenian Contexto backend."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

WORDS_FILE = DATA / "words.txt"
ANSWERS_FILE = DATA / "answers.txt"

# Ranking vocabulary: the words the game ranks guesses against (the "ladder").
EMBEDDINGS_FILE = DATA / "embeddings.npy"
EMBEDDINGS_WORDS_FILE = DATA / "embeddings_words.txt"

# Full word2vec table: every word the model knows. Used only to rank
# out-of-ladder guesses (e.g. inflected forms) the way Contexto does.
W2V_VECTORS_FILE = DATA / "w2v_vectors.npy"
W2V_WORDS_FILE = DATA / "w2v_words.txt"

# The semantic backend is a word2vec model (skip-gram, no subwords) trained on
# the CC-100 Armenian web corpus with phrase detection (see train_word2vec.py),
# exported to numpy. Kept only for reference; the runtime loads no neural model.
MODEL_NAME = "word2vec (skip-gram, 300d, phrase-aware, trained on CC-100 Armenian)"

# Daily game numbering starts here. game_number = (today - EPOCH).days
EPOCH = _dt.date(2024, 1, 1)
