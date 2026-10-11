"""Export the trained word2vec model into plain numpy artifacts.

Run with the training venv (gensim installed):

    .venv_train/bin/python src/export_word2vec.py

Produces (consumed at runtime by numpy only), identical in format to the other
exporters so the runtime is unchanged:
  * data/embeddings.npy / data/embeddings_words.txt  -- ranking "ladder"
  * data/w2v_vectors.npy / data/w2v_words.txt        -- full table for OOV guesses
  * data/words.txt                                   -- ladder word list

The ladder is the model vocabulary intersected with real Armenian base words
(hunspell), capped to the most-frequent slice for "commonness", plus every
curated answer. All vectors are L2-normalized float32 so cosine == dot product.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MODEL_PATH = RAW / "word2vec.model"

ARM_WORD = re.compile(r"^[\u0531-\u0556\u0561-\u0586\u0587]+$")
MIN_LEN = 2

# Commonness cutoff for the ranking ladder. model.wv.index_to_key is sorted by
# frequency, so a word's position is its frequency rank. Keeps the game to words
# people actually know. Curated answers are always included regardless.
LADDER_MAX_RANK = 80000


def load_hunspell_base() -> set[str]:
    """Base-form (lemma) Armenian words from the hunspell dictionary."""
    base: set[str] = set()
    path = RAW / "hunspell_hy.dic"
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.split("/")[0].strip().lower()
        if len(word) >= MIN_LEN and ARM_WORD.match(word):
            base.add(word)
    return base


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main() -> None:
    model = Word2Vec.load(str(MODEL_PATH))
    wv = model.wv
    print(f"Loaded word2vec: {len(wv):,} words, dim={wv.vector_size}")

    # --- full table (for out-of-ladder guesses) ------------------------------
    full_words = [
        w for w in wv.index_to_key if ARM_WORD.match(w) and len(w) >= MIN_LEN
    ]
    full_matrix = normalize_matrix(np.vstack([wv[w] for w in full_words]))
    np.save(DATA / "w2v_vectors.npy", full_matrix)
    (DATA / "w2v_words.txt").write_text("\n".join(full_words) + "\n", encoding="utf-8")
    print(f"Full table: {len(full_words):,} words -> w2v_vectors.npy")

    # --- ranking ladder: real, reasonably-common base words only -------------
    base = load_hunspell_base()
    index = {w: i for i, w in enumerate(full_words)}
    ladder = [
        w for i, w in enumerate(full_words)
        if i < LADDER_MAX_RANK and w in base
    ]

    # Every daily answer must be rankable, so union in answer words the model
    # knows even if hunspell lacks them or they fall past the cap.
    answers_path = DATA / "answers.txt"
    if answers_path.exists():
        in_ladder = set(ladder)
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and w in index and w not in in_ladder:
                ladder.append(w)
                in_ladder.add(w)

    ladder_matrix = full_matrix[[index[w] for w in ladder]]
    np.save(DATA / "embeddings.npy", ladder_matrix)
    (DATA / "embeddings_words.txt").write_text("\n".join(ladder) + "\n", encoding="utf-8")
    (DATA / "words.txt").write_text("\n".join(ladder) + "\n", encoding="utf-8")
    print(f"Ranking ladder: {len(ladder):,} words -> embeddings.npy")


if __name__ == "__main__":
    main()
