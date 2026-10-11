"""Build the ranking artifacts for the game.

The semantic backend is a word2vec model (skip-gram, no subwords) trained on the
large CC-100 Armenian web corpus with phrase detection (see src/train_word2vec.py).
Training + export need a venv where gensim builds, which is not Python 3.14:

    # one-time: create the training venv
    python3.12 -m venv .venv_train
    .venv_train/bin/pip install gensim numpy

    # one-time: fetch + decompress the CC-100 Armenian corpus (~775MB -> ~5.5GB)
    curl -sS -o data/raw/cc100_hy.txt.xz https://data.statmt.org/cc-100/hy.txt.xz
    xz -dk -T0 data/raw/cc100_hy.txt.xz

    # (re)build everything
    .venv_train/bin/python src/train_word2vec.py     # trains data/raw/word2vec.model
    .venv_train/bin/python src/export_word2vec.py    # writes data/embeddings.npy etc.

This module is a convenience dispatcher: if gensim is importable it runs the
export step; otherwise it prints the instructions above.
"""
from __future__ import annotations


def main() -> None:
    try:
        import gensim  # noqa: F401
    except ImportError:
        print(__doc__)
        return
    from .export_word2vec import main as export_main

    export_main()


if __name__ == "__main__":
    main()
