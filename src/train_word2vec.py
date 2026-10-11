"""Train a word2vec model on large, diverse Armenian text.

Why word2vec (and not fastText): fastText builds vectors from character n-grams
(subwords), so in a heavily-suffixing language like Armenian, words that merely
share an affix (e.g. the causative "-ացնել": նորացնել/renew, զորացնել/strengthen,
ստորացնել/humiliate) get artificially similar vectors even though their meanings
differ. word2vec has NO subwords, so similarity comes purely from context: words
cluster only if they are actually used in the same situations.

Why a large, diverse corpus: training word2vec on Armenian Wikipedia alone made
everyday words pick up encyclopedic/religious senses (մայր/mother -> աթոռ/chair
via "Մայր Աթոռ"/Mother See). Training on CC-100 (a broad web crawl) dominates
those niche collocations, so everyday words get everyday senses.

Run with the training venv (gensim installed):

    .venv_train/bin/python src/train_word2vec.py

Then export numpy artifacts:

    .venv_train/bin/python src/export_word2vec.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.phrases import Phrases

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODEL_PATH = RAW / "word2vec.model"

# Corpus files (plain text, one sentence/paragraph per line). The big web crawl
# dominates; Wikipedia + news add topical coverage (science, geography, animals).
CORPUS_FILES = [
    RAW / "cc100_hy.txt",
    RAW / "wiki_sentences.txt",
    RAW / "leipzig_sentences.txt",
]

ARM_TOKEN = re.compile(r"[\u0531-\u0556\u0561-\u0586\u0587]+")
MIN_TOKEN_LEN = 2
# Leipzig lines are "<id>\t<sentence>"; drop a leading numeric id if present.
LEIPZIG_PREFIX = re.compile(r"^\d+\t")


def tokenize(line: str) -> list[str]:
    line = LEIPZIG_PREFIX.sub("", line).lower()
    return [t for t in ARM_TOKEN.findall(line) if len(t) >= MIN_TOKEN_LEN]


class Corpus:
    """Streams and tokenizes all corpus files; re-iterable for gensim."""

    def __init__(self, files: list[Path]) -> None:
        self.files = [f for f in files if f.exists()]

    def __iter__(self):
        for path in self.files:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    tokens = tokenize(line)
                    if tokens:
                        yield tokens


class PhrasedCorpus:
    """Wraps a Corpus, merging frequent bigrams into single phrase tokens.

    Armenian web text is full of fixed proper-noun collocations -- above all
    "Մայր Աթոռ" (Mother See of Etchmiadzin), which occurs ~30k times in CC-100.
    Left as two words, it pollutes the vector of "մայր"/mother so it ends up next
    to "աթոռ"/chair. Collapsing it to a single token "մայր_աթոռ" lets standalone
    "մայր" keep only its genuine mother contexts. Phrase tokens contain "_" and
    so are naturally excluded from the (Armenian-only) exported vocabulary.
    """

    def __init__(self, corpus: Corpus, bigram) -> None:
        self.corpus = corpus
        self.bigram = bigram

    def __iter__(self):
        for tokens in self.corpus:
            yield self.bigram[tokens]


class EpochLogger(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        self.epoch += 1
        print(f"  epoch {self.epoch} done", flush=True)


def main() -> None:
    present = [f.name for f in CORPUS_FILES if f.exists()]
    missing = [f.name for f in CORPUS_FILES if not f.exists()]
    print(f"Corpus files present: {present}")
    if missing:
        print(f"(skipping missing: {missing})")

    corpus = Corpus(CORPUS_FILES)
    workers = os.cpu_count() or 4

    # Detect frequent collocations (e.g. "Մայր Աթոռ") and merge them so they
    # don't pollute the vectors of their component words.
    print("Detecting phrases (one pass)...", flush=True)
    phrases = Phrases(corpus, min_count=50, threshold=12.0, delimiter="_")
    bigram = phrases.freeze()
    del phrases
    phrased = PhrasedCorpus(corpus, bigram)

    model = Word2Vec(
        sentences=phrased,
        vector_size=300,
        window=5,
        min_count=20,      # big corpus: drop rare words / typos
        sg=1,              # skip-gram
        negative=10,
        sample=1e-4,
        epochs=5,
        workers=workers,
        callbacks=[EpochLogger()],
    )
    model.save(str(MODEL_PATH))
    print(f"Saved {MODEL_PATH}  (vocab={len(model.wv):,}, dim={model.wv.vector_size})")


if __name__ == "__main__":
    main()
