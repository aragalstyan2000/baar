"""Build an *enjoyable* Armenian vocabulary for the Contexto game.

Strategy
--------
We only want words that Armenians actually know and use, so we rank words by
real corpus frequency (news sentences + conversational subtitles) and keep only
those that are valid base-form lemmas in the hunspell dictionary. Intersecting
with hunspell base forms simultaneously:

  * drops obscure / technical words (low frequency),
  * drops misspellings and tokenisation noise (not valid lemmas),
  * drops most proper nouns (stored capitalised in hunspell), and
  * collapses inflected clutter (only base forms survive).

Outputs:
  data/answers.txt  - curated daily secrets (content words, stopwords removed)

The ranking ladder (data/words.txt / embeddings.npy) is produced separately by
src/export_word2vec.py.

Run:
    python -m src.build_wordlist
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

# Armenian letters (lowercase incl. ligature և).
_LOWER = {chr(c) for c in range(0x0561, 0x0588)}
_TOKEN = re.compile(r"[\u0531-\u0556\u0561-\u0587]+")

MIN_LEN = 3
MAX_LEN = 20

# Tuning knobs --------------------------------------------------------------
ANSWER_MIN_FREQ = 25    # min frequency when frequency answers are enabled
ANSWER_MIN_LEN = 4      # avoid tiny particles as secrets
ANSWER_LIMIT = 1200     # cap on the daily-secret pool
# Daily secrets are drawn only from the curated seed by default (every secret is
# a word all Armenians know). Set True to also allow common news-frequency words.
INCLUDE_FREQUENCY_ANSWERS = False
SEED_FILE = DATA / "seed_answers.txt"

# Function words / stopwords that are common but make poor secrets.
STOPWORDS = {
    "այդ", "այս", "այն", "այսօր", "այստեղ", "այնտեղ", "այսպես", "այդպես",
    "այնպես", "այսինքն", "ահա", "համար", "հետ", "մասին", "իսկ", "բայց",
    "մեր", "ինչ", "եթե", "ենք", "պետք", "կարող", "կարելի", "շատ", "ավելի",
    "նաեւ", "նաև", "մենք", "հետո", "մեջ", "այլ", "միայն", "կամ", "իրենց",
    "էին", "վրա", "քանի", "նրա", "ինչպես", "մեկ", "ըստ", "արդեն", "երբ",
    "եղել", "անգամ", "սակայն", "ամեն", "բոլոր", "կողմից", "առաջ", "բան",
    "նման", "պիտի", "նրանց", "հիմա", "հենց", "մոտ", "նրանք", "որեւէ",
    "որևէ", "ինձ", "որպես", "ունի", "ինքը", "մեզ", "վերջին", "ընթացքում",
    "որոնք", "դուրս", "որովհետեւ", "որովհետև", "դեռ", "ապա", "մեկը",
    "նույն", "ձեր", "դրա", "ում", "մինչեւ", "մինչև", "նրան", "իրեն",
    "իհարկե", "նույնիսկ", "քան", "քիչ", "տակ", "որտեղ", "դեպ", "որպեսզի",
    "դուք", "կան", "շուրջ", "ընդամենը", "այնքան", "պատճառով", "որում",
    "մյուս", "եւս", "ևս", "համաձայն", "դրանք", "բացի", "լինել", "որը",
    "որի", "դրանից", "այնուհետեւ", "այնուհետև", "ովքեր", "ինչու", "ինչո",
    "երբեք", "միշտ", "գուցե", "թերեւս", "թերևս", "ուր", "ուրեմն", "ուստի",
    "սույն", "վերը", "ներքեւ", "ներքև", "այժմ", "նախ",
    "երկու", "երեք", "չորս", "հինգ", "վեց", "յոթ", "ութ", "ինը", "տաս",
    "առաջին", "երկրորդ", "հազար", "միլիոն", "տոկոս",
    "եւ", "ետ", "դեպի", "հատ", "նրանից",
    "երկուշաբթի", "երեքշաբթի", "չորեքշաբթի", "հինգշաբթի", "ուրբաթ",
    "շաբաթ", "կիրակի", "երեկ", "վաղը",
}


def _clean(word: str) -> bool:
    return MIN_LEN <= len(word) <= MAX_LEN and all(ch in _LOWER for ch in word)


def build_frequencies() -> collections.Counter:
    """Combine news-sentence and subtitle frequencies into one counter."""
    freq: collections.Counter = collections.Counter()

    sentences = RAW / "leipzig_sentences.txt"
    if sentences.exists():
        with sentences.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                for word in _TOKEN.findall(parts[1].lower()):
                    if _clean(word):
                        freq[word] += 1

    subtitles = RAW / "freq_hy.txt"
    if subtitles.exists():
        with subtitles.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit() and _clean(parts[0]):
                    freq[parts[0]] += int(parts[1])

    return freq


def load_hunspell_base() -> set[str]:
    base: set[str] = set()
    with (RAW / "hunspell_hy.dic").open(encoding="utf-8") as fh:
        next(fh, None)  # count line
        for line in fh:
            lemma = line.strip().split("/", 1)[0].strip()
            if 2 <= len(lemma) <= MAX_LEN and all(ch in _LOWER for ch in lemma):
                base.add(lemma)
    return base


def load_seed() -> list[str]:
    """Read the curated everyday-word seed list (order preserved, deduped)."""
    if not SEED_FILE.exists():
        return []
    seen: set[str] = set()
    seed: list[str] = []
    for line in SEED_FILE.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if not word or word.startswith("#"):
            continue
        if word not in seen:
            seen.add(word)
            seed.append(word)
    return seed


def main() -> None:
    base = load_hunspell_base()
    seed = load_seed()

    # Seed words are curated, so we trust Armenian-clean spellings even when the
    # (incomplete) hunspell dictionary lacks them; absences are reported only.
    valid_seed = [w for w in seed if 2 <= len(w) <= MAX_LEN
                  and all(ch in _LOWER for ch in w)]
    not_in_dict = [w for w in valid_seed if w not in base]

    # Answer pool: curated seed words (always), optionally extended with common
    # frequency-based content words.
    freq_answers: list[str] = []
    if INCLUDE_FREQUENCY_ANSWERS:
        freq = build_frequencies()
        freq_answers = [
            w for w, c in freq.most_common()
            if c >= ANSWER_MIN_FREQ
            and w in base
            and len(w) >= ANSWER_MIN_LEN
            and w not in STOPWORDS
        ]
    answers: list[str] = []
    seen_ans: set[str] = set()
    for w in valid_seed + freq_answers:
        if w not in seen_ans:
            seen_ans.add(w)
            answers.append(w)
    answers = answers[:ANSWER_LIMIT]

    (DATA / "answers.txt").write_text("\n".join(answers) + "\n", encoding="utf-8")

    print(f"answer pool words: {len(answers)}  (seed kept: {len(valid_seed)})")
    if not_in_dict:
        print(f"seed not in dict ({len(not_in_dict)}, kept anyway): {' '.join(not_in_dict)}")
    print("sample answers   :", " ".join(answers[:25]))


if __name__ == "__main__":
    main()
