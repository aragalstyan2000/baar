# Baar — Armenian Contexto (backend)

An Armenian-language alternative to [contexto.me](https://contexto.me/en/).
Guess the secret Armenian word; each guess is ranked by semantic similarity to
the secret (rank 1 = the word itself, lower = closer in meaning).

Like Contexto, ranking uses **word2vec word embeddings** (context-based), so
neighbors are semantically related — not words that merely look or sound alike.

## Architecture

```
data/
  raw/                 raw sources: hunspell dict, CC-100 Armenian corpus
                       (cc100_hy.txt), Wikipedia/Leipzig corpora, word2vec.model
  seed_answers.txt     curated everyday words (edit this to curate secrets)
  words.txt            ranking ladder word list (= embeddings_words.txt)
  answers.txt          daily-secret pool (built from the seed list)
  embeddings.npy       L2-normalized ladder vectors (word2vec ∩ real words)
  embeddings_words.txt word order matching embeddings.npy (~16k words)
  w2v_vectors.npy      full word2vec table (for out-of-ladder guesses)
  w2v_words.txt        word order matching w2v_vectors.npy (~275k words)
src/
  config.py            paths + daily epoch
  build_wordlist.py    builds answers.txt from the curated seed list
  train_word2vec.py    trains word2vec on CC-100 + wiki + news (with phrases)
  export_word2vec.py   turns word2vec.model into the numpy artifacts above
  precompute.py        convenience dispatcher around export_word2vec
  embeddings.py        loads the full word2vec table (numpy only) for OOV guesses
  similarity.py        VocabularyIndex: ranking by cosine similarity
  game.py              GameEngine: daily secret selection + guess handling
  api.py               FastAPI app (+ serves the web UI)
web/
  index.html           mobile-first single-page web UI
  styles.css           responsive styling (light/dark, proximity bars)
  app.js               game logic (guesses, hint, give up, share, offline state)
```

### Semantic model (why it feels like Contexto)

The ranking backend is a **word2vec** model (skip-gram, 300-dim, **no subwords**)
trained locally on the **CC-100 Armenian web corpus** (~25M lines) plus Wikipedia
and news, with **phrase detection** (gensim `Phrases`). Two design choices make
the neighbors trustworthy:

- **No subwords.** fastText builds vectors from character n-grams, which in
  suffix-heavy Armenian wrongly links words that merely share an affix (e.g. the
  causative `-ացնել`: *նորացնել* "renew" ≈ *նվաստացնել* "humiliate"). Pure
  word2vec removes that leakage, so neighbors are related by *meaning*.
- **Phrase detection.** Frequent collocations are collapsed into single tokens
  (e.g. *մայր աթոռ* → *մայր_աթոռ*, the religious phrase "Mother See"), so the
  standalone word *մայր* (mother) is no longer pulled toward *աթոռ* (chair). It
  clusters with *հայր* (father), *երեխա* (child), *քույր* (sister) instead.

```
շուն  (dog)    → cat, horse, mouse, sheep …
ձմեռ  (winter) → summer, spring, autumn, cold …
մայր  (mother) → father, parent, sister, mommy …
աթոռ  (chair)  → armchair, table, sofa, bench …
```

The runtime loads only the exported numpy arrays — **no torch / transformers /
gensim at serving time** — so startup is instant (~0.1 s).

### Vocabulary design (why it's playable)

The **ranking ladder** (what guesses are ranked against) is the most-frequent
slice of the word2vec vocabulary intersected with valid **base-form** Armenian
words from the hunspell dictionary (plus every curated answer). A frequency cap
(`LADDER_MAX_RANK`) drops rare/archaic forms, proper nouns, typos and inflected
clutter, leaving ~16k real everyday words, so ranks stay meaningful.

Daily **secrets** come from `data/seed_answers.txt`, a curated list of everyday
words (nature, animals, food, family, body, home, places, verbs, adjectives…).
Edit that file and re-run `build_wordlist` to change the secrets.

Guesses that are not on the ladder but that the vectors still know (typically an
inflected form) are still ranked, using the full table. A word the model has
never seen is rejected — exactly like Contexto's "I don't know this word".

## Setup

Runtime (serving) needs only numpy + FastAPI:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Interactive API docs: http://127.0.0.1:8000/docs

## Web UI

The FastAPI app also serves a mobile-first web client (no build step): just run
the server and open the root URL.

```bash
uvicorn src.api:app --reload
# then open http://127.0.0.1:8000/
```

Features:
- Contexto-style ranked guess list with log-scaled proximity bars coloured from
  green (close) to red (far); the latest guess is pinned on top.
- Responsive and touch-friendly; light/dark via the system theme; iOS-safe input
  (no zoom-on-focus), safe-area padding.
- Hint, Give up, Random game, Daily game, Reset, and Share (Web Share API).
- Progress is saved per game in `localStorage`, so a refresh keeps your guesses.
- `?game=<n>` in the URL loads a specific game number (`/` is today's daily).

The UI talks to the same JSON API below, so a native app can reuse it as-is.

### Rebuilding the model (offline)

Training the word2vec model with gensim needs a venv where gensim builds (not
Python 3.14), so training/export runs in a separate venv:

```bash
python3.12 -m venv .venv_train
.venv_train/bin/pip install gensim numpy

# curate the daily secrets (optional)
python -m src.build_wordlist

# fetch + decompress the CC-100 Armenian corpus (one-time, ~775MB -> ~5.5GB)
curl -sS -o data/raw/cc100_hy.txt.xz https://data.statmt.org/cc-100/hy.txt.xz
xz -dk -T0 data/raw/cc100_hy.txt.xz

# train word2vec (skip-gram, no subwords, with phrase detection) — ~50 min
.venv_train/bin/python src/train_word2vec.py     # -> data/raw/word2vec.model

# export numpy artifacts
.venv_train/bin/python src/export_word2vec.py    # -> data/embeddings.npy etc.
```

## API

`{game}` is either `daily` (today's puzzle) or an integer game number.

| Method | Path                          | Description                               |
|--------|-------------------------------|-------------------------------------------|
| GET    | `/health`                     | Service + vocabulary size                 |
| GET    | `/api/game/{game}`            | Game info (number, date, total words)     |
| POST   | `/api/game/{game}/guess`      | Body `{ "word": "ծով" }` → rank           |
| GET    | `/api/game/{game}/hint`       | `?best_rank=` → a closer word             |
| GET    | `/api/game/{game}/giveup`     | Reveal the secret word                     |

### Guess response

```json
{
  "word": "ջուր",
  "rank": 4,
  "similarity": 0.87,
  "found": true,
  "win": false,
  "in_vocabulary": true,
  "message": ""
}
```

Out-of-ladder but known words are ranked with `in_vocabulary: false`. Unknown
words (not in the model) return HTTP 404.

## Daily puzzle

`game_number = (today - 2024-01-01).days`, and the secret is
`answers[game_number % len(answers)]` — deterministic and stable per day.
