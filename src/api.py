"""FastAPI backend for the Armenian Contexto-style word game.

Run:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import datetime as _dt
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .game import GameEngine

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

engine: GameEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = GameEngine.load()
    yield


app = FastAPI(
    title="Baar — Armenian Contexto",
    description="Guess the secret Armenian word. Lower rank = closer in meaning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> GameEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Game engine not ready.")
    return engine


def resolve_number(game: str, eng: GameEngine) -> int:
    if game == "daily":
        return eng.today_number()
    try:
        return int(game)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="game must be 'daily' or an integer.") from exc


# --- schemas -----------------------------------------------------------------
class GuessRequest(BaseModel):
    word: str = Field(..., description="The Armenian word to guess.")


class GuessResponse(BaseModel):
    word: str
    rank: int | None
    similarity: float | None
    found: bool
    win: bool
    in_vocabulary: bool
    message: str = ""


class GameInfo(BaseModel):
    game_number: int
    date: _dt.date
    total_words: int
    is_daily: bool


class WordRank(BaseModel):
    word: str
    rank: int


# --- routes ------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    eng = get_engine()
    return {"status": "ok", "total_words": len(eng.index)}


@app.get("/api/game/{game}", response_model=GameInfo)
def game_info(game: str) -> GameInfo:
    eng = get_engine()
    number = resolve_number(game, eng)
    return GameInfo(
        game_number=number,
        date=eng.date_for_number(number),
        total_words=len(eng.index),
        is_daily=(number == eng.today_number()),
    )


@app.post("/api/game/{game}/guess", response_model=GuessResponse)
def guess(game: str, body: GuessRequest) -> GuessResponse:
    eng = get_engine()
    number = resolve_number(game, eng)
    result = eng.guess(number, body.word)
    if not result.found:
        raise HTTPException(status_code=404, detail=result.message or "Word not found.")
    return GuessResponse(**result.__dict__)


@app.get("/api/game/{game}/hint", response_model=GuessResponse)
def hint(game: str, best_rank: int | None = None) -> GuessResponse:
    eng = get_engine()
    number = resolve_number(game, eng)
    result = eng.hint(number, best_rank)
    return GuessResponse(**result.__dict__)


@app.get("/api/game/{game}/giveup", response_model=GuessResponse)
def giveup(game: str) -> GuessResponse:
    eng = get_engine()
    number = resolve_number(game, eng)
    word = eng.giveup(number)
    return GuessResponse(
        word=word, rank=1, similarity=1.0, found=True, win=True,
        in_vocabulary=True, message="The secret word.",
    )


# --- static web UI (mounted last so API routes take precedence) --------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
