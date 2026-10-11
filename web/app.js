"use strict";

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts);

const el = {
  form: $("guess-form"),
  input: $("guess-input"),
  submit: $("guess-submit"),
  toast: $("toast"),
  list: $("guess-list"),
  latest: $("latest"),
  empty: $("empty-state"),
  gameLabel: $("game-label"),
  count: $("guess-count"),
  win: $("win-banner"),
  winSub: $("win-sub"),
  winTitle: $("win-title"),
  menuBtn: $("menu-btn"),
  sheet: $("sheet"),
  backdrop: $("sheet-backdrop"),
};

let game = null;      // { game_number, date, total_words, is_daily }
let guesses = [];     // [{ word, rank, similarity, win, hint, in_vocabulary }]
let finished = false; // won or gave up

// ---------- state persistence ----------
const storeKey = () => `baar:game:${game.game_number}`;

function saveState() {
  try {
    localStorage.setItem(storeKey(), JSON.stringify({ guesses, finished }));
  } catch (_) {}
}
function loadState() {
  try {
    const raw = localStorage.getItem(storeKey());
    if (raw) {
      const s = JSON.parse(raw);
      guesses = Array.isArray(s.guesses) ? s.guesses : [];
      finished = !!s.finished;
    } else {
      guesses = [];
      finished = false;
    }
  } catch (_) {
    guesses = [];
    finished = false;
  }
}

// ---------- visual mapping (log-scaled proximity + hue) ----------
function proximity(rank) {
  const total = (game && game.total_words) || 20000;
  if (rank <= 1) return 1;
  const p = 1 - Math.log(rank) / Math.log(total);
  return Math.min(1, Math.max(0, p));
}
function barWidth(rank) {
  return Math.max(4, proximity(rank) * 100);
}
function barColor(rank) {
  if (rank === 1) return "linear-gradient(90deg,#16a34a,#22c55e)";
  const p = proximity(rank);          // 0 (far) .. 1 (close)
  const hue = Math.round(p * 120);    // 0 red .. 120 green
  const light = 70 - p * 18;          // closer = a touch deeper
  return `hsl(${hue} 85% ${light}%)`;
}

// ---------- rendering ----------
function rowHTML(g, opts = {}) {
  const li = document.createElement(opts.li ? "li" : "div");
  li.className = "row" + (opts.pinned ? " row--pinned" : "") +
    (opts.animate ? " row--new" : "") + (g.rank === 1 ? " row--win" : "");
  const fill = document.createElement("span");
  fill.className = "row__fill";
  fill.style.width = barWidth(g.rank) + "%";
  fill.style.background = barColor(g.rank);
  const word = document.createElement("span");
  word.className = "row__word";
  word.textContent = g.word + (g.hint ? " 💡" : "");
  const rank = document.createElement("span");
  rank.className = "row__rank";
  rank.textContent = "№" + g.rank.toLocaleString("hy-AM");
  li.append(fill, word, rank);
  return li;
}

function render(newWord) {
  // meta
  if (game) {
    el.gameLabel.textContent = game.is_daily
      ? `Օրվա խաղ · ${game.date}`
      : `Խաղ №${game.game_number}`;
  }
  const tries = guesses.filter((g) => !g.hint).length;
  el.count.textContent = `${tries.toLocaleString("hy-AM")} փորձ`;

  // empty state
  el.empty.hidden = guesses.length > 0;

  // sorted list (ascending rank)
  const sorted = [...guesses].sort((a, b) => a.rank - b.rank);
  el.list.innerHTML = "";
  for (const g of sorted) {
    el.list.appendChild(rowHTML(g, { li: true, animate: g.word === newWord }));
  }

  // pinned latest (only when not finished and there is a last guess)
  el.latest.hidden = true;
  el.latest.innerHTML = "";
  if (!finished && newWord) {
    const last = guesses.find((g) => g.word === newWord);
    if (last) {
      const label = document.createElement("div");
      label.className = "pin-label";
      label.textContent = "Վերջին գուշակությունը";
      el.latest.append(label, rowHTML(last, { pinned: true, animate: true }));
      el.latest.hidden = false;
    }
  }

  // win banner
  const won = guesses.some((g) => g.win);
  el.win.hidden = !won;
  if (won) {
    el.winSub.textContent = `${tries.toLocaleString("hy-AM")} փորձով`;
  }
}

// ---------- toast ----------
let toastTimer = null;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove("show"), 2200);
}

// ---------- modal dialog ----------
function showDialog({
  title,
  message,
  emoji = "",
  confirmText = "Լավ",
  cancelText = null,
  variant = "confirm",
}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";

    const modal = document.createElement("div");
    modal.className = "modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.innerHTML =
      (emoji ? `<div class="modal__emoji">${emoji}</div>` : "") +
      `<h2 class="modal__title"></h2><p class="modal__msg"></p>` +
      `<div class="modal__actions"></div>`;
    modal.querySelector(".modal__title").textContent = title;
    modal.querySelector(".modal__msg").textContent = message;
    const actions = modal.querySelector(".modal__actions");

    const close = (val) => {
      document.removeEventListener("keydown", onKey);
      backdrop.remove();
      resolve(val);
    };
    const onKey = (e) => {
      if (e.key === "Escape") close(false);
      else if (e.key === "Enter") close(true);
    };

    if (cancelText) {
      const cancel = document.createElement("button");
      cancel.className = "modal__btn modal__btn--cancel";
      cancel.textContent = cancelText;
      cancel.addEventListener("click", () => close(false));
      actions.appendChild(cancel);
    }
    const ok = document.createElement("button");
    ok.className =
      "modal__btn " + (variant === "warn" ? "modal__btn--warn" : "modal__btn--confirm");
    ok.textContent = confirmText;
    ok.addEventListener("click", () => close(true));
    actions.appendChild(ok);

    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) close(false);
    });
    document.addEventListener("keydown", onKey);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    ok.focus();
  });
}

// ---------- actions ----------
async function loadGame(gameId) {
  el.submit.disabled = true;
  try {
    game = await api(`/api/game/${gameId}`).then((r) => {
      if (!r.ok) throw new Error("game");
      return r.json();
    });
  } catch (_) {
    toast("Չհաջողվեց բեռնել խաղը։");
    return;
  }
  loadState();
  // sync URL (without reload) for non-daily games
  const url = new URL(location.href);
  if (game.is_daily) url.searchParams.delete("game");
  else url.searchParams.set("game", game.game_number);
  history.replaceState(null, "", url);
  el.submit.disabled = finished;
  el.input.disabled = finished;
  render();
}

async function submitGuess(raw) {
  const word = (raw || "").trim().toLowerCase();
  if (!word || !game || finished) return;

  const existing = guesses.find((g) => g.word === word);
  if (existing) {
    toast(`«${word}» արդեն փորձել ես (№${existing.rank}).`);
    highlight(word);
    return;
  }

  el.submit.disabled = true;
  let res;
  try {
    res = await api(`/api/game/${game.game_number}/guess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word }),
    });
  } catch (_) {
    toast("Ցանցի սխալ։");
    el.submit.disabled = false;
    return;
  }

  if (res.status === 404) {
    toast(`«${word}» բառը բառարանում չկա։`);
    el.submit.disabled = false;
    return;
  }
  if (!res.ok) {
    toast("Սխալ առաջացավ։");
    el.submit.disabled = false;
    return;
  }

  const data = await res.json();
  guesses.push({
    word: data.word,
    rank: data.rank,
    similarity: data.similarity,
    win: !!data.win,
    in_vocabulary: !!data.in_vocabulary,
    hint: false,
  });
  if (data.win) {
    finished = true;
    el.input.disabled = true;
    burst();
  }
  saveState();
  el.input.value = "";
  el.submit.disabled = finished;
  render(data.word);
}

function highlight(word) {
  const rows = el.list.querySelectorAll(".row__word");
  for (const w of rows) {
    if (w.textContent.replace(" 💡", "") === word) {
      const row = w.closest(".row");
      row.classList.remove("row--new");
      void row.offsetWidth;
      row.classList.add("row--new");
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      break;
    }
  }
}

async function doHint() {
  if (!game || finished) return;
  const ok = await showDialog({
    emoji: "💡",
    title: "Հուշո՞ւմ",
    message: "Կբացվի քո լավագույն գուշակությունից ավելի մոտ բառ։",
    confirmText: "Ցույց տալ հուշումը",
    cancelText: "Չեղարկել",
  });
  if (!ok) return;
  const best = guesses.reduce(
    (m, g) => (m === null ? g.rank : Math.min(m, g.rank)),
    null
  );
  const q = best !== null ? `?best_rank=${best}` : "";
  try {
    const data = await api(`/api/game/${game.game_number}/hint${q}`).then((r) =>
      r.json()
    );
    if (guesses.some((g) => g.word === data.word)) {
      toast(`Հուշում՝ «${data.word}» (№${data.rank}) արդեն ունես։`);
      highlight(data.word);
      return;
    }
    guesses.push({
      word: data.word,
      rank: data.rank,
      similarity: data.similarity,
      win: false,
      in_vocabulary: true,
      hint: true,
    });
    saveState();
    render(data.word);
    toast(`Հուշում՝ «${data.word}» (№${data.rank}).`);
  } catch (_) {
    toast("Չհաջողվեց հուշել։");
  }
}

async function doGiveUp() {
  if (!game || finished) return;
  const ok = await showDialog({
    emoji: "🏳️",
    title: "Հանձնվե՞լ",
    message: "Սա ցույց կտա օրվա գաղտնի բառը։ Խաղն ավարտված կհամարվի։",
    confirmText: "Ցույց տալ բառը",
    cancelText: "Չեղարկել",
    variant: "warn",
  });
  if (!ok) return;
  try {
    const data = await api(`/api/game/${game.game_number}/giveup`).then((r) =>
      r.json()
    );
    if (!guesses.some((g) => g.word === data.word)) {
      guesses.push({
        word: data.word,
        rank: 1,
        similarity: 1,
        win: true,
        in_vocabulary: true,
        hint: false,
      });
    }
    finished = true;
    el.input.disabled = true;
    el.submit.disabled = true;
    el.winTitle.textContent = "Գաղտնի բառը՝";
    saveState();
    render(data.word);
  } catch (_) {
    toast("Չհաջողվեց։");
  }
}

async function resetGame() {
  const ok = await showDialog({
    emoji: "♻️",
    title: "Զրոյացնե՞լ խաղը",
    message: "Այս խաղի ամբողջ առաջընթացը կջնջվի։",
    confirmText: "Զրոյացնել",
    cancelText: "Չեղարկել",
    variant: "warn",
  });
  if (!ok) return;
  guesses = [];
  finished = false;
  el.input.disabled = false;
  el.submit.disabled = false;
  el.winTitle.textContent = "Ապրե՛ս։ Գտար բառը։";
  localStorage.removeItem(storeKey());
  render();
}

function randomGame() {
  const n = Math.floor(Math.random() * 400); // within the curated answer pool
  loadGame(String(n));
}

// ---------- share ----------
function share() {
  const tries = guesses.filter((g) => !g.hint).length;
  const label = game.is_daily ? `Օրվա խաղ ${game.date}` : `Խաղ №${game.game_number}`;
  const text = `Baar — ${label}\nԳտա ${tries} փորձով։ 🎉`;
  if (navigator.share) {
    navigator.share({ text }).catch(() => {});
  } else {
    navigator.clipboard?.writeText(text);
    toast("Պատճենվեց սեղմատախտակին։");
  }
}

// ---------- confetti ----------
function burst() {
  const n = 80;
  const colors = ["#22c55e", "#4f46e5", "#f59e0b", "#ef4444", "#06b6d4"];
  for (let i = 0; i < n; i++) {
    const p = document.createElement("div");
    p.style.cssText = `position:fixed;top:-10px;left:${Math.random() * 100}vw;
      width:8px;height:12px;z-index:50;pointer-events:none;
      background:${colors[i % colors.length]};opacity:0.95;
      transform:rotate(${Math.random() * 360}deg);border-radius:2px;`;
    document.body.appendChild(p);
    const dur = 1400 + Math.random() * 1200;
    p.animate(
      [
        { transform: p.style.transform + " translateY(0)", opacity: 1 },
        {
          transform: `translateY(100vh) rotate(${Math.random() * 720}deg)`,
          opacity: 0,
        },
      ],
      { duration: dur, easing: "cubic-bezier(.2,.6,.3,1)" }
    ).onfinish = () => p.remove();
  }
}

// ---------- sheet ----------
function openSheet() {
  el.sheet.hidden = false;
  el.backdrop.hidden = false;
}
function closeSheet() {
  el.sheet.hidden = true;
  el.backdrop.hidden = true;
}

// ---------- wire up ----------
el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  submitGuess(el.input.value);
});
el.menuBtn.addEventListener("click", openSheet);
el.backdrop.addEventListener("click", closeSheet);
$("sheet-close").addEventListener("click", closeSheet);
$("hint-btn").addEventListener("click", () => { closeSheet(); doHint(); });
$("giveup-btn").addEventListener("click", () => { closeSheet(); doGiveUp(); });
$("random-btn").addEventListener("click", () => { closeSheet(); randomGame(); });
$("daily-btn").addEventListener("click", () => { closeSheet(); loadGame("daily"); });
$("reset-btn").addEventListener("click", () => { closeSheet(); resetGame(); });
$("share-btn").addEventListener("click", share);

// ---------- boot ----------
(function init() {
  const params = new URLSearchParams(location.search);
  const g = params.get("game") || "daily";
  loadGame(g).then(() => el.input.focus());
})();
