"""Loads the trained sentiment model and scores headlines."""

import json
import re
import threading
from functools import lru_cache

import joblib
import numpy as np

from intelligence.sentiment.train import METRICS_PATH, MODEL_PATH, clean, main as train

_lock = threading.Lock()

# Unambiguous price-move phrases. The statistical model sometimes misreads these
# (e.g. "hits 52-week low" as bullish), so they nudge the score in a fixed direction.
LEXICON = [
    (re.compile(r"52-?week low|all-?time low|multi-?year low|six-?year low|new lows?|\b(?:plunge|tumble|slump|crash|sink|slide|tank)(?:s|d|ed)?\b|\bdown \d+(?:\.\d+)?%|\bfalls? \d+(?:\.\d+)?%|\bwipes? out|\brated (?:sell|underperform)", re.I), -1.0),
    (re.compile(r"52-?week high|all-?time high|record high|\b(?:surge|soar|skyrocket|rall(?:y|ies|ied)|zoom)(?:s|d|ed)?\b|\bup \d+(?:\.\d+)?%|\bjumps? \d+(?:\.\d+)?%|\brated (?:buy|outperform)", re.I), 1.0),
]
LEXICON_WEIGHT = 0.35


def _lexicon(text: str) -> float:
    hits = [v for rx, v in LEXICON if rx.search(text)]
    return float(sum(hits) / len(hits)) if hits else 0.0


@lru_cache
def _load() -> dict:
    with _lock:
        if not MODEL_PATH.exists():
            train()  # first run: download the dataset and train (~30 s)
        return joblib.load(MODEL_PATH)


def score(texts: list[str]) -> list[dict]:
    """For each text: label, confidence, and sentiment = P(bullish) - P(bearish) in [-1, 1]."""
    if not texts:
        return []
    bundle = _load()
    model, labels = bundle["model"], bundle["labels"]
    probs = model.predict_proba([clean(t) for t in texts])
    classes = [labels[int(c)] for c in model.classes_]
    out = []
    for text, p in zip(texts, probs):
        d = dict(zip(classes, map(float, p)))
        best = int(np.argmax(p))
        model_sent = d["bullish"] - d["bearish"]
        lex = _lexicon(text)
        sent = model_sent if lex == 0 else (1 - LEXICON_WEIGHT) * model_sent + LEXICON_WEIGHT * lex
        label = classes[best]
        if lex:  # the label follows the corrected score
            label = "bullish" if sent > 0.1 else "bearish" if sent < -0.1 else "neutral"
        out.append({"label": label, "confidence": float(p[best]), "sentiment": float(sent), "model_sentiment": model_sent, "lexicon": lex, "probs": d})
    return out


def explain(text: str, top: int = 6) -> list[dict]:
    """Words/phrases that pushed this text toward bullish or bearish (linear model weights x tf-idf)."""
    bundle = _load()
    pipe = bundle["model"]
    feats, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    x = feats.transform([clean(text)])
    names = np.concatenate([t.get_feature_names_out() for _, t in feats.transformer_list])
    n_words = len(feats.transformer_list[0][1].get_feature_names_out())  # explain with whole words only
    classes = [bundle["labels"][int(c)] for c in clf.classes_]
    direction = clf.coef_[classes.index("bullish")] - clf.coef_[classes.index("bearish")]
    row = x.tocoo()
    contrib = sorted(((names[j] if j < len(names) else "?", float(v * direction[j])) for j, v in zip(row.col, row.data) if j < n_words), key=lambda t: abs(t[1]), reverse=True)
    seen, out = set(), []
    for term, c in contrib:
        t = term.strip()
        if len(t) < 3 or t in seen:
            continue
        seen.add(t)
        out.append({"term": t, "push": "bullish" if c > 0 else "bearish", "weight": round(c, 3)})
        if len(out) == top:
            break
    return out


def metrics() -> dict:
    _load()
    return json.loads(METRICS_PATH.read_text())
