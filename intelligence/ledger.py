"""Prediction ledger: the AI's live, out-of-sample track record.

The first outlook for each company each day is stored with the price at that
moment. Later, actual returns after 1, 5 and 21 trading days are filled in, and
the track record compares what the AI said with what happened.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import select

from intelligence.db import Prediction, SessionLocal, aware, now

log = logging.getLogger(__name__)
HORIZONS = {"1d": 1, "5d": 5, "21d": 21}


def record(outlook: dict) -> bool:
    if outlook.get("price") is None:
        return False
    day = now().date().isoformat()
    with SessionLocal() as db:
        exists = db.scalar(select(Prediction.id).where(Prediction.symbol == outlook["symbol"], Prediction.day == day))
        if exists:
            return False
        db.add(Prediction(
            symbol=outlook["symbol"], market=outlook["market"], day=day, price=outlook["price"], score=outlook["score"],
            label=outlook["label"], components={c["key"]: c["value"] for c in outlook["components"]},
        ))
        db.commit()
        return True


def evaluate() -> int:
    """Fill in realised returns for predictions whose horizons have passed."""
    with SessionLocal() as db:
        pending = list(db.scalars(select(Prediction).where(Prediction.made_at <= now() - timedelta(days=1))).all())
        pending = [p for p in pending if not p.outcome or len(p.outcome) < len(HORIZONS)]
        if not pending:
            return 0
        symbols = sorted({p.symbol for p in pending})
        start = min(aware(p.made_at) for p in pending).date()
        try:
            raw = yf.download(symbols, start=str(start), interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        except Exception as exc:
            log.warning("ledger download failed: %s", exc)
            return 0
        filled = 0
        for p in pending:
            try:
                closes = (raw[p.symbol] if isinstance(raw.columns, pd.MultiIndex) else raw)["Close"].dropna()
            except KeyError:
                continue
            after = closes[closes.index.date > aware(p.made_at).date()]
            outcome = dict(p.outcome or {})
            for k, n in HORIZONS.items():
                if k not in outcome and len(after) >= n:
                    outcome[k] = float(after.iloc[n - 1] / p.price - 1)
            if outcome != (p.outcome or {}):
                p.outcome = outcome
                filled += 1
        db.commit()
        return filled


def track_record() -> dict:
    with SessionLocal() as db:
        rows = list(db.scalars(select(Prediction)).all())
    out = {"predictions_logged": len(rows), "first_logged": min((p.day for p in rows), default=None), "horizons": {}}
    for k in HORIZONS:
        done = [(p.score, p.outcome[k]) for p in rows if p.outcome and k in p.outcome]
        if not done:
            out["horizons"][k] = {"evaluated": 0}
            continue
        s = np.array([d[0] for d in done])
        r = np.array([d[1] for d in done])
        bull, bear = r[s >= 10], r[s <= -10]
        called = (s >= 10) | (s <= -10)
        out["horizons"][k] = {
            "evaluated": int(len(done)),
            "bullish_calls": int(len(bull)),
            "bullish_hit_rate": float((bull > 0).mean()) if len(bull) else None,
            "bullish_avg_return": float(bull.mean()) if len(bull) else None,
            "bearish_calls": int(len(bear)),
            "bearish_hit_rate": float((bear < 0).mean()) if len(bear) else None,
            "bearish_avg_return": float(bear.mean()) if len(bear) else None,
            "all_avg_return": float(r.mean()),
            "directional_accuracy": float((np.sign(s[called]) == np.sign(r[called])).mean()) if called.any() else None,
            "score_return_correlation": float(pd.Series(s).corr(pd.Series(r), method="spearman")) if len(done) > 10 else None,
        }
    return out


def history(symbol: str, limit: int = 60) -> list[dict]:
    with SessionLocal() as db:
        rows = db.scalars(select(Prediction).where(Prediction.symbol == symbol).order_by(Prediction.day.desc()).limit(limit)).all()
    return [{"day": p.day, "price": p.price, "score": p.score, "label": p.label, "outcome": p.outcome} for p in rows]
