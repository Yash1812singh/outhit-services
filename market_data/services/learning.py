"""The prediction feedback loop: log every prediction, score it when its horizon has
passed, and use those scored outcomes to decide which model variant to trust next.

Why not "retrain after every wrong call"? A single miss is mostly noise; reacting
to each one makes a model chase randomness and get worse. Instead the loop keeps
evidence: each model variant's rank correlation with what actually happened (IC),
first in walk-forward testing and then on live, published predictions. The
variant with the best recent evidence is used; when another starts doing better
on real outcomes, the model switches. Recent results count more than old ones,
so it adapts as markets change.
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import select

from market_data.db import SessionLocal
from market_data.models import PredictionRecord

log = logging.getLogger(__name__)

RECENT_FOLDS = 4  # backtest folds that count as "recent" evidence
LIVE_FULL_WEIGHT_DATES = 40  # resolved live dates at which live evidence weighs as much as the backtest
MISSES_SHOWN = 8


# ---------------------------------------------------------------------------
# Choosing a variant from evidence
# ---------------------------------------------------------------------------


def daily_ic(df: pd.DataFrame, score: str, outcome: str = "fwd_ret") -> pd.Series:
    """Rank correlation between a score and the realised return, one value per date."""
    out = {}
    for d, g in df.groupby("date"):
        g = g[[score, outcome]].dropna()
        if len(g) > 10 and g[score].nunique() > 1:
            out[d] = spearmanr(g[score], g[outcome]).statistic
    return pd.Series(out, dtype=float).sort_index()


def choose(backtest_ic: dict[str, list[float]], live_ic: dict[str, pd.Series] | None = None, default: str = "blend") -> tuple[str, dict[str, dict]]:
    """Pick the variant with the best evidence.

    backtest_ic: per variant, mean IC of each walk-forward fold (oldest first).
    live_ic: per variant, daily IC on resolved live predictions.
    Recent folds count double; live evidence is blended in by how much of it exists."""
    board: dict[str, dict] = {}
    for name, folds in backtest_ic.items():
        folds = [f for f in folds if np.isfinite(f)]
        if not folds:
            continue
        w = np.array([2.0 if i >= len(folds) - RECENT_FOLDS else 1.0 for i in range(len(folds))])
        bt = float(np.dot(w, folds) / w.sum())
        live = live_ic.get(name) if live_ic else None
        n_live = int(live.notna().sum()) if live is not None else 0
        live_mean = float(live.mean()) if n_live else None
        k = min(1.0, n_live / LIVE_FULL_WEIGHT_DATES) * 0.5  # live can reach half the say
        score = bt if live_mean is None else (1 - k) * bt + k * live_mean
        board[name] = {"backtest_ic": bt, "recent_ic": float(np.mean(folds[-RECENT_FOLDS:])), "live_ic": live_mean,
                       "live_dates": n_live, "score": score}
    if not board:
        return default, board
    best = max(board, key=lambda k: board[k]["score"])
    # Only move away from the default on a clear margin, so the choice doesn't flip on noise.
    if default in board and best != default and board[best]["score"] - board[default]["score"] < 0.005:
        best = default
    for name in board:
        board[name]["chosen"] = name == best
    return best, board


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def record(market: str, horizon: int, as_of: date, rows: list[dict]) -> int:
    """Store today's predictions (one per symbol). Re-running the same day keeps the first record."""
    with SessionLocal() as db:
        have = set(db.scalars(select(PredictionRecord.symbol).where(
            PredictionRecord.market == market, PredictionRecord.horizon == horizon, PredictionRecord.as_of == as_of)))
        new = [PredictionRecord(market=market, horizon=horizon, as_of=as_of, **r) for r in rows if r["symbol"] not in have]
        db.add_all(new)
        db.commit()
    return len(new)


def resolve(market: str, horizon: int, closes: pd.DataFrame) -> int:
    """Fill in what happened for every prediction whose horizon has now passed."""
    index = closes.index
    pos = {d.date(): i for i, d in enumerate(index)}
    done = 0
    with SessionLocal() as db:
        pending = db.scalars(select(PredictionRecord).where(
            PredictionRecord.market == market, PredictionRecord.horizon == horizon, PredictionRecord.resolved_at.is_(None))).all()
        by_date: dict[date, list[PredictionRecord]] = {}
        for r in pending:
            by_date.setdefault(r.as_of, []).append(r)
        now = datetime.now(timezone.utc)
        for d, recs in by_date.items():
            i = pos.get(d)
            if i is None or i + horizon >= len(index):
                continue  # not enough trading days yet (or a date we have no prices for)
            end = index[i + horizon]
            rets = {}
            for r in recs:
                if r.symbol in closes.columns:
                    a, b = closes.at[index[i], r.symbol], closes.at[end, r.symbol]
                    if pd.notna(a) and pd.notna(b) and a > 0:
                        rets[r.symbol] = float(b / a - 1)
            if len(rets) < 5:
                continue
            median = float(np.median(list(rets.values())))
            for r in recs:
                if r.symbol in rets:
                    r.fwd_ret = rets[r.symbol]
                    r.outperform = rets[r.symbol] > median
                r.resolved_at = now  # symbols with no price (delisted) are closed out without a result
                done += 1
        db.commit()
    if done:
        log.info("Resolved %d %s/%dd predictions", done, market, horizon)
    return done


def _resolved_frame(market: str, horizon: int) -> pd.DataFrame:
    with SessionLocal() as db:
        recs = db.scalars(select(PredictionRecord).where(
            PredictionRecord.market == market, PredictionRecord.horizon == horizon,
            PredictionRecord.fwd_ret.is_not(None))).all()
    rows = [{"date": r.as_of, "symbol": r.symbol, "prob": r.prob, "decile": r.decile, "variant": r.variant,
             "fwd_ret": r.fwd_ret, "outperform": float(r.outperform), **{f"v_{k}": v for k, v in (r.variants or {}).items()}}
            for r in recs]
    return pd.DataFrame(rows)


def live_variant_ic(market: str, horizon: int) -> dict[str, pd.Series]:
    df = _resolved_frame(market, horizon)
    if df.empty:
        return {}
    return {c[2:]: daily_ic(df, c) for c in df.columns if c.startswith("v_")}


def track_record(market: str, horizon: int, names: dict[str, str] | None = None) -> dict:
    """How the published predictions have actually done."""
    with SessionLocal() as db:
        total = db.query(PredictionRecord).filter_by(market=market, horizon=horizon).count()
        pending = db.query(PredictionRecord).filter_by(market=market, horizon=horizon, resolved_at=None).count()
        first = db.scalar(select(PredictionRecord.as_of).where(PredictionRecord.market == market, PredictionRecord.horizon == horizon)
                          .order_by(PredictionRecord.as_of).limit(1))
    df = _resolved_frame(market, horizon)
    out: dict = {"logged": total, "pending": pending, "resolved": int(len(df)), "first_prediction": str(first) if first else None,
                 "resolved_dates": int(df["date"].nunique()) if not df.empty else 0}
    if df.empty:
        return out
    ic = daily_ic(df, "prob")
    top, bottom = df[df["decile"] == 10], df[df["decile"] == 1]
    out.update({
        "ic_mean": float(ic.mean()) if len(ic) else None,
        "ic_positive_share": float((ic > 0).mean()) if len(ic) else None,
        "top_hit_rate": float(top["outperform"].mean()) if len(top) else None,
        "top_up_rate": float((top["fwd_ret"] > 0).mean()) if len(top) else None,
        "top_avg_return": float(top["fwd_ret"].mean()) if len(top) else None,
        "bottom_avg_return": float(bottom["fwd_ret"].mean()) if len(bottom) else None,
        "avg_return": float(df["fwd_ret"].mean()),
        "by_decile": [{"decile": int(d), "count": int(len(g)), "outperform_rate": float(g["outperform"].mean()),
                       "up_rate": float((g["fwd_ret"] > 0).mean()), "avg_return": float(g["fwd_ret"].mean())}
                      for d, g in df.groupby("decile")],
        "recent": [{"date": str(d), "ic": float(v)} for d, v in ic.tail(30).items()],
    })
    # The model's biggest misses: strongest bullish calls that lagged most, strongest bearish calls that jumped most.
    misses = pd.concat([top.sort_values("fwd_ret").head(MISSES_SHOWN // 2).assign(call="bullish"),
                        bottom.sort_values("fwd_ret", ascending=False).head(MISSES_SHOWN // 2).assign(call="bearish")])
    out["biggest_misses"] = [{"date": str(r["date"]), "symbol": r["symbol"], "name": (names or {}).get(r["symbol"], r["symbol"]),
                              "call": r["call"], "prob": float(r["prob"]), "fwd_ret": float(r["fwd_ret"])}
                             for _, r in misses.iterrows() if (r["call"] == "bullish") != bool(r["outperform"])]
    if len(ic) >= 20:
        sd = ic.std(ddof=1)
        t = float(ic.mean() / (sd / np.sqrt(len(ic)))) if sd > 1e-12 else float(np.sign(ic.mean()) * 99)
        out["ic_t_stat"] = t
        out["live_verdict"] = ("working" if t >= 2 else "failing" if t <= -2 else "unproven")
    else:
        out["live_verdict"] = "too_early"
    return out


def watched() -> list[tuple[str, int]]:
    """Every (market, horizon) that has ever published predictions."""
    with SessionLocal() as db:
        return [(m, h) for m, h in db.execute(select(PredictionRecord.market, PredictionRecord.horizon).distinct())]
