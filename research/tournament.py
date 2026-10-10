"""The method tournament: which stock-selection method actually works, per market and horizon?

For every horizon (1 week, 1 month, 3 months, 1 year):
1. Score every stock on every sampled date with every method (methods.py), and
   train the two ML models walk-forward (each fold uses only data that was
   fully known before the fold starts, with a horizon-length gap).
2. Measure each method by its information coefficient (rank correlation
   between today's score and the next-N-day return), the return spread between
   its top and bottom fifth, and the top fifth's hit rate.
3. Combine methods with an ADAPTIVE, COST-AWARE ensemble: on each date it averages
   the up-to-3 methods with the best positive IC over the previous ~12 months that
   also made money after trading costs over that window, using only outcomes that
   were already known on that date. Costs are charged on each method's turnover. Markets change which effects
   work (in India, reversal/low-volatility led in 2024-25, then momentum took
   over), so a ranking fixed once goes stale; the adaptive one follows the shift.
4. Every number reported is out-of-sample. The test period is split into an
   earlier "selection" half and a later "holdout" half; the verdict is based on
   the holdout half. A fixed ensemble chosen on the selection half is reported
   alongside for comparison.
5. Rank today's stocks with the methods the adaptive rule picks today.

If no method had a positive IC over the last 12 months, the verdict is
"no edge" and the ranking is shown for transparency only.
"""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from research import data, earnings, fundamentals_history, jobs, methods

log = logging.getLogger(__name__)

HORIZONS = {5: "1 week", 21: "1 month", 63: "3 months", 252: "1 year"}
STEP = {5: 5, 21: 5, 63: 10, 252: 21}
FOLDS = 6
VALIDATION_SHARE = 0.6
UNIVERSE = 300
# Round-trip trading cost charged on turnover (brokerage, STT/stamp duty, spread, impact).
COST_ROUND_TRIP = {"in": 0.004, "us": 0.001}
DEFAULT_COST = 0.003
TTL = 12 * 3600


def _row_ic(score: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Spearman correlation per date (row) between score and forward return."""
    mask = score.notna() & fwd.notna()
    s = score.where(mask).rank(axis=1)
    f = fwd.where(mask).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    num = (s * f).sum(axis=1)
    den = np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(mask.sum(axis=1) >= 20)


def _quintiles(score: pd.DataFrame) -> pd.DataFrame:
    return np.ceil(score.rank(axis=1, pct=True) * 5).clip(1, 5)


def _stats(score: pd.DataFrame, fwd: pd.DataFrame, dates, every: int, cost: float = 0.0) -> dict:
    sc, fw = score.loc[dates], fwd.loc[dates]
    ic = _row_ic(sc, fw).dropna()
    q = _quintiles(sc)
    top = fw.where(q == 5).mean(axis=1)
    bottom = fw.where(q == 1).mean(axis=1)
    median = fw.median(axis=1)
    top_hit = (fw.where(q == 5).gt(median, axis=0)).sum(axis=1) / (q == 5).sum(axis=1).replace(0, np.nan)
    # Turnover of the top fifth between successive non-overlapping rebalances, and the
    # top fifth's excess over the average stock after paying for that turnover.
    reb = q.iloc[::every]
    tops = [set(r[r == 5].index) for _, r in reb.iterrows()]
    turns = [1 - len(a & b) / max(1, len(b)) for a, b in zip(tops, tops[1:]) if b]
    turnover = float(np.mean(turns)) if turns else None
    excess = (top - fw.mean(axis=1)).mean() if top.notna().any() else None
    net = (float(excess) - (turnover or 0) * cost) if excess is not None else None
    # t-stat on non-overlapping dates only, so overlapping holding periods don't inflate it
    ind = ic.iloc[::every]
    t = float(ind.mean() / (ind.std(ddof=1) / math.sqrt(len(ind)))) if len(ind) > 2 and ind.std(ddof=1) > 0 else None
    return {
        "ic": float(ic.mean()) if len(ic) else None,
        "ic_t": t,
        "ic_positive_share": float((ic > 0).mean()) if len(ic) else None,
        "top_return": float(top.mean()) if top.notna().any() else None,
        "bottom_return": float(bottom.mean()) if bottom.notna().any() else None,
        "spread": float((top - bottom).mean()) if top.notna().any() else None,
        "top_beats_median": float(top_hit.mean()) if top_hit.notna().any() else None,
        "top_excess": None if excess is None else float(excess),
        "turnover": turnover,
        "net_top_excess": net,
        "dates": int(len(ic)),
    }


def _net_series(score: pd.DataFrame, fwd: pd.DataFrame, every: int, cost: float) -> pd.Series:
    """Per date: top-fifth excess return over the average stock, minus the cost of the turnover
    since the previous non-overlapping rebalance (`every` sample dates earlier)."""
    q = _quintiles(score)
    top = q.eq(5)
    excess = fwd.where(top).mean(axis=1) - fwd.mean(axis=1)
    prev = top.shift(every)
    stay = (top & prev.fillna(False).astype(bool)).sum(axis=1)
    turnover = 1 - stay / top.sum(axis=1).replace(0, np.nan)
    turnover = turnover.where(prev.notna().any(axis=1))
    return excess - turnover.fillna(0.5) * cost


def _ml_cols(feats: dict) -> list[str]:
    return methods.ML_FEATURES + [k for k in methods.OPTIONAL_ML_FEATURES if k in feats]


def _ml_table(feats: dict[str, pd.DataFrame], dates) -> pd.DataFrame:
    cols = {k: feats[k].loc[dates].rank(axis=1, pct=True) for k in _ml_cols(feats)}
    t = pd.concat({k: v.stack(future_stack=True) for k, v in cols.items()}, axis=1)
    t.index.names = ["date", "symbol"]
    return t


def _models():
    return {
        "ml_gbm": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0, random_state=0),
        "ml_linear": LogisticRegression(C=0.1, max_iter=500),
    }


def _fit_predict(table: pd.DataFrame, label: pd.Series, train_idx, pred_idx) -> dict[str, pd.Series]:
    X = table.fillna(0.5)
    tr = X.loc[train_idx]
    y = label.loc[train_idx]
    ok = y.notna()
    out = {}
    for name, m in _models().items():
        m.fit(tr[ok], y[ok])
        out[name] = pd.Series(m.predict_proba(X.loc[pred_idx])[:, 1], index=pred_idx)
    return out


def run_horizon(closes: pd.DataFrame, feats: dict, table: pd.DataFrame, horizon: int, progress, cost: float = DEFAULT_COST) -> dict:
    step = STEP[horizon]
    all_dates = closes.index
    sample = all_dates[252::step]
    if all_dates[-1] not in sample:
        sample = sample.append(pd.DatetimeIndex([all_dates[-1]]))
    fwd = (closes.shift(-horizon) / closes - 1).loc[sample]
    labeled = [d for d in sample if fwd.loc[d].notna().sum() >= 20]
    if len(labeled) < 30:
        return {"available": False, "reason": "Not enough price history for this horizon."}
    val_dates = labeled[int(len(labeled) * (1 - VALIDATION_SHARE)):]
    folds = np.array_split(np.array(val_dates), FOLDS)
    selection = [d for f in folds[: FOLDS // 2] for d in f]
    holdout = [d for f in folds[FOLDS // 2 :] for d in f]
    every = max(1, math.ceil(horizon / step))

    available = methods.available_methods(feats)
    scores = {k: fn(feats).reindex(index=sample, columns=closes.columns) for k, (_, fn) in available.items()}

    # Walk-forward ML: each fold predicted by a model trained only on labels known before it.
    progress(f"{HORIZONS[horizon]}: training machine-learning models walk-forward…")
    tbl = table.loc[table.index.get_level_values("date").isin(sample)]
    median = fwd.median(axis=1)
    label = fwd.gt(median, axis=0).astype(float).where(fwd.notna()).stack(future_stack=True).reindex(tbl.index)
    pos = {d: i for i, d in enumerate(all_dates)}
    ml = {k: pd.Series(np.nan, index=tbl.index) for k in methods.ML_METHODS}
    dates_lvl = tbl.index.get_level_values("date")
    for f in folds:
        start = pos[pd.Timestamp(f[0])]
        cutoff = all_dates[max(0, start - horizon - 1)]
        train_idx = tbl.index[(dates_lvl <= cutoff) & label.notna().to_numpy()]
        pred_idx = tbl.index[dates_lvl.isin(f)]
        if len(train_idx) < 3000:
            continue
        for k, v in _fit_predict(tbl, label, train_idx, pred_idx).items():
            ml[k].loc[pred_idx] = v
    # Today's ML scores from a model trained on everything labelled.
    last = all_dates[-1]
    train_all = tbl.index[label.notna().to_numpy()]
    for k, v in _fit_predict(tbl, label, train_all, tbl.index[dates_lvl == last]).items():
        ml[k].loc[v.index] = v
    for k in methods.ML_METHODS:
        scores[k] = ml[k].unstack().reindex(index=sample, columns=closes.columns)

    progress(f"{HORIZONS[horizon]}: scoring methods on the selection and holdout periods…")
    table_rows = []
    for k, sc in scores.items():
        table_rows.append({"method": k, "label": methods.LABELS[k], "selection": _stats(sc, fwd, selection, every, cost),
                           "holdout": _stats(sc, fwd, holdout, every, cost)})

    # Static ensemble (reference): members chosen once on the selection period.
    chosen = sorted([r for r in table_rows if (r["selection"]["ic"] or 0) > 0], key=lambda r: r["selection"]["ic"], reverse=True)[:3]
    static_members = [r["method"] for r in chosen]
    if static_members:
        static = sum(methods._z(scores[m]) for m in static_members) / len(static_members)
        table_rows.append({"method": "static_ensemble", "label": "Fixed ensemble (" + ", ".join(methods.LABELS[m].split(":")[-1].strip() for m in static_members) + ")",
                           "selection": _stats(static, fwd, selection, every, cost), "holdout": _stats(static, fwd, holdout, every, cost)})

    # Adaptive ensemble (primary): on each date, combine the up-to-3 methods with the best
    # positive IC over the previous ~12 months, counting only outcomes already known that day.
    ics = pd.DataFrame({k: _row_ic(sc, fwd) for k, sc in scores.items()})
    nets = pd.DataFrame({k: _net_series(sc, fwd, every, cost) for k, sc in scores.items()})
    lookback = max(6, round(252 / step))
    ens = pd.DataFrame(np.nan, index=sample, columns=closes.columns)
    active: dict = {}
    for i, d in enumerate(sample):
        known = ics.iloc[max(0, i - every - lookback) : max(0, i - every)]  # outcomes for these dates were known by d
        if len(known) < lookback // 2:
            continue
        mean_ic = known.mean().dropna()
        # Cost-aware: a method must also have made money after trading costs over the same window.
        known_net = nets.iloc[max(0, i - every - lookback) : max(0, i - every)].mean()
        ok = [m for m in mean_ic.index if mean_ic[m] > 0 and known_net.get(m, 0) > 0]
        best = sorted(ok, key=lambda m: mean_ic[m], reverse=True)[:3]
        if best:
            ens.loc[d] = (sum(methods._z(scores[m].loc[[d]]) for m in best) / len(best)).iloc[0]
            active[d] = best
    scores["ensemble"] = ens
    members = active.get(last, [])
    val_all = selection + holdout
    coverage = sum(1 for d in holdout if d in active) / max(1, len(holdout))
    ens_stats = {"method": "ensemble", "label": methods.LABELS["ensemble"], "members": members,
                 "selection": _stats(ens, fwd, selection, every, cost), "holdout": _stats(ens, fwd, holdout, every, cost),
                 "full_validation": _stats(ens, fwd, val_all, every, cost), "active_share_holdout": coverage,
                 "member_history": {str(pd.Timestamp(d).date()): m for d, m in list(active.items())[-12:]}}

    h = ens_stats["holdout"]
    if not members:
        level, text = "none", "No method has shown a positive edge over the last 12 months in this market, so there is no reliable signal for this horizon right now."
    elif (h["ic"] or 0) >= 0.03 and (h["ic_t"] or 0) >= 2 and (h["spread"] or 0) > 0:
        level, text = "meaningful", "The adaptive ensemble kept working out-of-sample: a meaningful, statistically supported edge."
    elif (h["ic"] or 0) > 0.01 and (h["spread"] or 0) > 0:
        level, text = "modest", "The adaptive ensemble kept a small out-of-sample edge. Useful as a tilt; expect many individual misses."
    elif (h["ic"] or 0) < -0.02:
        level, text = "negative", "Out-of-sample, the adaptive ensemble did WORSE than chance: the market kept changing which methods work. Don't rely on this ranking."
    else:
        level, text = "none", "Out-of-sample, the adaptive ensemble showed no reliable edge. Treat this ranking as no better than chance."
    if level in ("meaningful", "modest") and (h.get("net_top_excess") or 0) <= 0:
        level, text = "none", (f"The ranking predicted returns, but its top picks changed so often ({(h.get('turnover') or 0):.0%} turnover per rebalance) "
                               f"that trading costs of {cost:.2%} per round trip ate the edge. Not worth trading on its own.")
    if not members:
        scores["ensemble"] = methods._z(scores["technical"])  # shown for transparency only
        ens = scores["ensemble"]

    # Calibration: how each ensemble quintile actually did on the holdout period.
    qh = _quintiles(ens.loc[holdout])
    fh = fwd.loc[holdout]
    buckets = []
    for q in range(1, 6):
        r = fh.where(qh == q).stack().dropna()
        buckets.append({"quintile": q, "avg_return": float(r.mean()) if len(r) else None, "prob_up": float((r > 0).mean()) if len(r) else None, "observations": int(len(r))})

    today = scores["ensemble"].loc[last].dropna()
    pct = today.rank(pct=True)
    member_pct = {m: scores[m].loc[last].rank(pct=True) for m in members}
    ranking = []
    for sym, p in pct.sort_values(ascending=False).items():
        q = int(min(5, max(1, math.ceil(p * 5))))
        ranking.append({
            "symbol": sym, "percentile": float(p), "quintile": q,
            "by_method": {m: (None if pd.isna(member_pct[m].get(sym)) else float(member_pct[m][sym])) for m in members},
            "hist_avg_return": buckets[q - 1]["avg_return"], "hist_prob_up": buckets[q - 1]["prob_up"],
        })

    return {
        "available": True,
        "horizon_days": horizon,
        "horizon_label": HORIZONS[horizon],
        "selection_period": [str(pd.Timestamp(selection[0]).date()), str(pd.Timestamp(selection[-1]).date())],
        "holdout_period": [str(pd.Timestamp(holdout[0]).date()), str(pd.Timestamp(holdout[-1]).date())],
        "methods": sorted(table_rows, key=lambda r: r["selection"]["ic"] or -1, reverse=True),
        "ensemble": ens_stats,
        "verdict_level": level,
        "verdict": text,
        "quintiles": buckets,
        "ranking": ranking,
    }


def compute(market: str, progress=lambda m: None) -> dict:
    progress("Loading the market's largest companies…")
    uni = data.universe(market, UNIVERSE)
    if uni.empty:
        raise RuntimeError("No companies available for this market.")
    progress(f"Downloading 5 years of prices for {len(uni)} companies…")
    closes, volumes = data.daily(list(uni["symbol"]))
    if closes.shape[1] < 40:
        raise RuntimeError(f"Only {closes.shape[1]} companies have enough price history.")
    progress("Computing features…")
    feats = methods.features(closes, volumes)
    symbols = list(closes.columns)
    try:
        fund = fundamentals_history.panels(fundamentals_history.load(symbols, progress), closes)
        feats.update({k: v for k, v in fund.items() if k in ("quality", "earnings_yield")})
        fund_cov = fundamentals_history.coverage(fund)
    except Exception:
        log.exception("fundamentals history unavailable")
        fund_cov = {"companies": 0, "from": None}
    try:
        sp = earnings.surprise_panel(symbols, closes, progress)
        if sp is not None:
            feats["surprise"] = sp
    except Exception:
        log.exception("earnings surprises unavailable")
    sample_dates = closes.index[252:]
    table = _ml_table(feats, sample_dates)
    cost = COST_ROUND_TRIP.get(market, DEFAULT_COST)
    out = {"status": "ready", "market": market, "as_of": str(closes.index[-1].date()), "universe": int(closes.shape[1]),
           "computed_at": pd.Timestamp.now(tz="UTC").isoformat(), "cost_round_trip": cost,
           "fundamentals_coverage": fund_cov, "earnings_coverage": int(feats["surprise"].notna().any().sum()) if "surprise" in feats else 0,
           "horizons": {}}
    for h in HORIZONS:
        progress(f"Running the {HORIZONS[h]} tournament…")
        out["horizons"][str(h)] = run_horizon(closes, feats, table, h, progress, cost)
    last = closes.index[-1]
    factor_now = {}
    for k in ("quality", "earnings_yield", "surprise", "mom_12_1", "vol_63"):
        if k in feats:
            row = feats[k].iloc[-1] if k != "surprise" else feats[k].ffill(limit=5).iloc[-1]
            factor_now[k] = {"value": row.to_dict(), "pct": row.rank(pct=True).to_dict()}
    out["factors_now"] = factor_now
    meta = uni.set_index("symbol")
    out["companies"] = {s: {"name": meta.at[s, "name"], "sector": meta.at[s, "sector"], "currency": meta.at[s, "currency"],
                            "price": float(closes[s].dropna().iloc[-1]),
                            "vol_63": float(feats["vol_63"][s].dropna().iloc[-1]) if feats["vol_63"][s].notna().any() else None}
                        for s in closes.columns if s in meta.index}
    return out


def get(market: str) -> dict:
    return jobs.run(f"tournament:{market}", lambda progress: compute(market, progress), TTL)
