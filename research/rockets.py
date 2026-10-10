"""Rockets & steady growers: a dedicated model for the market's biggest future winners.

Two targets, both relative to the market (so the model learns what distinguishes
winners rather than guessing the market's direction):

  rocket   ends the next 6 months (126 trading days) in the top 10% of the market by return
  grower   ends the next 12 months in the top 20% by return AND with a smaller-than-median
           worst drop along the way (a steady compounder, not a lottery ticket)

~45 inputs drawn from what research says separates big winners: momentum at
several horizons, 52-week-high breakouts and the number of new highs
(George & Hwang; O'Neil's CAN SLIM), volatility contraction, volume surges and
accumulation (up-volume vs down-volume), relative strength vs. the stock's sector
and the sector's own momentum, size and liquidity, return skew and lottery-like
jumps (which predict *lower* returns; Bali, Cakici & Whitelaw), point-in-time
quality, profitability, leverage, revenue/EPS growth and growth acceleration,
and earnings-surprise size and streak.

Validation is walk-forward with a horizon-length embargo. The headline numbers
are out-of-sample: the share of the model's top-5% picks that actually became
rockets/growers, versus the base rate (lift), and their average return versus
the market. Big winners are rare and volatile; most individual picks will miss.
"""

import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from research import backend, data, earnings, fundamentals_history, governance, intel, jobs, methods

log = logging.getLogger(__name__)

UNIVERSE = 500
FOLDS = 5
TTL = 12 * 3600
KINDS = {
    "rocket": {"horizon": 126, "step": 5, "top": 0.10, "label": "Potential rockets", "period": "next 6 months", "big_move": 0.5},
    "grower": {"horizon": 252, "step": 10, "top": 0.20, "label": "Steady growers", "period": "next 12 months", "big_move": 0.25},
}

FEATURE_LABELS = {
    "ret_5": "1-week return", "ret_10": "2-week return", "ret_21": "1-month return", "ret_42": "2-month return", "ret_63": "3-month return",
    "ret_126": "6-month return", "ret_252": "1-year return", "mom_12_1": "12-1 month momentum", "mom_6_1w": "6-month momentum",
    "vol_21": "1-month volatility", "vol_63": "3-month volatility", "vol_126": "6-month volatility", "vol_252": "1-year volatility",
    "vol_contraction": "Volatility contraction (1m vs 6m)", "dist_sma50": "Distance from 50-day avg", "dist_sma200": "Distance from 200-day avg",
    "sma50_200": "50-day vs 200-day avg", "high_52w": "Closeness to 52-week high", "low_52w": "Distance above 52-week low",
    "new_highs_63": "New 52-week highs (last 3 months)", "rsi_14": "RSI (14)", "macd_hist": "MACD momentum", "volume_trend": "Volume trend (20d vs 60d)",
    "volume_surge_5": "Volume surge (5d vs 60d)", "updown_volume_63": "Accumulation (up vs down volume)", "up_days_63": "Share of up days",
    "max_ret_21": "Biggest 1-day jump", "skew_63": "Return skew", "beta_126": "Beta", "sector_mom_126": "Sector momentum (6m)",
    "sector_rel_126": "Return vs sector (6m)", "dollar_volume": "Trading value (size/liquidity)", "illiquidity": "Illiquidity",
    "log_price": "Share price level", "quality": "Quality (profitability, safety)", "q_gpa": "Gross profit / assets", "q_roe": "Return on equity",
    "q_leverage": "Debt / assets", "earnings_yield": "Earnings yield (value)", "rev_growth": "Revenue growth (YoY)", "eps_growth": "Earnings growth (YoY)",
    "rev_accel": "Revenue growth acceleration", "net_margin": "Net margin", "surprise": "Latest earnings surprise", "surprise_streak": "Earnings beats (last 4)",
}
VOL_FEATURES = {"vol_21", "vol_63", "vol_126", "vol_252"}
NO_VALUE = {"illiquidity", "dollar_volume", "log_price", "rsi_14", "macd_hist", "beta_126", "skew_63", "quality", "q_gpa", "q_leverage", "earnings_yield", "high_52w", "vol_contraction", "updown_volume_63", "volume_trend", "volume_surge_5"}
PCT_FEATURES = {"ret_5", "ret_10", "ret_21", "ret_42", "ret_63", "ret_126", "ret_252", "mom_12_1", "mom_6_1w", "dist_sma50", "dist_sma200",
                "low_52w", "sector_mom_126", "sector_rel_126", "rev_growth", "eps_growth", "net_margin", "max_ret_21", "vol_63"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _surprise_streak(symbols: list[str], closes: pd.DataFrame) -> pd.DataFrame | None:
    with ThreadPoolExecutor(max_workers=6) as ex:
        hist = dict(zip(symbols, ex.map(earnings._surprises, symbols)))
    idx, cols = closes.index, {}
    for sym, ed in hist.items():
        if ed is None or ed.empty or sym not in closes:
            continue
        beats = (ed["Surprise(%)"] > 0).astype(float)
        streak = beats.rolling(4, min_periods=1).sum()
        known = pd.Series(streak.to_numpy(), index=ed.index + pd.Timedelta(days=1))
        known = known[~known.index.duplicated(keep="last")]
        cols[sym] = known.reindex(idx.union(known.index)).ffill().reindex(idx)
    return pd.DataFrame(cols).reindex(columns=closes.columns) if len(cols) >= 20 else None


def build_features(closes: pd.DataFrame, volumes: pd.DataFrame, sectors: pd.Series, progress=lambda m: None) -> dict[str, pd.DataFrame]:
    f = methods.features(closes, volumes)
    r = closes
    lr = np.log(r).diff()
    ret = r.pct_change()
    hi252 = r.rolling(252, min_periods=200).max()
    f["ret_10"] = r / r.shift(10) - 1
    f["ret_42"] = r / r.shift(42) - 1
    f["vol_126"] = lr.rolling(126, min_periods=90).std() * math.sqrt(252)
    f["vol_contraction"] = f["vol_21"] / f["vol_126"]
    f["new_highs_63"] = (r >= hi252 * 0.999).astype(float).where(hi252.notna()).rolling(63, min_periods=40).sum()
    f["volume_surge_5"] = volumes.rolling(5, min_periods=3).mean() / volumes.rolling(60, min_periods=45).mean()
    upv = volumes.where(ret > 0, 0).rolling(63, min_periods=45).sum()
    dnv = volumes.where(ret < 0, 0).rolling(63, min_periods=45).sum()
    f["updown_volume_63"] = upv / dnv.replace(0, np.nan)
    f["up_days_63"] = (ret > 0).astype(float).rolling(63, min_periods=45).mean()
    f["skew_63"] = lr.rolling(63, min_periods=45).skew()
    dv = (r * volumes).rolling(63, min_periods=45).mean()
    f["dollar_volume"] = np.log(dv.replace(0, np.nan))
    f["illiquidity"] = (ret.abs() / (r * volumes).replace(0, np.nan)).rolling(63, min_periods=45).mean()
    f["log_price"] = np.log(r)
    # Sector momentum and stock-vs-sector strength.
    r126 = f["ret_126"]
    sec = sectors.reindex(r.columns).fillna("Other")
    med = pd.DataFrame({s: r126[cols].median(axis=1) for s, cols in sec.groupby(sec).groups.items()})
    f["sector_mom_126"] = pd.DataFrame({c: med[sec[c]] for c in r.columns})
    f["sector_rel_126"] = r126 - f["sector_mom_126"]

    symbols = list(r.columns)
    try:
        fund = fundamentals_history.panels(fundamentals_history.load(symbols, progress), r)
        for k in ("quality", "q_gpa", "q_roe", "q_leverage", "earnings_yield", "rev_growth", "eps_growth", "rev_accel", "net_margin"):
            if k in fund:
                f[k] = fund[k]
    except Exception:
        log.exception("fundamentals unavailable for rockets")
    try:
        sp = earnings.surprise_panel(symbols, r, progress)
        if sp is not None:
            f["surprise"] = sp
        st = _surprise_streak(symbols, r)
        if st is not None:
            f["surprise_streak"] = st
    except Exception:
        log.exception("earnings unavailable for rockets")
    return {k: v for k, v in f.items() if k in FEATURE_LABELS}


def _table(feats: dict[str, pd.DataFrame], dates) -> pd.DataFrame:
    t = pd.concat({k: v.reindex(dates).rank(axis=1, pct=True).stack(future_stack=True) for k, v in feats.items()}, axis=1)
    t.index.names = ["date", "symbol"]
    return t


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _targets(closes: pd.DataFrame, dates, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = KINDS[kind]
    h = cfg["horizon"]
    fwd = (closes.shift(-h) / closes - 1).reindex(dates)
    thresh = fwd.quantile(1 - cfg["top"], axis=1)
    label = fwd.ge(thresh, axis=0).astype(float).where(fwd.notna())
    if kind == "grower":
        pos = {d: i for i, d in enumerate(closes.index)}
        arr = closes.to_numpy()
        dd = pd.DataFrame(np.nan, index=fwd.index, columns=closes.columns)
        for d in fwd.index:
            i = pos[d]
            if i + h >= len(arr):
                continue
            win = arr[i : i + h + 1]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # columns with no prices yet
                dd.loc[d] = np.nanmin(win / np.fmax.accumulate(win, axis=0), axis=0) - 1
        steady = dd.ge(dd.median(axis=1), axis=0)
        label = (label.astype(bool) & steady).astype(float).where(fwd.notna() & dd.notna())
    return label, fwd


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=200,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


# ---------------------------------------------------------------------------
# Train, validate, predict
# ---------------------------------------------------------------------------

def compute(market: str, kind: str, progress=lambda m: None) -> dict:
    cfg = KINDS[kind]
    h, step = cfg["horizon"], cfg["step"]
    progress(f"Loading the {UNIVERSE} largest companies…")
    uni = data.universe(market, UNIVERSE)
    if uni.empty:
        raise RuntimeError("No companies available for this market.")
    meta = uni.set_index("symbol")
    progress(f"Downloading 5 years of prices for {len(uni)} companies…")
    closes, volumes = data.daily(list(uni["symbol"]))
    if closes.shape[1] < 60:
        raise RuntimeError(f"Only {closes.shape[1]} companies have enough history.")
    progress("Building ~45 signals per stock…")
    feats = build_features(closes, volumes, meta["sector"], progress)
    cols = list(feats)

    sample = closes.index[252::step]
    if closes.index[-1] not in sample:
        sample = sample.append(pd.DatetimeIndex([closes.index[-1]]))
    label, fwd = _targets(closes, sample, kind)
    table = _table(feats, sample)
    y = label.stack(future_stack=True).reindex(table.index)
    fr = fwd.stack(future_stack=True).reindex(table.index)
    X = table.fillna(0.5)
    dates_lvl = table.index.get_level_values("date")

    labeled_dates = [d for d in sample if label.loc[d].notna().sum() >= 30]
    if len(labeled_dates) < 20:
        raise RuntimeError("Not enough history to validate this model.")
    val = labeled_dates[int(len(labeled_dates) * 0.4):]
    folds = np.array_split(np.array(val), FOLDS)
    pos = {d: i for i, d in enumerate(closes.index)}

    progress("Training walk-forward and testing on unseen periods…")
    oos = pd.Series(np.nan, index=table.index)
    last_model, last_idx = None, None
    for f in folds:
        cutoff = closes.index[max(0, pos[pd.Timestamp(f[0])] - h - 1)]
        tr = table.index[(dates_lvl <= cutoff) & y.notna().to_numpy()]
        te = table.index[dates_lvl.isin(f) & y.notna().to_numpy()]
        if len(tr) < 5000 or len(te) == 0:
            continue
        m = _model().fit(X.loc[tr], y.loc[tr])
        oos.loc[te] = m.predict_proba(X.loc[te])[:, 1]
        last_model, last_idx = m, te
    ev = oos.dropna()
    if ev.empty:
        raise RuntimeError("Not enough history to validate this model.")
    df = pd.DataFrame({"p": ev, "y": y.loc[ev.index], "r": fr.loc[ev.index]}).reset_index()
    df["pct"] = df.groupby("date")["p"].rank(pct=True)
    top5 = df[df["pct"] >= 0.95]
    top10 = df[df["pct"] >= 0.90]
    base = float(df["y"].mean())
    big = cfg["big_move"]
    every = max(1, round(h / step))
    per_date = top5.groupby("date")["y"].mean()
    independent = per_date.iloc[::every]
    deciles = []
    df["decile"] = np.ceil(df["pct"] * 10).clip(1, 10).astype(int)
    for d, g in df.groupby("decile"):
        deciles.append({"decile": int(d), "hit_rate": float(g["y"].mean()), "avg_return": float(g["r"].mean()),
                        "big_move_rate": float((g["r"] >= big).mean()), "observations": int(len(g))})
    valid = {
        "period": [str(pd.Timestamp(val[0]).date()), str(pd.Timestamp(val[-1]).date())],
        "predictions_tested": int(len(df)),
        "independent_periods": int(len(independent)),
        "auc": float(roc_auc_score(df["y"], df["p"])) if df["y"].nunique() > 1 else None,
        "base_rate": base,
        "top5_hit_rate": float(top5["y"].mean()),
        "top10_hit_rate": float(top10["y"].mean()),
        "lift_top5": float(top5["y"].mean() / base) if base else None,
        "top5_avg_return": float(top5["r"].mean()),
        "all_avg_return": float(df["r"].mean()),
        "top5_median_return": float(top5["r"].median()),
        "top5_big_move_rate": float((top5["r"] >= big).mean()),
        "all_big_move_rate": float((df["r"] >= big).mean()),
        "top5_crash_rate": float((top5["r"] <= -0.25).mean()),
        "all_crash_rate": float((df["r"] <= -0.25).mean()),
        "deciles": deciles,
    }
    # Is this skill, or just picking volatile stocks? Re-measure the lift inside groups of
    # stocks with similar volatility (thirds of the market by 3-month volatility).
    volr = feats["vol_63"].reindex(sorted(set(df["date"]))).rank(axis=1, pct=True).stack(future_stack=True)
    df["vol_third"] = np.ceil(volr.reindex(pd.MultiIndex.from_frame(df[["date", "symbol"]])).to_numpy() * 3).clip(1, 3)
    lifts = []
    for t, g in df.dropna(subset=["vol_third"]).groupby("vol_third"):
        gp = g.groupby("date")["p"].rank(pct=True)
        top = g[gp >= 0.8]
        if len(top) and g["y"].mean() > 0:
            lifts.append({"third": int(t), "lift": float(top["y"].mean() / g["y"].mean()), "base_rate": float(g["y"].mean())})
    valid["lift_within_volatility"] = lifts
    valid["skill_beyond_volatility"] = float(np.mean([x["lift"] for x in lifts])) if lifts else None

    lift = valid["lift_top5"] or 0
    excess = valid["top5_avg_return"] - valid["all_avg_return"]
    if lift >= 1.5 and excess > 0:
        level, text = "meaningful", f"The model's top 5% picks became {cfg['label'].lower()} {lift:.1f}x as often as a random stock, out-of-sample."
    elif lift >= 1.2 and excess > 0:
        level, text = "modest", f"The top picks became {cfg['label'].lower()} {lift:.1f}x as often as a random stock: a real but modest edge."
    elif lift < 0.9:
        level, text = "negative", "The model's top picks did worse than random out-of-sample. Don't use this list."
    else:
        level, text = "none", "The model did not find big winners more often than chance out-of-sample."
    notes = []
    sk = valid["skill_beyond_volatility"]
    if sk is not None and sk < 1.15 and level in ("meaningful", "modest"):
        notes.append(f"Among stocks of similar volatility the lift is only {sk:.1f}x, so much of the edge comes from which volatility group it favours rather than from picking within it.")
    elif sk is not None and level in ("meaningful", "modest"):
        notes.append(f"It still finds winners {sk:.1f}x as often among stocks of similar volatility, so it is not only picking risky stocks.")
    if valid["top5_crash_rate"] > 1.5 * valid["all_crash_rate"]:
        notes.append(f"Its picks also fell 25%+ {valid['top5_crash_rate']:.0%} of the time vs {valid['all_crash_rate']:.0%} for the market: keep positions small.")
    if valid["independent_periods"] < 4:
        if level == "meaningful":
            level = "modest"
        notes.append(f"Tested on only {valid['independent_periods']} non-overlapping periods, so treat these figures as rough.")
    valid["verdict_level"], valid["verdict"], valid["notes"] = level, text, notes

    progress("Ranking today's stocks…")
    importance = []
    if last_model is not None:
        sub = last_idx[: min(len(last_idx), 6000)]
        pi = permutation_importance(last_model, X.loc[sub], y.loc[sub], n_repeats=2, random_state=0, scoring="roc_auc")
        importance = sorted(({"feature": c, "label": FEATURE_LABELS[c], "importance": float(v)} for c, v in zip(cols, pi.importances_mean)),
                            key=lambda x: x["importance"], reverse=True)
    important = [x["feature"] for x in importance[:12]]

    train_all = table.index[y.notna().to_numpy()]
    final = _model().fit(X.loc[train_all], y.loc[train_all])
    last = closes.index[-1]
    now_idx = table.index[dates_lvl == last]
    probs = pd.Series(final.predict_proba(X.loc[now_idx])[:, 1], index=now_idx.get_level_values("symbol"))
    pct = probs.rank(pct=True)
    liquid = feats["dollar_volume"].iloc[-1]
    ok = liquid >= liquid.quantile(0.10)  # skip the least-traded 10%: too easy to get trapped in
    by_decile = {d["decile"]: d for d in deciles}
    ranked = [s for s in pct.sort_values(ascending=False).index if ok.get(s, False)]

    def drivers(sym: str) -> list[str]:
        out = []
        ranks = table.loc[(last, sym)] if (last, sym) in table.index else None
        if ranks is None:
            return out
        cand = sorted(((k, ranks[k]) for k in important if pd.notna(ranks[k])), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
        for k, rk in cand[:2]:
            v = feats[k].at[last, sym] if sym in feats[k].columns else None
            if v is None or pd.isna(v) or k in NO_VALUE:
                shown = ""
            elif k in VOL_FEATURES:
                shown = f" {v:.0%}"
            elif k == "surprise":
                shown = f" {v:+.0f}%"
            elif k in ("new_highs_63", "surprise_streak"):
                shown = f" {v:.0f}"
            elif k in PCT_FEATURES or k in ("q_roe",):
                shown = f" {v:+.0%}"
            else:
                shown = f" {v:.2f}"
            tier = "top" if rk >= 0.5 else "bottom"
            out.append(f"{FEATURE_LABELS[k]}{shown}: {tier} {max(1, round((1 - rk if rk >= 0.5 else rk) * 100))}%")
        return out

    picks = []
    for sym in ranked[:40]:
        name = meta.at[sym, "name"] if sym in meta.index else sym
        d = int(min(10, max(1, math.ceil(pct[sym] * 10))))
        gv = governance.assess(sym, name, market) if market == "in" else {"flags": []}
        q = backend._get(f"/api/stocks/quality/{sym}", ttl=24 * 3600, timeout=60) or {}
        o = intel.latest([sym]).get(sym)
        picks.append({
            "symbol": sym, "name": name, "sector": meta.at[sym, "sector"] if sym in meta.index else None,
            "currency": meta.at[sym, "currency"] if sym in meta.index else None, "price": float(closes[sym].dropna().iloc[-1]),
            "probability": float(probs[sym]), "percentile": float(pct[sym]),
            "hist_hit_rate": by_decile.get(d, {}).get("hit_rate"), "hist_avg_return": by_decile.get(d, {}).get("avg_return"),
            "return_6m": _v(feats["ret_126"], last, sym), "return_1y": _v(feats["ret_252"], last, sym), "volatility": _v(feats["vol_63"], last, sym),
            "drivers": drivers(sym), "red_flags": list(q.get("red_flags") or []) + list(gv.get("flags", [])),
            "ai": None if not o else {"score": o["score"], "label": o["label"]},
        })
        if len(picks) == 25:
            break

    return {
        "status": "ready", "market": market, "kind": kind, "label": cfg["label"], "period": cfg["period"],
        "definition": (f"Top 10% of the market by return over the {cfg['period']}" if kind == "rocket"
                       else f"Top 20% by return over the {cfg['period']} with a smaller-than-median worst drop along the way"),
        "big_move": cfg["big_move"], "as_of": str(last.date()), "universe": int(closes.shape[1]), "features": len(cols),
        "validation": valid, "feature_importance": importance[:15], "picks": picks,
        "warning": "Big winners are rare and volatile: even with a real edge, most individual picks will not become rockets, and some will fall sharply. Keep each position small and diversified.",
    }


def _v(frame: pd.DataFrame, d, sym) -> float | None:
    v = frame.at[d, sym] if sym in frame.columns else None
    return None if v is None or pd.isna(v) else float(v)


def get(market: str, kind: str) -> dict:
    if kind not in KINDS:
        raise KeyError(kind)
    return jobs.run(f"rockets:{market}:{kind}", lambda p: compute(market, kind, p), TTL)
