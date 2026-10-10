"""Machine-learning forecasts of which stocks will outperform over the next 1 or 3 months.

How it works
------------
1. Universe: the ~300 most-traded stocks in a market, 5 years of daily prices,
   plus the market's benchmark index.
2. Features (all computed only from data available on that day): returns over
   1 week to 1 year, 12-1 month momentum, volatility, trend (distance from
   50/200-day averages), RSI, MACD, distance from 52-week high/low, volume
   surges, biggest one-day jump, beta and return relative to the index, plus
   market-wide context (index trend, breadth). Stock features are converted to
   percentile ranks within each day so the model learns "relatively strong
   vs. weak", which holds up across bull and bear markets.
3. Target: did the stock beat the median stock over the next N trading days?
   (Predicting whether the whole market rises is mostly noise; ranking stocks
   against each other is where evidence of predictability exists.)
4. Model: gradient-boosted trees (scikit-learn), deliberately shallow and
   heavily regularised to limit overfitting.
5. Honest validation: walk-forward over the last two years. Each test window
   is predicted by a model trained only on earlier data, with an N-day gap so
   no future information leaks in. The UI shows these out-of-sample results,
   and each prediction's "chance of rising" is the historical rate for its
   decile, not the model's raw output.

Known bias: the universe is today's listed companies, so firms that collapsed
and delisted are missing (survivorship bias). Real-world results will be
somewhat worse than the validation figures.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from market_data.markets import MARKETS_BY_CODE
from market_data.services import indicators as ind
from market_data.services import market_data, ranking

log = logging.getLogger(__name__)

HORIZONS = {21: "1 month", 63: "3 months"}
UNIVERSE = 300
MIN_STOCKS = 40
STEP = 5  # sample one row per stock per week to limit overlapping labels
FOLDS = 8
VALIDATION_DAYS = 504  # ~2 years of walk-forward testing
RESULT_TTL = 12 * 3600
TOP_N = 25

FEATURE_LABELS = {
    "ret_5": "1-week return",
    "ret_21": "1-month return",
    "ret_63": "3-month return",
    "ret_126": "6-month return",
    "ret_252": "1-year return",
    "mom_12_1": "12-month momentum (ex. last month)",
    "vol_21": "1-month volatility",
    "vol_63": "3-month volatility",
    "vol_ratio": "Volatility rising vs. falling",
    "dist_sma50": "Distance from 50-day average",
    "dist_sma200": "Distance from 200-day average",
    "sma50_200": "50-day vs 200-day average",
    "rsi_14": "RSI (14)",
    "macd": "MACD momentum",
    "dist_high_252": "Distance from 52-week high",
    "dist_low_252": "Distance from 52-week low",
    "volume_trend": "Volume surge (20d vs 60d)",
    "max_ret_21": "Biggest 1-day jump (last month)",
    "rel_63": "3-month return vs index",
    "beta_126": "Beta to index",
    "mkt_ret_21": "Index 1-month return",
    "mkt_ret_63": "Index 3-month return",
    "mkt_above_200": "Index above its 200-day average",
    "mkt_vol_21": "Index volatility",
    "breadth": "Share of stocks in uptrends",
}
MARKET_FEATURES = ["mkt_ret_21", "mkt_ret_63", "mkt_above_200", "mkt_vol_21", "breadth"]
PCT_FEATURES = {"ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1", "vol_21", "vol_63", "dist_sma50", "dist_sma200",
                "sma50_200", "dist_high_252", "dist_low_252", "max_ret_21", "rel_63", "mkt_ret_21", "mkt_ret_63", "mkt_vol_21", "breadth"}


class NotEnoughData(Exception):
    pass


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------


def load_prices(code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.DataFrame]:
    df, _ = ranking.market_frame(code)
    meta = df[df["liquid"]].sort_values("turnover", ascending=False).head(UNIVERSE).set_index("symbol")
    history: dict[str, pd.DataFrame] = {}
    symbols = list(meta.index)
    for i in range(0, len(symbols), 100):
        history.update(market_data.get_history(symbols[i : i + 100]))
    if len(history) < MIN_STOCKS:
        raise NotEnoughData(f"Only {len(history)} stocks have price history in this market; at least {MIN_STOCKS} are needed.")

    closes = pd.DataFrame({s: f["Close"] for s, f in history.items()}).sort_index()
    volumes = pd.DataFrame({s: f["Volume"] for s, f in history.items()}).reindex(closes.index)
    # Keep days on which most of the market traded; carry prices over short gaps (holidays, halts).
    closes = closes[closes.notna().mean(axis=1) >= 0.5].ffill(limit=3)
    volumes = volumes.reindex(closes.index)

    index = None
    m = MARKETS_BY_CODE[code]
    if m.index:
        idx = market_data.get_history([m.index]).get(m.index.upper())
        if idx is not None:
            index = idx["Close"].reindex(closes.index).ffill(limit=5)
    return closes, volumes, index, meta.loc[[s for s in meta.index if s in closes.columns]]


def features(closes: pd.DataFrame, volumes: pd.DataFrame, index: pd.Series | None) -> dict[str, pd.DataFrame]:
    """Each feature as a (date x symbol) frame, using only past data at every date."""
    r = closes
    lr = np.log(r).diff()
    sma50, sma200 = r.rolling(50, min_periods=40).mean(), r.rolling(200, min_periods=150).mean()
    f: dict[str, pd.DataFrame] = {}
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = r / r.shift(n) - 1
    f["mom_12_1"] = r.shift(21) / r.shift(252) - 1
    f["vol_21"] = lr.rolling(21, min_periods=15).std() * np.sqrt(252)
    f["vol_63"] = lr.rolling(63, min_periods=45).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["dist_sma50"] = r / sma50 - 1
    f["dist_sma200"] = r / sma200 - 1
    f["sma50_200"] = sma50 / sma200 - 1
    f["rsi_14"] = r.apply(ind.rsi)
    f["macd"] = r.apply(lambda c: ind.macd(c)[2]) / r
    f["dist_high_252"] = r / r.rolling(252, min_periods=200).max() - 1
    f["dist_low_252"] = r / r.rolling(252, min_periods=200).min() - 1
    f["volume_trend"] = volumes.rolling(20, min_periods=15).mean() / volumes.rolling(60, min_periods=45).mean()
    f["max_ret_21"] = lr.rolling(21, min_periods=15).max()

    above = (r > sma200).where(sma200.notna())
    breadth = above.mean(axis=1)
    if index is not None:
        ir = np.log(index).diff()
        idx_ret_63 = index / index.shift(63) - 1
        f["rel_63"] = f["ret_63"].sub(idx_ret_63, axis=0)
        cov = lr.mul(ir, axis=0).rolling(126, min_periods=90).mean() - lr.rolling(126, min_periods=90).mean().mul(ir.rolling(126, min_periods=90).mean(), axis=0)
        f["beta_126"] = cov.div(ir.rolling(126, min_periods=90).var(), axis=0)
        market = {
            "mkt_ret_21": index / index.shift(21) - 1,
            "mkt_ret_63": idx_ret_63,
            "mkt_above_200": (index > index.rolling(200, min_periods=150).mean()).astype(float),
            "mkt_vol_21": ir.rolling(21, min_periods=15).std() * np.sqrt(252),
        }
    else:
        market = {}
    market["breadth"] = breadth
    for name, series in market.items():
        f[name] = pd.DataFrame(np.repeat(series.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    return f


def build_table(closes: pd.DataFrame, feats: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Long table: one row per (date, symbol) with percentile-ranked stock features,
    raw market features, the forward return and the outperformance label."""
    stock_cols = [k for k in feats if k not in MARKET_FEATURES]
    cols = {k: feats[k].rank(axis=1, pct=True) for k in stock_cols}
    cols.update({k: feats[k] for k in MARKET_FEATURES if k in feats})
    fwd = closes.shift(-horizon) / closes - 1
    cols["fwd_ret"] = fwd
    cols["raw_" + "ret_252"] = feats["ret_252"]

    warm = 252
    dates = closes.index[warm::STEP]
    if closes.index[-1] not in dates:
        dates = dates.append(pd.DatetimeIndex([closes.index[-1]]))
    table = pd.concat({k: v.loc[dates].stack(future_stack=True) for k, v in cols.items()}, axis=1)
    table.index.names = ["date", "symbol"]
    table = table.reset_index()
    # Need most features present to be useful.
    feat_cols = [k for k in cols if k not in ("fwd_ret", "raw_ret_252")]
    table = table[table[feat_cols].notna().mean(axis=1) >= 0.7]
    median = table.groupby("date")["fwd_ret"].transform("median")
    table["outperform"] = (table["fwd_ret"] > median).astype(float).where(table["fwd_ret"].notna())
    return table


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=300,
        l2_regularization=1.0,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def _deciles(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["prob"].transform(lambda p: np.ceil(p.rank(pct=True) * 10).clip(1, 10)).astype(int)


def walk_forward(table: pd.DataFrame, all_dates: pd.DatetimeIndex, feat_cols: list[str], horizon: int) -> tuple[pd.DataFrame, HistGradientBoostingClassifier, pd.DataFrame]:
    labeled = table[table["outperform"].notna()]
    dates = np.sort(labeled["date"].unique())
    test_dates = dates[dates >= all_dates[max(0, len(all_dates) - horizon - VALIDATION_DAYS)]]
    if len(test_dates) < FOLDS * 2:
        raise NotEnoughData("Not enough history to validate the model.")
    pos = {d: i for i, d in enumerate(all_dates)}

    oos, last_model, last_test = [], None, None
    for chunk in np.array_split(test_dates, FOLDS):
        start = pos[pd.Timestamp(chunk[0])]
        cutoff = all_dates[max(0, start - horizon - 1)]  # labels of training rows must be known before the test window
        train = labeled[labeled["date"] <= cutoff]
        test = labeled[labeled["date"].isin(chunk)]
        if len(train) < 2000 or test.empty:
            continue
        model = _model().fit(train[feat_cols], train["outperform"])
        out = test[["date", "symbol", "fwd_ret", "outperform"]].copy()
        out["prob"] = model.predict_proba(test[feat_cols])[:, 1]
        oos.append(out)
        last_model, last_test = model, test
    if not oos:
        raise NotEnoughData("Not enough history to validate the model.")
    return pd.concat(oos), last_model, last_test


def evaluate(oos: pd.DataFrame) -> dict:
    oos = oos.copy()
    oos["decile"] = _deciles(oos)
    ics = [spearmanr(g["prob"], g["fwd_ret"]).statistic for _, g in oos.groupby("date") if len(g) > 10]
    deciles = []
    for d, g in oos.groupby("decile"):
        deciles.append({
            "decile": int(d),
            "observations": int(len(g)),
            "avg_return": float(g["fwd_ret"].mean()),
            "median_return": float(g["fwd_ret"].median()),
            "prob_up": float((g["fwd_ret"] > 0).mean()),
            "prob_outperform": float(g["outperform"].mean()),
        })
    top, bottom = deciles[-1], deciles[0]
    auc = float(roc_auc_score(oos["outperform"], oos["prob"]))
    ic = float(np.nanmean(ics)) if ics else None
    spread = top["avg_return"] - bottom["avg_return"]
    monotonic = float(spearmanr([d["decile"] for d in deciles], [d["avg_return"] for d in deciles]).statistic)

    if auc >= 0.56 and spread > 0:
        level, text = "meaningful", "The model showed a meaningful edge in testing: its top-ranked stocks clearly beat its bottom-ranked ones. Still, expect many individual misses."
    elif auc >= 0.52 and spread > 0:
        level, text = "modest", "The model showed a modest edge in testing: better than a coin flip on average, but individual predictions are often wrong. Use it to tilt, not to bet."
    elif auc < 0.48:
        level, text = "negative", "In testing the model did WORSE than chance here: patterns from earlier years reversed in this market recently. Do not use these lists to pick stocks."
    else:
        level, text = "none", "The model showed no reliable edge in testing for this market and horizon. Treat these lists as no better than chance."
    return {
        "period_start": str(pd.Timestamp(oos["date"].min()).date()),
        "period_end": str(pd.Timestamp(oos["date"].max()).date()),
        "predictions_tested": int(len(oos)),
        "auc": auc,
        "accuracy": float(((oos["prob"] > 0.5) == (oos["outperform"] == 1)).mean()),
        "information_coefficient": ic,
        "baseline_prob_up": float((oos["fwd_ret"] > 0).mean()),
        "top_decile": top,
        "bottom_decile": bottom,
        "top_minus_bottom_return": spread,
        "decile_monotonicity": monotonic,
        "deciles": deciles,
        "verdict_level": level,
        "verdict": text,
    }


# ---------------------------------------------------------------------------
# Training + prediction
# ---------------------------------------------------------------------------


def _drivers(row: pd.Series, raw: dict[str, float], important: list[str]) -> list[str]:
    """Plain-English reasons: the important features where this stock is most extreme vs. its peers."""
    scored = [(k, row[k]) for k in important if k in row and pd.notna(row[k]) and k not in MARKET_FEATURES]
    scored.sort(key=lambda kv: abs(kv[1] - 0.5), reverse=True)
    out = []
    for k, pct in scored[:3]:
        tier = "top" if pct >= 0.5 else "bottom"
        share = max(1, round((1 - pct if tier == "top" else pct) * 100))
        value = raw.get(k)
        shown = ""
        if value is not None and pd.notna(value):
            shown = f" ({value:+.0%})" if k in PCT_FEATURES else f" ({value:.2f})"
        out.append(f"{FEATURE_LABELS[k]}{shown}: {tier} {share}% of stocks")
    return out


def train_and_predict(code: str, horizon: int, progress=lambda msg: None) -> dict:
    t0 = time.time()
    progress("Downloading 5 years of prices for the most-traded stocks…")
    closes, volumes, index, meta = load_prices(code)
    progress(f"Computing {len(FEATURE_LABELS)} features for {closes.shape[1]} stocks…")
    feats = features(closes, volumes, index)
    table = build_table(closes, feats, horizon)
    feat_cols = [c for c in FEATURE_LABELS if c in table.columns]

    progress("Testing the model walk-forward on the last 2 years…")
    oos, last_model, last_test = walk_forward(table, closes.index, feat_cols, horizon)
    validation = evaluate(oos)

    imp = permutation_importance(last_model, last_test[feat_cols], last_test["outperform"], n_repeats=3, random_state=0, scoring="roc_auc")
    importance = sorted(
        ({"feature": c, "label": FEATURE_LABELS[c], "importance": float(v)} for c, v in zip(feat_cols, imp.importances_mean)),
        key=lambda x: x["importance"],
        reverse=True,
    )
    important = [x["feature"] for x in importance[:8]]

    progress("Training the final model and ranking today's stocks…")
    labeled = table[table["outperform"].notna()]
    model = _model().fit(labeled[feat_cols], labeled["outperform"])
    latest_date = closes.index[-1]
    now = table[table["date"] == latest_date].copy()
    now["prob"] = model.predict_proba(now[feat_cols])[:, 1]
    now["decile"] = np.ceil(now["prob"].rank(pct=True) * 10).clip(1, 10).astype(int)
    by_decile = {d["decile"]: d for d in validation["deciles"]}

    def row(r: pd.Series) -> dict:
        s = r["symbol"]
        m = meta.loc[s]
        hist = by_decile.get(int(r["decile"]), {})
        raw = {k: feats[k].at[latest_date, s] for k in important if k in feats}
        return {
            "symbol": s,
            "name": m["name"],
            "sector": m["sector"],
            "currency": m["currency"],
            "price": float(closes.at[latest_date, s]),
            "return_1y": None if pd.isna(r["raw_ret_252"]) else float(r["raw_ret_252"]),
            "probability_outperform": float(r["prob"]),
            "decile": int(r["decile"]),
            "historical_prob_up": hist.get("prob_up"),
            "historical_avg_return": hist.get("avg_return"),
            "drivers": _drivers(r, raw, important),
        }

    ranked = now.sort_values("prob", ascending=False)
    return {
        "status": "ready",
        "market": code,
        "market_name": MARKETS_BY_CODE[code].name,
        "horizon_days": horizon,
        "horizon_label": HORIZONS[horizon],
        "as_of": str(latest_date.date()),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_seconds": round(time.time() - t0, 1),
        "universe": int(closes.shape[1]),
        "training_rows": int(len(labeled)),
        "features": len(feat_cols),
        "validation": validation,
        "feature_importance": importance,
        "bullish": [row(r) for _, r in ranked.head(TOP_N).iterrows()],
        "bearish": [row(r) for _, r in ranked.tail(TOP_N).iloc[::-1].iterrows()],
        "all": [
            {"symbol": r["symbol"], "name": meta.loc[r["symbol"], "name"], "sector": meta.loc[r["symbol"], "sector"],
             "probability_outperform": float(r["prob"]), "decile": int(r["decile"])}
            for _, r in ranked.iterrows()
        ],
        "caveats": [
            "Survivorship bias: only companies listed today are included, so the test results are somewhat flattering.",
            "Validation used weekly snapshots with overlapping holding periods, so the effective number of independent tests is smaller than it looks.",
            "The model uses prices and volumes only. It knows nothing about news, earnings surprises, management or regulation; check those before acting.",
        ],
    }


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


@dataclass
class Job:
    status: str = "training"
    message: str = "Starting…"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict | None = None
    error: str | None = None


_jobs: dict[tuple[str, int], Job] = {}
_jobs_lock = threading.Lock()


def _run(key: tuple[str, int]) -> None:
    job = _jobs[key]

    def progress(msg: str) -> None:
        job.message = msg

    try:
        job.result = train_and_predict(key[0], key[1], progress)
        job.status = "ready"
    except (NotEnoughData, ranking.NoData) as exc:
        job.status, job.error = "error", str(exc)
    except Exception as exc:  # surface unexpected failures instead of hanging the UI
        log.exception("Prediction job %s failed", key)
        job.status, job.error = "error", f"Training failed: {exc}"
    job.finished = time.time()


def get(code: str, horizon: int) -> dict:
    """Return the cached prediction, or start training and report progress."""
    key = (code, horizon)
    with _jobs_lock:
        job = _jobs.get(key)
        stale = job is not None and job.finished is not None and (
            (job.status == "ready" and time.time() - job.finished > RESULT_TTL) or (job.status == "error" and time.time() - job.finished > 300)
        )
        if job is None or stale:
            job = _jobs[key] = Job()
            threading.Thread(target=_run, args=(key,), daemon=True, name=f"predict-{code}-{horizon}").start()
    if job.status == "ready":
        return job.result
    if job.status == "error":
        return {"status": "error", "message": job.error}
    return {"status": "training", "message": job.message, "elapsed_seconds": round(time.time() - job.started)}
