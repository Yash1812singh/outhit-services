"""Machine-learning forecasts of which stocks will outperform over the next trading day, 1 month or 3 months.

How it works
------------
1. Universe: the ~300 most-traded stocks in a market, 5 years of daily prices,
   plus the market's benchmark index.
2. Features (all computed only from data available on that day): returns over
   1 week to 1 year, 12-1 month momentum, volatility, trend (distance from
   50/200-day averages), RSI, MACD, distance from 52-week high/low, volume
   surges, biggest one-day jump, beta and return relative to the index, plus
   market-wide context (index trend, breadth). The next-day model adds
   short-term inputs: 1-3 day returns (short-term reversal), the overnight gap,
   the intraday move, where the close sits in the day's range, range expansion
   and volume shocks. Stock features are converted to
   percentile ranks within each day so the model learns "relatively strong
   vs. weak", which holds up across bull and bear markets.
3. Target: did the stock beat the median stock over the next N trading days?
   (Predicting whether the whole market rises is mostly noise; ranking stocks
   against each other is where evidence of predictability exists.)
4. Model: an equal blend of gradient-boosted trees (deliberately shallow and
   heavily regularised) and a strongly regularised logistic regression. The
   trees catch interactions, the linear model keeps steady ranking signals
   (momentum, low volatility) from being drowned out by noise; in walk-forward
   tests the blend ranked stocks more consistently than either alone.
5. Honest validation: walk-forward over the last two years. Each test window
   is predicted by a model trained only on earlier data, with an N-day gap so
   no future information leaks in. For the next-day model the validation also
   reports the return you could actually have captured by buying at the next
   open, since the close-to-close move is not tradable after the fact.
   The UI shows these out-of-sample results,
   and each prediction's "chance of rising" is the historical rate for its
   decile, not the model's raw output.

6. Learning from outcomes: every published prediction is stored, and once its
   horizon has passed it is scored against what really happened
   (services/learning.py). Six model variants (trees, linear, blend; equal or
   recent-weighted history) compete; the one with the best recent evidence,
   from walk-forward windows and from scored live predictions, is used. The
   reported test results include this adaptation, since each test window used
   only the evidence available before it. A background loop keeps predicting
   and scoring every day, so the evidence keeps growing.

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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_data.markets import MARKETS_BY_CODE
from market_data.services import indicators as ind
from market_data.services import learning, market_data, ranking

log = logging.getLogger(__name__)

HORIZONS = {1: "next trading day", 21: "1 month", 63: "3 months"}
UNIVERSE = 300
MIN_STOCKS = 40
STEP = 5  # sample one row per stock per week to limit overlapping labels (daily for the next-day model)
FOLDS = 8
VALIDATION_DAYS = 504  # ~2 years of walk-forward testing
RESULT_TTL = 12 * 3600
RESULT_TTL_NEXT_DAY = 2 * 3600  # the next-day list goes stale as soon as a new session closes
TOP_N = 25
RECENT_HALF_LIFE = 252  # trading days: a year-old row counts half as much as today's

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
SHORT_LABELS = {
    "ret_1": "Last day's return",
    "ret_2": "2-day return",
    "ret_3": "3-day return",
    "gap_1": "Overnight gap (open vs previous close)",
    "intraday_1": "Intraday move (close vs open)",
    "clv_1": "Close position in the day's range",
    "range_ratio": "Day's range vs 20-day average",
    "volume_shock": "Volume vs 20-day average",
    "dist_sma10": "Distance from 10-day average",
    "vol_5": "1-week volatility",
    "mkt_ret_1": "Index last-day return",
    "mkt_ret_5": "Index 1-week return",
}
FEATURE_LABELS.update(SHORT_LABELS)
MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_ret_63", "mkt_above_200", "mkt_vol_21", "breadth"]
PCT_FEATURES = {"ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1", "vol_21", "vol_63", "dist_sma50", "dist_sma200",
                "sma50_200", "dist_high_252", "dist_low_252", "max_ret_21", "rel_63", "mkt_ret_21", "mkt_ret_63", "mkt_vol_21", "breadth",
                "ret_1", "ret_2", "ret_3", "gap_1", "intraday_1", "dist_sma10", "vol_5", "mkt_ret_1", "mkt_ret_5"}


# "All asset classes": liquid US-listed ETFs (one trading calendar) covering stocks, sectors,
# regions, bonds of every duration and credit quality, gold and commodities, property and
# currencies, plus the largest cryptocurrencies. Ranked against each other like stocks.
ASSETS_CODE = "assets"
ASSETS_NAME = "All asset classes"
ASSETS_INDEX = "ACWI"  # MSCI All Country World: the market backdrop for every asset
ASSET_UNIVERSE: dict[str, list[tuple[str, str]]] = {
    "US stocks": [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000 small caps"), ("MDY", "S&P mid caps"),
                  ("VTV", "US value"), ("VUG", "US growth"), ("USMV", "US minimum volatility"), ("RSP", "S&P 500 equal weight")],
    "Sectors": [("XLK", "Technology"), ("XLF", "Financials"), ("XLE", "Energy"), ("XLV", "Health care"), ("XLI", "Industrials"),
                ("XLY", "Consumer discretionary"), ("XLP", "Consumer staples"), ("XLU", "Utilities"), ("XLB", "Materials"),
                ("XLC", "Communication services"), ("SMH", "Semiconductors"), ("IBB", "Biotech"), ("ITA", "Aerospace & defence")],
    "World stocks": [("EFA", "Developed markets ex-US"), ("EEM", "Emerging markets"), ("VGK", "Europe"), ("EWJ", "Japan"),
                     ("INDA", "India"), ("MCHI", "China"), ("EWZ", "Brazil"), ("EWY", "South Korea"), ("EWT", "Taiwan")],
    "Bonds": [("SHY", "US Treasuries 1-3y"), ("IEI", "US Treasuries 3-7y"), ("IEF", "US Treasuries 7-10y"), ("TLT", "US Treasuries 20y+"),
              ("TIP", "Inflation-protected Treasuries"), ("LQD", "Investment-grade corporate"), ("HYG", "High-yield corporate"),
              ("EMB", "Emerging-market bonds"), ("MUB", "Municipal bonds"), ("BNDX", "International bonds"), ("AGG", "US aggregate bonds")],
    "Gold & commodities": [("GLD", "Gold"), ("SLV", "Silver"), ("USO", "Crude oil"), ("DBC", "Commodity basket"), ("DBA", "Agriculture"),
                           ("UNG", "Natural gas"), ("CPER", "Copper")],
    "Real estate": [("VNQ", "US REITs"), ("VNQI", "International REITs")],
    "Currencies": [("UUP", "US dollar"), ("FXE", "Euro"), ("FXY", "Japanese yen")],
    "Crypto": [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"), ("XRP-USD", "XRP"), ("BNB-USD", "BNB")],
}


def market_name(code: str) -> str:
    return ASSETS_NAME if code == ASSETS_CODE else MARKETS_BY_CODE[code].name


class NotEnoughData(Exception):
    pass


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------


def _asset_meta() -> pd.DataFrame:
    rows = [{"symbol": sym, "name": name, "sector": group, "currency": "USD"} for group, items in ASSET_UNIVERSE.items() for sym, name in items]
    return pd.DataFrame(rows).set_index("symbol")


def load_prices(code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.DataFrame, dict[str, pd.DataFrame]]:
    if code == ASSETS_CODE:
        meta = _asset_meta()
    else:
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
    # Open/high/low are not carried over gaps: a halted day has no real bar.
    bars = {
        k.lower(): pd.DataFrame({s: f[k] for s, f in history.items() if k in f}).reindex(index=closes.index, columns=closes.columns)
        for k in ("Open", "High", "Low")
    }

    index = None
    index_symbol = ASSETS_INDEX if code == ASSETS_CODE else MARKETS_BY_CODE[code].index
    if index_symbol:
        idx = market_data.get_history([index_symbol]).get(index_symbol.upper())
        if idx is not None:
            index = idx["Close"].reindex(closes.index).ffill(limit=5)
    return closes, volumes, index, meta.loc[[s for s in meta.index if s in closes.columns]], bars


def features(closes: pd.DataFrame, volumes: pd.DataFrame, index: pd.Series | None,
             bars: dict[str, pd.DataFrame] | None = None, short_term: bool = False) -> dict[str, pd.DataFrame]:
    """Each feature as a (date x symbol) frame, using only past data at every date.
    short_term adds the next-day inputs (open/high/low ones only when bars are given)."""
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

    if short_term:
        for n in (1, 2, 3):
            f[f"ret_{n}"] = r / r.shift(n) - 1
        f["dist_sma10"] = r / r.rolling(10, min_periods=8).mean() - 1
        f["vol_5"] = lr.rolling(5, min_periods=4).std() * np.sqrt(252)
        avg_vol = volumes.rolling(20, min_periods=15).mean()
        f["volume_shock"] = volumes / avg_vol.replace(0, np.nan)
        if bars and all(k in bars for k in ("open", "high", "low")):
            o, h, lo = bars["open"], bars["high"], bars["low"]
            span = (h - lo).where(h > lo)
            f["gap_1"] = o / r.shift(1) - 1
            f["intraday_1"] = r / o - 1
            f["clv_1"] = (r - lo) / span
            day_range = span / r
            f["range_ratio"] = day_range / day_range.rolling(20, min_periods=15).mean()

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
        if short_term:
            market["mkt_ret_1"] = index / index.shift(1) - 1
            market["mkt_ret_5"] = index / index.shift(5) - 1
    else:
        market = {}
    market["breadth"] = breadth
    for name, series in market.items():
        f[name] = pd.DataFrame(np.repeat(series.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    return f


def build_table(closes: pd.DataFrame, feats: dict[str, pd.DataFrame], horizon: int, opens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table: one row per (date, symbol) with percentile-ranked stock features,
    raw market features, the forward return and the outperformance label.
    With opens, also the tradable return from the next open (fwd_ret_open)."""
    stock_cols = [k for k in feats if k not in MARKET_FEATURES]
    cols = {k: feats[k].rank(axis=1, pct=True) for k in stock_cols}
    cols.update({k: feats[k] for k in MARKET_FEATURES if k in feats})
    fwd = closes.shift(-horizon) / closes - 1
    cols["fwd_ret"] = fwd
    cols["raw_" + "ret_252"] = feats["ret_252"]
    extra = ["fwd_ret", "raw_ret_252"]
    if opens is not None:
        cols["fwd_ret_open"] = closes.shift(-horizon) / opens.shift(-1) - 1
        extra.append("fwd_ret_open")

    warm = 252
    dates = closes.index[warm::step_for(horizon)]
    if closes.index[-1] not in dates:
        dates = dates.append(pd.DatetimeIndex([closes.index[-1]]))
    table = pd.concat({k: v.loc[dates].stack(future_stack=True) for k, v in cols.items()}, axis=1)
    table.index.names = ["date", "symbol"]
    table = table.reset_index()
    # Need most features present to be useful.
    feat_cols = [k for k in cols if k not in extra]
    table = table[table[feat_cols].notna().mean(axis=1) >= 0.7]
    median = table.groupby("date")["fwd_ret"].transform("median")
    table["outperform"] = (table["fwd_ret"] > median).astype(float).where(table["fwd_ret"].notna())
    return table


def step_for(horizon: int) -> int:
    return 1 if horizon == 1 else STEP


# Candidate models. The feedback loop (services/learning.py) decides which to trust:
# the trees, the linear model or their blend, each trained either on all history
# equally or with recent years weighted more (adapts faster when markets change).
VARIANTS = {
    "blend": ("blend", None),
    "trees": ("trees", None),
    "linear": ("linear", None),
    "blend_recent": ("blend", RECENT_HALF_LIFE),
    "trees_recent": ("trees", RECENT_HALF_LIFE),
    "linear_recent": ("linear", RECENT_HALF_LIFE),
}
VARIANT_LABELS = {
    "blend": "Trees + linear blend",
    "trees": "Gradient-boosted trees",
    "linear": "Logistic regression",
    "blend_recent": "Blend, recent years weighted",
    "trees_recent": "Trees, recent years weighted",
    "linear_recent": "Linear, recent years weighted",
}
DEFAULT_VARIANT = "blend"


def _linear():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))


def recency_weights(dates: pd.Series, half_life: int | None) -> np.ndarray | None:
    """Sample weights halving every `half_life` trading days (~252 a year) back from the newest row."""
    if half_life is None:
        return None
    d = pd.DatetimeIndex(dates)
    age_days = (d.max() - d).days.to_numpy() * (252 / 365)
    return 0.5 ** (age_days / half_life)


class Parts:
    """The two fitted building blocks for one weighting scheme; every variant is a mix of them."""

    def __init__(self, train: pd.DataFrame, feat_cols: list[str], half_life: int | None):
        w = recency_weights(train["date"], half_life)
        self.trees = _trees().fit(train[feat_cols], train["outperform"], sample_weight=w)
        self.linear = _linear().fit(train[feat_cols], train["outperform"], logisticregression__sample_weight=w)

    def proba(self, X: pd.DataFrame, kind: str) -> np.ndarray:
        if kind == "trees":
            return self.trees.predict_proba(X)[:, 1]
        if kind == "linear":
            return self.linear.predict_proba(X)[:, 1]
        return (self.trees.predict_proba(X)[:, 1] + self.linear.predict_proba(X)[:, 1]) / 2


class VariantModel(ClassifierMixin, BaseEstimator):
    """A fitted variant behind the scikit-learn interface (used for permutation importance)."""

    def __init__(self, parts: Parts | None = None, kind: str = "blend"):
        self.parts = parts
        self.kind = kind

    def fit(self, X, y):  # already fitted; present for the interface
        return self

    @property
    def classes_(self):
        return np.array([0.0, 1.0])

    def predict_proba(self, X):
        p = self.parts.proba(X, self.kind)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(float)


def fit_all(train: pd.DataFrame, feat_cols: list[str]) -> dict[int | None, Parts]:
    return {hl: Parts(train, feat_cols, hl) for hl in {h for _, h in VARIANTS.values()}}


def variant_probs(parts: dict[int | None, Parts], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: parts[hl].proba(X, kind) for name, (kind, hl) in VARIANTS.items()}


def _trees() -> HistGradientBoostingClassifier:
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


@dataclass
class WalkForward:
    oos: pd.DataFrame  # adaptive out-of-sample predictions (prob) plus every variant's (v_<name>)
    fold_ic: dict[str, list[float]]  # per variant, mean daily IC of each fold, oldest first
    choices: list[dict]  # which variant the adaptive model used in each fold, and why
    last_parts: dict
    last_test: pd.DataFrame


def walk_forward(table: pd.DataFrame, all_dates: pd.DatetimeIndex, feat_cols: list[str], horizon: int) -> WalkForward:
    """Every test window is predicted by models trained only on earlier data. The variant used in
    each window is the one with the best IC on the windows before it, which is exactly how the
    live model learns from scored outcomes, so the reported results include that adaptation."""
    labeled = table[table["outperform"].notna()]
    dates = np.sort(labeled["date"].unique())
    test_dates = dates[dates >= all_dates[max(0, len(all_dates) - horizon - VALIDATION_DAYS)]]
    if len(test_dates) < FOLDS * 2:
        raise NotEnoughData("Not enough history to validate the model.")
    pos = {d: i for i, d in enumerate(all_dates)}

    oos, choices = [], []
    fold_ic: dict[str, list[float]] = {v: [] for v in VARIANTS}
    last_parts, last_test = None, None
    for chunk in np.array_split(test_dates, FOLDS):
        start = pos[pd.Timestamp(chunk[0])]
        cutoff = all_dates[max(0, start - horizon - 1)]  # labels of training rows must be known before the test window
        train = labeled[labeled["date"] <= cutoff]
        test = labeled[labeled["date"].isin(chunk)]
        if len(train) < 2000 or test.empty:
            continue
        parts = fit_all(train, feat_cols)
        probs = variant_probs(parts, test[feat_cols])
        out = test[[c for c in ("date", "symbol", "fwd_ret", "fwd_ret_open", "outperform") if c in test.columns]].copy()
        for name, p in probs.items():
            out[f"v_{name}"] = p
        # Decide before looking at this window: only earlier windows' outcomes count.
        chosen, _ = learning.choose({v: ics for v, ics in fold_ic.items() if ics}, default=DEFAULT_VARIANT) if any(fold_ic.values()) else (DEFAULT_VARIANT, {})
        out["prob"] = out[f"v_{chosen}"]
        out["variant"] = chosen
        for name in VARIANTS:
            ic = learning.daily_ic(out, f"v_{name}")
            fold_ic[name].append(float(ic.mean()) if len(ic) else float("nan"))
        choices.append({"from": str(pd.Timestamp(chunk[0]).date()), "to": str(pd.Timestamp(chunk[-1]).date()), "variant": chosen,
                        "ic": fold_ic[chosen][-1]})
        oos.append(out)
        last_parts, last_test = parts, test
    if not oos:
        raise NotEnoughData("Not enough history to validate the model.")
    return WalkForward(pd.concat(oos), fold_ic, choices, last_parts, last_test)


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
            **({"avg_return_from_open": float(g["fwd_ret_open"].mean()), "prob_up_from_open": float((g["fwd_ret_open"] > 0).mean())}
               if "fwd_ret_open" in g and g["fwd_ret_open"].notna().any() else {}),
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
    tradable = None
    if "avg_return_from_open" in top:
        tradable = top["avg_return_from_open"] - bottom["avg_return_from_open"]
        if level in ("meaningful", "modest") and top["avg_return_from_open"] <= float(oos["fwd_ret_open"].mean()):
            level = "none"
            text = ("The model ranks stocks better than chance close-to-close, but the edge disappeared when buying at the next open: "
                    "most of the move happens overnight, before you can act. Treat these lists as no better than chance.")
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
        "top_minus_bottom_return_from_open": tradable,
        "avg_return_from_open": float(oos["fwd_ret_open"].mean()) if "fwd_ret_open" in oos else None,
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
    progress("Downloading 5 years of prices…" if code == ASSETS_CODE else "Downloading 5 years of prices for the most-traded stocks…")
    closes, volumes, index, meta, bars = load_prices(code)
    next_day = horizon == 1
    feats = features(closes, volumes, index, bars, short_term=next_day)
    progress(f"Computing {len(feats)} features for {closes.shape[1]} stocks…")
    table = build_table(closes, feats, horizon, bars.get("open") if next_day else None)
    feat_cols = [c for c in FEATURE_LABELS if c in table.columns]

    # Feedback first: score every earlier prediction whose horizon has now passed.
    learning.resolve(code, horizon, closes)

    progress(f"Testing {len(VARIANTS)} model variants walk-forward on the last 2 years…")
    wf = walk_forward(table, closes.index, feat_cols, horizon)
    validation = evaluate(wf.oos)

    # Learn: which variant has the best evidence, from the backtest and from real, scored predictions?
    live_ic = learning.live_variant_ic(code, horizon)
    chosen, board = learning.choose(wf.fold_ic, live_ic, default=DEFAULT_VARIANT)
    kind, _ = VARIANTS[chosen]

    last_model = VariantModel(wf.last_parts[VARIANTS[chosen][1]], kind)
    imp = permutation_importance(last_model, wf.last_test[feat_cols], wf.last_test["outperform"], n_repeats=3, random_state=0, scoring="roc_auc")
    importance = sorted(
        ({"feature": c, "label": FEATURE_LABELS[c], "importance": float(v)} for c, v in zip(feat_cols, imp.importances_mean)),
        key=lambda x: x["importance"],
        reverse=True,
    )
    important = [x["feature"] for x in importance[:8]]

    progress("Training the final models on all history and ranking today's list…")
    labeled = table[table["outperform"].notna()]
    parts = fit_all(labeled, feat_cols)
    latest_date = closes.index[-1]
    now = table[table["date"] == latest_date].copy()
    probs = variant_probs(parts, now[feat_cols])
    now["prob"] = probs[chosen]
    now["decile"] = np.ceil(now["prob"].rank(pct=True) * 10).clip(1, 10).astype(int)

    # Publish to the ledger so these calls get checked against reality later.
    learning.record(code, horizon, latest_date.date(), [
        {"symbol": r["symbol"], "prob": float(r["prob"]), "decile": int(r["decile"]), "variant": chosen,
         "variants": {v: round(float(p[i]), 5) for v, p in probs.items()}, "price": float(closes.at[latest_date, r["symbol"]])}
        for i, (_, r) in enumerate(now.iterrows())
    ])
    record = learning.track_record(code, horizon, names=meta["name"].to_dict())
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
            "historical_avg_return_from_open": hist.get("avg_return_from_open"),
            "drivers": _drivers(r, raw, important),
        }

    ranked = now.sort_values("prob", ascending=False)
    return {
        "status": "ready",
        "market": code,
        "market_name": market_name(code),
        "horizon_days": horizon,
        "horizon_label": HORIZONS[horizon],
        "as_of": str(latest_date.date()),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_seconds": round(time.time() - t0, 1),
        "universe": int(closes.shape[1]),
        "training_rows": int(len(labeled)),
        "features": len(feat_cols),
        "validation": validation,
        "learning": {
            "chosen": chosen,
            "chosen_label": VARIANT_LABELS[chosen],
            "leaderboard": sorted(({"variant": v, "label": VARIANT_LABELS[v], **stats} for v, stats in board.items()),
                                  key=lambda x: x["score"], reverse=True),
            "adaptive_history": wf.choices,
            "track_record": record,
        },
        "feature_importance": importance,
        "bullish": [row(r) for _, r in ranked.head(TOP_N).iterrows()],
        "bearish": [row(r) for _, r in ranked.tail(TOP_N).iloc[::-1].iterrows()],
        "all": [
            {"symbol": r["symbol"], "name": meta.loc[r["symbol"], "name"], "sector": meta.loc[r["symbol"], "sector"],
             "probability_outperform": float(r["prob"]), "decile": int(r["decile"])}
            for _, r in ranked.iterrows()
        ],
        "caveats": ([
            "Next-day moves are mostly random. Even the top bucket falls on a large share of days, so never put much money on a single name.",
            "Trading costs (brokerage, taxes, spread) can eat a next-day edge entirely; check the from-open return above before trading on it.",
        ] if next_day else []) + [
            "Survivorship bias: only assets listed today are included, so the test results are somewhat flattering.",
            "Validation used weekly snapshots with overlapping holding periods, so the effective number of independent tests is smaller than it looks."
            if not next_day else "Stocks move together on any given day, so the effective number of independent tests is smaller than the prediction count suggests.",
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
            (job.status == "ready" and time.time() - job.finished > (RESULT_TTL_NEXT_DAY if horizon == 1 else RESULT_TTL)) or (job.status == "error" and time.time() - job.finished > 300)
        )
        if job is None or stale:
            job = _jobs[key] = Job()
            threading.Thread(target=_run, args=(key,), daemon=True, name=f"predict-{code}-{horizon}").start()
    if job.status == "ready":
        return job.result
    if job.status == "error":
        return {"status": "error", "message": job.error}
    return {"status": "training", "message": job.message, "elapsed_seconds": round(time.time() - job.started)}


# ---------------------------------------------------------------------------
# Always-on learning loop
# ---------------------------------------------------------------------------

LOOP_EVERY = 6 * 3600
ALWAYS_WATCH = [("us", 1), ("us", 21), ("in", 1), ("in", 21), (ASSETS_CODE, 21), (ASSETS_CODE, 1)]
_loop_stop = threading.Event()


def _learning_loop() -> None:
    """Keep publishing and scoring predictions even when nobody opens the page,
    so the track record (and the evidence the model learns from) grows every day."""
    _loop_stop.wait(120)  # let the service settle after start-up
    while not _loop_stop.is_set():
        keys = list(dict.fromkeys(ALWAYS_WATCH + learning.watched()))
        for code, horizon in keys:
            if _loop_stop.is_set():
                return
            try:
                while get(code, horizon).get("status") == "training" and not _loop_stop.is_set():
                    _loop_stop.wait(10)  # one model at a time keeps the CPU free for users
            except Exception:
                log.exception("Learning loop failed for %s/%s", code, horizon)
        _loop_stop.wait(LOOP_EVERY)


def start_learning_loop() -> None:
    _loop_stop.clear()
    threading.Thread(target=_learning_loop, name="prediction-learning", daemon=True).start()


def stop_learning_loop() -> None:
    _loop_stop.set()
