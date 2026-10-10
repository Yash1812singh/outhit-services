"""Stock-selection methods that compete in the tournament.

Each method maps a price panel to a (date x symbol) score where higher = more
bullish, using only data available on each date. They are the best-documented
price-based effects in the academic and practitioner literature:

- 12-1 momentum (Jegadeesh & Titman 1993): past-year winners, skipping the last month.
- 6-month momentum: a shorter-lookback version.
- Risk-adjusted momentum: 12-1 return divided by volatility (smoother winners).
- 52-week-high proximity (George & Hwang 2004): stocks near their highs keep going.
- Trend: price vs. 200-day average and 50/200 crossover.
- Short-term reversal (Jegadeesh 1990; Lehmann 1990): last month's losers bounce.
- 1-week reversal: the very-short-term version.
- Low volatility (Ang et al. 2006; Baker et al. 2011): calmer stocks deliver better risk-adjusted returns.
- Technical composite: Outhit's trend + momentum + MACD + RSI signal.
- Two machine-learning models (gradient-boosted trees, regularised logistic
  regression) trained on all of the above features (see tournament.py).
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def features(closes: pd.DataFrame, volumes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = closes
    lr = np.log(r).diff()
    sma50 = r.rolling(50, min_periods=40).mean()
    sma200 = r.rolling(200, min_periods=150).mean()
    vol63 = lr.rolling(63, min_periods=45).std() * np.sqrt(TRADING_DAYS)
    vol252 = lr.rolling(252, min_periods=180).std() * np.sqrt(TRADING_DAYS)
    ema12, ema26 = r.ewm(span=12, adjust=False).mean(), r.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    mkt = r.pct_change().mean(axis=1)  # equal-weighted market return
    f = {
        "ret_5": r / r.shift(5) - 1,
        "ret_21": r / r.shift(21) - 1,
        "ret_63": r / r.shift(63) - 1,
        "ret_126": r / r.shift(126) - 1,
        "ret_252": r / r.shift(252) - 1,
        "mom_12_1": r.shift(21) / r.shift(252) - 1,
        "mom_6_1w": r.shift(5) / r.shift(126) - 1,
        "vol_21": lr.rolling(21, min_periods=15).std() * np.sqrt(TRADING_DAYS),
        "vol_63": vol63,
        "vol_252": vol252,
        "dist_sma50": r / sma50 - 1,
        "dist_sma200": r / sma200 - 1,
        "sma50_200": sma50 / sma200 - 1,
        "high_52w": r / r.rolling(252, min_periods=200).max(),
        "low_52w": r / r.rolling(252, min_periods=200).min() - 1,
        "rsi_14": _rsi(r),
        "macd_hist": (macd - macd.ewm(span=9, adjust=False).mean()) / r,
        "volume_trend": volumes.rolling(20, min_periods=15).mean() / volumes.rolling(60, min_periods=45).mean(),
        "max_ret_21": lr.rolling(21, min_periods=15).max(),
        "beta_126": (lr.mul(mkt, axis=0).rolling(126, min_periods=90).mean()
                     - lr.rolling(126, min_periods=90).mean().mul(mkt.rolling(126, min_periods=90).mean(), axis=0))
                    .div(mkt.rolling(126, min_periods=90).var(), axis=0),
    }
    return f


def _z(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank, centred on 0."""
    return df.rank(axis=1, pct=True) - 0.5


HEURISTICS = {
    "momentum_12_1": ("12-1 month momentum", lambda f: f["mom_12_1"]),
    "momentum_6m": ("6-month momentum", lambda f: f["mom_6_1w"]),
    "risk_adj_momentum": ("Risk-adjusted momentum", lambda f: f["mom_12_1"] / f["vol_252"]),
    "high_52w": ("52-week-high proximity", lambda f: f["high_52w"]),
    "trend": ("Trend (200-day & 50/200)", lambda f: _z(f["dist_sma200"]) + _z(f["sma50_200"])),
    "reversal_1m": ("1-month reversal", lambda f: -f["ret_21"]),
    "reversal_1w": ("1-week reversal", lambda f: -f["ret_5"]),
    "low_volatility": ("Low volatility", lambda f: -f["vol_63"]),
    "technical": ("Technical composite (trend, momentum, MACD, RSI)", lambda f: (
        0.35 * _z(f["dist_sma200"]) + 0.35 * _z(f["mom_12_1"]) + 0.15 * _z(f["macd_hist"])
        + 0.15 * ((30 - f["rsi_14"]).clip(lower=0) - (f["rsi_14"] - 70).clip(lower=0)) / 30  # oversold +, overbought -
    )),
}

# Factors that need point-in-time fundamentals or earnings data (added to the feature dict when available).
FUNDAMENTAL_METHODS = {
    "quality": ("Quality (profitability, safety, growth)", ("quality",), lambda f: f["quality"]),
    "value_ey": ("Value (earnings yield)", ("earnings_yield",), lambda f: f["earnings_yield"]),
    "quality_momentum": ("Quality + momentum", ("quality",), lambda f: _z(f["quality"]) + _z(f["mom_12_1"])),
    "earnings_surprise": ("Earnings surprise (post-earnings drift)", ("surprise",), lambda f: f["surprise"]),
}
OPTIONAL_ML_FEATURES = ["quality", "earnings_yield", "surprise"]

ML_FEATURES = ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1", "mom_6_1w", "vol_21", "vol_63", "vol_252",
               "dist_sma50", "dist_sma200", "sma50_200", "high_52w", "low_52w", "rsi_14", "macd_hist", "volume_trend", "max_ret_21", "beta_126"]
ML_METHODS = {"ml_gbm": "Machine learning: gradient-boosted trees", "ml_linear": "Machine learning: logistic regression"}

LABELS = ({k: v[0] for k, v in HEURISTICS.items()} | {k: v[0] for k, v in FUNDAMENTAL_METHODS.items()}
          | ML_METHODS | {"ensemble": "Outhit ensemble (best methods combined)"})


def available_methods(f: dict) -> dict:
    """Heuristic methods whose inputs exist in this feature set."""
    out = {k: v for k, v in HEURISTICS.items()}
    for k, (label, needs, fn) in FUNDAMENTAL_METHODS.items():
        if all(n in f and f[n].notna().any().any() for n in needs):
            out[k] = (label, fn)
    return out
