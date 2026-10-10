"""The Outhit signal model, its own backtest, and probabilistic forecasts.

Design principles:
- Transparent: the score is a weighted sum of four named factors, each in
  [-1, 1], and every factor is returned with a plain-English explanation.
- Self-auditing: the exact same score is computed for every past day and
  compared with what actually happened next, so each asset ships with an
  honest hit rate instead of a promise.
- No point predictions: forecasts are probability ranges from bootstrapped
  historical returns, with the drift shrunk toward a long-run prior so a
  hot streak is not extrapolated forever.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_data.services import indicators as ind
from market_data.universe import BONDS, CRYPTO, EQUITY_IN, EQUITY_US, GOLD, INDEX_FUND, REIT, Asset

WEIGHTS = {"trend": 0.35, "momentum": 0.35, "macd": 0.15, "rsi": 0.15}
BACKTEST_HORIZON = 63  # ~3 months of trading days
BULLISH_AT = 15
BEARISH_AT = -15

# Long-run annual return assumptions used to anchor forecasts. Deliberately
# conservative; they are roughly in line with published capital-market
# assumptions, not with recent performance.
LONG_RUN_RETURN = {
    INDEX_FUND: 0.08,
    EQUITY_US: 0.08,
    EQUITY_IN: 0.10,
    BONDS: 0.045,
    GOLD: 0.05,
    REIT: 0.07,
    CRYPTO: 0.10,
}
HISTORY_WEIGHT_IN_DRIFT = 0.3


def _clip(s: pd.Series | float) -> pd.Series | float:
    return np.clip(s, -1.0, 1.0)


def factor_series(close: pd.Series) -> pd.DataFrame:
    """Each factor's value in [-1, 1] for every day, using only data up to that day."""
    ppy = ind.periods_per_year(close)
    k = ppy / ind.TRADING_DAYS  # stretches trading-day windows for 7-day-a-week assets
    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    vol = ind.rolling_volatility(close, round(63 * k), ppy)

    trend = (_clip((close / sma200 - 1) / 0.10) + _clip((sma50 / sma200 - 1) / 0.05)) / 2

    # Volatility-scaled momentum: 12-month return skipping the last month (the
    # classic 12-1 factor) plus 3-month return. Scaling by vol stops crypto
    # from always maxing out.
    r_12_1 = close.shift(round(21 * k)) / close.shift(ppy) - 1
    r_3m = close / close.shift(round(63 * k)) - 1
    momentum = 0.6 * _clip(r_12_1 / vol) + 0.4 * _clip(r_3m / (vol * 0.5))

    _, _, hist = ind.macd(close)
    daily_vol = vol / np.sqrt(ppy)
    macd_f = _clip(hist / (close * daily_vol))

    r = ind.rsi(close)
    # Mean reversion: overbought pulls the score down, oversold pushes it up.
    rsi_f = pd.Series(0.0, index=close.index)
    rsi_f = rsi_f.where(r <= 70, -(r - 70) / 30).where(r >= 30, (30 - r) / 30).where(r.notna())

    return pd.DataFrame({"trend": trend, "momentum": momentum, "macd": macd_f, "rsi": rsi_f})


def score_series(factors: pd.DataFrame) -> pd.Series:
    return sum(factors[k] * w for k, w in WEIGHTS.items()) * 100


def label_for(score: float) -> str:
    if score >= 40:
        return "Bullish"
    if score >= BULLISH_AT:
        return "Mildly bullish"
    if score > BEARISH_AT:
        return "Neutral"
    if score > -40:
        return "Mildly bearish"
    return "Bearish"


def risk_level(vol: float) -> str:
    if vol < 0.10:
        return "Low"
    if vol < 0.20:
        return "Moderate"
    if vol < 0.35:
        return "High"
    return "Very high"


def _explain(name: str, value: float, close: pd.Series) -> str:
    last = close.iloc[-1]
    if name == "trend":
        sma200 = ind.sma(close, 200).iloc[-1]
        sma50 = ind.sma(close, 50).iloc[-1]
        side = "above" if last > sma200 else "below"
        cross = "above" if sma50 > sma200 else "below"
        return f"Price is {abs(last / sma200 - 1):.1%} {side} its 200-day average; the 50-day average is {cross} the 200-day."
    if name == "momentum":
        r3 = ind.period_return(close, 91)
        r12 = ind.period_return(close, 365)
        parts = []
        if r12 is not None:
            parts.append(f"{r12:+.1%} over 12 months")
        if r3 is not None:
            parts.append(f"{r3:+.1%} over 3 months")
        return "Returns of " + " and ".join(parts) + ", adjusted for volatility." if parts else "Not enough history."
    if name == "macd":
        return "Short-term momentum (MACD) is " + ("rising." if value > 0 else "falling." if value < 0 else "flat.")
    if name == "rsi":
        r = ind.rsi(close).iloc[-1]
        if r > 70:
            return f"RSI is {r:.0f}: overbought, so a pullback is more likely than usual."
        if r < 30:
            return f"RSI is {r:.0f}: oversold, so a bounce is more likely than usual."
        return f"RSI is {r:.0f}: neither overbought nor oversold."
    return ""


FACTOR_LABELS = {
    "trend": "Long-term trend",
    "momentum": "Momentum",
    "macd": "Short-term momentum",
    "rsi": "Overbought / oversold",
}


def backtest(close: pd.Series, scores: pd.Series) -> dict:
    """How the signal has done on this asset: for every past day, what happened
    over the next ~3 months when the signal was bullish vs bearish."""
    horizon = round(BACKTEST_HORIZON * ind.periods_per_year(close) / ind.TRADING_DAYS)
    fwd = close.shift(-horizon) / close - 1
    frame = pd.DataFrame({"score": scores, "fwd": fwd}).dropna()
    if len(frame) < 60:
        return {"available": False, "horizon_days": horizon}

    def stats(mask: pd.Series) -> dict:
        sub = frame.loc[mask, "fwd"]
        if len(sub) == 0:
            return {"days": 0, "avg_return": None, "win_rate": None}
        return {"days": int(len(sub)), "avg_return": float(sub.mean()), "win_rate": float((sub > 0).mean())}

    bull = stats(frame["score"] >= BULLISH_AT)
    bear = stats(frame["score"] <= BEARISH_AT)
    base = stats(pd.Series(True, index=frame.index))

    # Directional accuracy: bullish days that rose plus bearish days that fell.
    directional = frame[(frame["score"] >= BULLISH_AT) | (frame["score"] <= BEARISH_AT)]
    hits = ((directional["score"] > 0) == (directional["fwd"] > 0)).mean() if len(directional) else None

    edge = None
    if bull["avg_return"] is not None and bear["avg_return"] is not None:
        edge = bull["avg_return"] - bear["avg_return"]

    return {
        "available": True,
        "horizon_days": horizon,
        "period_start": frame.index[0].date().isoformat(),
        "period_end": frame.index[-1].date().isoformat(),
        "bullish": bull,
        "bearish": bear,
        "all_days": base,
        "directional_accuracy": float(hits) if hits is not None else None,
        "edge": edge,
    }


def forecast(close: pd.Series, asset: Asset, paths: int = 4000, seed: int = 7) -> dict:
    """Bootstrapped one-year price cone. Daily returns are resampled from the
    last two years, re-centred on a drift that blends history with a long-run prior."""
    ppy = ind.periods_per_year(close)
    horizon = ppy
    r = ind.log_returns(close.iloc[-2 * ppy - 1 :]).to_numpy()
    if len(r) < 60:
        return {"available": False}

    hist_annual = r.mean() * ppy
    prior = np.log1p(LONG_RUN_RETURN.get(asset.asset_class, 0.07))
    drift = HISTORY_WEIGHT_IN_DRIFT * hist_annual + (1 - HISTORY_WEIGHT_IN_DRIFT) * prior
    centred = r - r.mean() + drift / ppy

    rng = np.random.default_rng(seed)
    draws = rng.choice(centred, size=(paths, horizon), replace=True)
    cum = np.exp(np.cumsum(draws, axis=1))
    last = float(close.iloc[-1])

    step = 7 if ppy > ind.TRADING_DAYS else 5  # one point per week
    days = list(range(step, horizon + 1, step))
    pct = np.percentile(cum[:, [d - 1 for d in days]], [5, 25, 50, 75, 95], axis=0) * last
    cone = [
        {"day": d, "p5": float(pct[0, i]), "p25": float(pct[1, i]), "p50": float(pct[2, i]), "p75": float(pct[3, i]), "p95": float(pct[4, i])}
            for i, d in enumerate(days)
    ]

    horizons = {}
    for label, fraction in (("1m", 1 / 12), ("3m", 0.25), ("6m", 0.5), ("1y", 1.0)):
        d = round(ppy * fraction)
        end = cum[:, d - 1]
        horizons[label] = {
            "days": d,
            "prob_gain": float((end > 1).mean()),
            "median_return": float(np.median(end) - 1),
            "range_90": [float(np.percentile(end, 5) - 1), float(np.percentile(end, 95) - 1)],
            "prob_loss_over_20pct": float((end < 0.8).mean()),
        }

    return {
        "available": True,
        "method": "bootstrap of daily returns (last 2y), drift = 30% recent history + 70% long-run assumption",
        "assumed_annual_drift": float(np.expm1(drift)),
        "cone": cone,
        "horizons": horizons,
    }


@dataclass
class Analysis:
    asset: Asset
    close: pd.Series
    factors: pd.DataFrame
    scores: pd.Series

    @property
    def score(self) -> float:
        return float(self.scores.iloc[-1])

    @property
    def ready(self) -> bool:
        return not np.isnan(self.score)


def analyse(asset: Asset, frame: pd.DataFrame) -> Analysis:
    close = frame["Close"].astype(float)
    factors = factor_series(close)
    return Analysis(asset, close, factors, score_series(factors))


def confidence(analysis: Analysis, bt: dict | None) -> str:
    """High only when the factors agree with each other AND the signal has a
    real track record on this asset."""
    latest = analysis.factors.iloc[-1]
    direction = np.sign(analysis.score)
    agree = sum(WEIGHTS[k] for k, v in latest.items() if np.sign(v) == direction and direction != 0)
    proven = bool(bt and bt.get("available") and (bt.get("directional_accuracy") or 0) >= 0.55 and (bt.get("edge") or 0) > 0)
    # The signal has historically pointed the wrong way on this asset.
    failed = bool(bt and bt.get("available") and bt.get("edge") is not None and bt["edge"] < 0)
    if abs(analysis.score) < BULLISH_AT or failed:
        return "Low"
    if agree >= 0.7 and proven:
        return "High"
    if agree >= 0.5:
        return "Medium"
    return "Low"


def summary(analysis: Analysis, include_backtest: bool = False) -> dict:
    """Compact view used by the overview table and the planner."""
    close = analysis.close
    vol = ind.annual_volatility(close)
    out = {
        "symbol": analysis.asset.symbol,
        "name": analysis.asset.name,
        "asset_class": analysis.asset.asset_class,
        "region": analysis.asset.region,
        "currency": analysis.asset.currency,
        "price": float(close.iloc[-1]),
        "as_of": close.index[-1].date().isoformat(),
        "change_1d": ind.daily_change(close),
        "change_1m": ind.period_return(close, 30),
        "change_1y": ind.period_return(close, 365),
        "volatility": vol,
        "risk_level": risk_level(vol),
        "score": round(analysis.score, 1) if analysis.ready else None,
        "signal": label_for(analysis.score) if analysis.ready else "Insufficient data",
        "sparkline": [round(float(v), 4) for v in close[close.index > close.index[-1] - pd.Timedelta(days=91)].tolist()],
    }
    if include_backtest:
        bt = backtest(close, analysis.scores) if analysis.ready else None
        out["confidence"] = confidence(analysis, bt) if analysis.ready else "Low"
    return out


def detail(analysis: Analysis) -> dict:
    close = analysis.close
    bt = backtest(close, analysis.scores) if analysis.ready else {"available": False}
    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    window = close.index[close.index > close.index[-1] - pd.Timedelta(days=730)]

    latest = analysis.factors.iloc[-1]
    factors = []
    for k, w in WEIGHTS.items():
        v = latest[k]
        factors.append(
            {
                "key": k,
                "label": FACTOR_LABELS[k],
                "weight": w,
                "value": None if np.isnan(v) else round(float(v), 3),
                "contribution": None if np.isnan(v) else round(float(v * w * 100), 1),
                "explanation": _explain(k, v, close) if not np.isnan(v) else "Not enough history yet.",
            }
        )

    def _num(x: float) -> float | None:
        return None if pd.isna(x) else round(float(x), 4)

    return {
        **summary(analysis),
        "confidence": confidence(analysis, bt) if analysis.ready else "Low",
        "factors": factors,
        "returns": {
            "1w": ind.period_return(close, 7),
            "1m": ind.period_return(close, 30),
            "3m": ind.period_return(close, 91),
            "6m": ind.period_return(close, 182),
            "ytd": ind.ytd_return(close),
            "1y": ind.period_return(close, 365),
            "3y": ind.period_return(close, 3 * 365),
        },
        "risk": {
            "volatility": ind.annual_volatility(close),
            "max_drawdown_5y": ind.max_drawdown(close),
            "sharpe_1y": ind.sharpe_ratio(close),
            "rsi": _num(ind.rsi(close).iloc[-1]),
        },
        "history": [
            {"date": d.date().isoformat(), "close": _num(close[d]), "sma50": _num(sma50[d]), "sma200": _num(sma200[d])}
            for d in window
        ],
        "score_history": [
            {"date": d.date().isoformat(), "score": _num(analysis.scores[d])} for d in window[::5]
        ],
        "backtest": bt,
        "forecast": forecast(close, analysis.asset),
    }
