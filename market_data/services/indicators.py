"""Pure technical-indicator functions on a daily close series."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def periods_per_year(close: pd.Series) -> int:
    """365 for assets that trade every day (crypto), 252 for exchange-traded ones."""
    last_year = close[close.index > close.index[-1] - pd.Timedelta(days=365)]
    return 365 if len(last_year) > 300 else TRADING_DAYS


def rolling_volatility(close: pd.Series, window: int = 63, ppy: int = TRADING_DAYS) -> pd.Series:
    """Annualised volatility of daily log returns."""
    return np.log(close).diff().rolling(window, min_periods=window).std() * np.sqrt(ppy)


def annual_volatility(close: pd.Series) -> float:
    ppy = periods_per_year(close)
    r = log_returns(close.iloc[-ppy - 1 :])
    return float(r.std() * np.sqrt(ppy)) if len(r) > 1 else float("nan")


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    return float((close / close.cummax() - 1).min())


def sharpe_ratio(close: pd.Series, risk_free: float = 0.04) -> float:
    ppy = periods_per_year(close)
    r = close.iloc[-ppy - 1 :].pct_change().dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    excess = r.mean() * ppy - risk_free
    return float(excess / (r.std() * np.sqrt(ppy)))


def daily_change(close: pd.Series) -> float | None:
    return float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None


def period_return(close: pd.Series, calendar_days: int) -> float | None:
    """Return versus the last close on or before `calendar_days` ago."""
    start = close[close.index <= close.index[-1] - pd.Timedelta(days=calendar_days)]
    if start.empty:
        return None
    return float(close.iloc[-1] / start.iloc[-1] - 1)


def ytd_return(close: pd.Series) -> float | None:
    this_year = close[close.index.year == close.index[-1].year]
    prior = close[close.index < this_year.index[0]]
    if prior.empty:
        return None
    return float(close.iloc[-1] / prior.iloc[-1] - 1)
