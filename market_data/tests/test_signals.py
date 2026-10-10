import numpy as np
import pandas as pd

from market_data.services import indicators as ind
from market_data.services import signals as sig
from market_data.universe import lookup
from market_data.tests.conftest import make_prices


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(100, 200, 100))
    down = pd.Series(np.linspace(200, 100, 100))
    assert ind.rsi(up).iloc[-1] > 95
    assert ind.rsi(down).iloc[-1] < 5
    r = ind.rsi(make_prices(0.1)["Close"]).dropna()
    assert r.between(0, 100).all()


def test_max_drawdown_and_period_return():
    s = pd.Series([100, 120, 60, 90], index=pd.date_range("2026-01-01", periods=4))
    assert ind.max_drawdown(s) == -0.5
    assert np.isclose(ind.period_return(s, 3), -0.1)
    assert ind.period_return(s, 30) is None


def test_periods_per_year():
    assert ind.periods_per_year(make_prices(0.1)["Close"]) == 252
    assert ind.periods_per_year(make_prices(0.1, calendar=True)["Close"]) == 365


def test_uptrend_scores_bullish_and_downtrend_bearish():
    up = sig.analyse(lookup("AAPL"), make_prices(0.8, vol=0.15))
    down = sig.analyse(lookup("AAPL"), make_prices(-0.8, vol=0.15))
    assert up.score > sig.BULLISH_AT
    assert down.score < sig.BEARISH_AT
    assert -100 <= down.score <= up.score <= 100


def test_backtest_uses_only_past_data_and_reports_counts():
    a = sig.analyse(lookup("SPY"), make_prices(0.1))
    bt = sig.backtest(a.close, a.scores)
    assert bt["available"]
    total = bt["bullish"]["days"] + bt["bearish"]["days"]
    assert 0 < total <= bt["all_days"]["days"]
    # Scores at day t must not change when future prices are removed (no look-ahead).
    truncated = sig.analyse(lookup("SPY"), make_prices(0.1).iloc[:-100])
    assert np.isclose(truncated.scores.iloc[-1], a.scores.iloc[-101])


def test_forecast_is_a_sane_probability_cone():
    a = sig.analyse(lookup("SPY"), make_prices(0.1))
    f = sig.forecast(a.close, a.asset)
    for h in f["horizons"].values():
        assert 0 <= h["prob_gain"] <= 1
        assert h["range_90"][0] < h["median_return"] < h["range_90"][1]
    for point in f["cone"]:
        assert point["p5"] <= point["p25"] <= point["p50"] <= point["p75"] <= point["p95"]


def test_forecast_drift_is_shrunk_toward_long_run():
    hot = sig.analyse(lookup("NVDA"), make_prices(1.5, vol=0.3))
    f = sig.forecast(hot.close, hot.asset)
    assert f["assumed_annual_drift"] < 0.8  # a 150%/yr streak is not extrapolated


def test_detail_has_everything_the_ui_needs():
    d = sig.detail(sig.analyse(lookup("BTC-USD"), make_prices(0.3, vol=0.6, calendar=True)))
    for key in ("factors", "returns", "risk", "history", "backtest", "forecast", "confidence"):
        assert key in d
    assert len(d["factors"]) == 4
    assert d["forecast"]["horizons"]["1y"]["days"] == 365
