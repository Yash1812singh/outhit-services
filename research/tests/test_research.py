import math

import numpy as np
import pandas as pd
import pytest

from research import allocation, funds, intraday_lab, methods, options, recommend, tournament
from research.db import RecommendationLog, SessionLocal


# ---------------- tournament ----------------

def _panel(n=80, days=1300, signal=0.0, seed=3):
    """Synthetic market where (optionally) last year's winners keep winning: a planted momentum effect."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-08", periods=days)
    trait = rng.normal(size=n)
    rets = rng.normal(0, 0.015, (days, n)) + signal * 0.0015 * trait
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"S{i}" for i in range(n)])
    vols = pd.DataFrame(rng.integers(1e5, 1e6, (days, n)), index=idx, columns=closes.columns).astype(float)
    return closes, vols


def _run(h, signal):
    closes, vols = _panel(signal=signal)
    feats = methods.features(closes, vols)
    table = tournament._ml_table(feats, closes.index[252:])
    return tournament.run_horizon(closes, feats, table, h, lambda m: None)


def test_row_ic_matches_spearman():
    a = pd.DataFrame([[1, 2, 3, 4] * 6])
    b = pd.DataFrame([[2, 4, 6, 8] * 6])
    assert tournament._row_ic(a, b).iloc[0] == pytest.approx(1.0)
    assert tournament._row_ic(a, -b).iloc[0] == pytest.approx(-1.0)


def test_tournament_finds_planted_momentum():
    r = _run(63, signal=1.0)
    assert r["available"]
    assert r["ensemble"]["holdout"]["ic"] > 0.03
    assert set(r["ensemble"]["members"]) & {"momentum_12_1", "momentum_6m", "risk_adj_momentum", "trend", "high_52w", "technical", "ml_gbm", "ml_linear"}
    assert r["verdict_level"] in ("meaningful", "modest")
    assert r["ranking"][0]["percentile"] >= r["ranking"][-1]["percentile"]


def test_tournament_reports_no_edge_on_noise():
    r = _run(21, signal=0.0)
    assert r["verdict_level"] in ("none", "negative", "modest")
    assert abs(r["ensemble"]["holdout"]["ic"] or 0) < 0.06


# ---------------- intraday ----------------

def _session(day, path, vol=1000.0):
    idx = pd.date_range(f"{day} 09:30", periods=len(path), freq="5min", tz="America/New_York")
    c = np.array(path, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 0.05, "Low": c - 0.05, "Close": c, "Volume": vol}, index=idx)


def test_gap_and_go_trade():
    s = _session("2026-10-05", [102, 102.3, 102.6] + list(np.linspace(102.7, 106, 70)))
    t = intraday_lab._trade("gap_and_go", s, prev_close=100.0, rel_vol=2.0)
    assert t and t["side"] == 1 and t["gross"] > 0


def test_orb_requires_volume():
    s = _session("2026-10-05", [100] * 6 + list(np.linspace(100.5, 103, 40)))
    assert intraday_lab._trade("orb", s, 100.0, rel_vol=0.8) is None
    assert intraday_lab._trade("orb", s, 100.0, rel_vol=2.0)["side"] == 1


# ---------------- options ----------------

def test_black_scholes_and_implied_vol_round_trip():
    p = options.bs_price(100, 105, 0.25, 0.05, 0.3, "call")
    assert options.implied_vol(p, 100, 105, 0.25, 0.05, "call") == pytest.approx(0.3, abs=1e-3)
    # put-call parity
    c, put = options.bs_price(100, 100, 0.5, 0.05, 0.2, "call"), options.bs_price(100, 100, 0.5, 0.05, 0.2, "put")
    assert c - put == pytest.approx(100 - 100 * math.exp(-0.05 * 0.5), abs=1e-6)


def test_bull_call_spread_payoff():
    legs = [{"action": "buy", "type": "call", "strike": 100, "price": 5.0}, {"action": "sell", "type": "call", "strike": 110, "price": 2.0}]
    ev = options._evaluate(legs, 100, 0.25, 30 / 365)
    assert ev["max_loss"] == pytest.approx(-3.0, abs=0.01)
    assert ev["max_profit"] == pytest.approx(7.0, abs=0.01)
    assert ev["breakevens"][0] == pytest.approx(103.0, abs=0.05)
    assert 0 < ev["prob_profit"] < 0.5


def _analytics(ratio):
    """Chain priced with Black-Scholes at 25% implied vol; realised vol set so IV/HV = ratio."""
    strikes = np.arange(70, 131, 1.0)
    spot, t, iv = 100.0, 30 / 365, 0.25
    ch = pd.DataFrame({
        "strike": strikes,
        "call_price": [options.bs_price(spot, k, t, 0.0, iv, "call") for k in strikes],
        "put_price": [options.bs_price(spot, k, t, 0.0, iv, "put") for k in strikes],
    })
    return {"_chain": ch, "spot": spot, "_t": t, "atm_iv": iv, "hv_20": iv / ratio, "expected_move": spot * iv * math.sqrt(t), "iv_hv_ratio": ratio}


@pytest.mark.parametrize("direction,ratio,drift,expected", [
    ("bullish", 1.4, 0.0, {"Bull put spread", "No trade"}),   # options expensive: sell premium, or pass if the edge is too thin
    ("bearish", 1.4, 0.0, {"Bear call spread", "No trade"}),  # ...but never buy expensive options
    ("neutral", 1.4, 0.0, {"Iron condor"}),
    ("bullish", 0.9, 0.04, {"Bull call spread", "Bull put spread"}),  # real upward edge: some bullish spread pays
    ("neutral", 0.8, 0.0, {"No trade"}),            # options cheap and no view: nothing has positive EV
])
def test_strategy_selection_by_expected_value(direction, ratio, drift, expected):
    s = options.choose_strategy(_analytics(ratio), direction, drift)
    assert s["name"] in expected
    if s["name"] != "No trade":
        assert s["max_loss"] < 0 and s["max_profit"] > 0  # always defined risk
        assert s["expected_value"] > 0


# ---------------- funds & allocation ----------------

def test_fund_metrics():
    idx = pd.bdate_range("2020-01-01", "2026-10-01")
    nav = pd.Series(10 * 1.12 ** (np.arange(len(idx)) / 252), index=idx)
    m = funds._metrics(nav)
    assert m["cagr_3y"] == pytest.approx(0.12, abs=0.01)
    assert m["max_drawdown_3y"] == pytest.approx(0, abs=1e-9)


def test_trend_rule_sidesteps_a_crash():
    idx = pd.bdate_range("2016-01-01", periods=1500)
    up = np.linspace(100, 200, 700)
    crash = np.linspace(200, 80, 300)
    recover = np.linspace(80, 150, 500)
    c = pd.Series(np.r_[up, crash, recover], index=idx)
    bt = allocation._backtest(c, 252)
    assert bt["trend_rule"]["max_drawdown"] > bt["buy_and_hold"]["max_drawdown"]  # smaller loss
    assert 0 < bt["time_invested"] < 1


# ---------------- ledger ----------------

def test_recommendations_logged_once_per_day():
    res = {"bullish": [{"symbol": "A", "price": 10.0, "score": 50}], "bearish": [{"symbol": "B", "price": 5.0, "score": -40}]}
    recommend.record("us", "short_term", res)
    recommend.record("us", "short_term", res)
    with SessionLocal() as db:
        rows = db.query(RecommendationLog).all()
        assert len(rows) == 2 and {r.side for r in rows} == {1, -1}
        rows[0].outcome = {"5d": {"return": 0.04, "benchmark": 0.01}}
        rows[1].outcome = {"5d": {"return": -0.02, "benchmark": 0.01}}
        db.commit()
    tr = recommend.track_record()["horizons"]["short_term"]["5d"]
    assert tr["evaluated"] == 2 and tr["hit_rate"] == 1.0  # the bearish idea fell, so it counts as a hit
