import numpy as np
import pandas as pd
import pytest

from research import earnings, fundamentals_history as fh, governance, portfolio, regime, tournament


def _closes(symbols, days=900):
    idx = pd.bdate_range(end="2026-10-08", periods=days)
    return pd.DataFrame({s: np.linspace(100, 120, days) for s in symbols}, index=idx)


# ---------------- point-in-time fundamentals ----------------

def test_fundamentals_only_known_after_reporting_lag():
    closes = _closes(["A", "B", "C"])
    st = {
        "A": [{"period_end": pd.Timestamp("2025-03-31"), "revenue": 100, "gross_profit": 60, "operating_income": 30, "net_income": 20, "eps": 5, "assets": 200, "equity": 100, "debt": 10}],
        "B": [{"period_end": pd.Timestamp("2025-03-31"), "revenue": 100, "gross_profit": 10, "operating_income": 2, "net_income": 1, "eps": 1, "assets": 200, "equity": 50, "debt": 120}],
        "C": [],
    }
    p = fh.panels(st, closes)
    q = p["quality"]
    assert q.loc[:"2025-06-28"].isna().all().all()  # not known before period end + 90 days
    assert q.loc["2025-07-01", "A"] > q.loc["2025-07-01", "B"]  # more profitable, less levered
    assert p["earnings_yield"].loc["2026-01-02", "A"] == pytest.approx(5 / closes.loc["2026-01-02", "A"])


# ---------------- earnings surprise ----------------

def test_surprise_panel_is_point_in_time_and_expires(monkeypatch):
    syms = [f"S{i}" for i in range(25)]
    closes = _closes(syms)
    ed = pd.DataFrame({"EPS Estimate": [1.0], "Reported EPS": [1.2], "Surprise(%)": [20.0]}, index=pd.DatetimeIndex(["2026-03-02"]))
    monkeypatch.setattr(earnings, "_surprises", lambda s: ed)
    p = earnings.surprise_panel(syms, closes)
    assert np.isnan(p.loc["2026-03-02", "S0"])  # announced that day: usable from the next session
    assert p.loc["2026-03-03", "S0"] == 20.0
    assert np.isnan(p.iloc[-1]["S0"])  # older than 63 trading days: expired


# ---------------- governance ----------------

def test_governance_flags_heavy_pledging_and_rewards_insider_buying(monkeypatch):
    monkeypatch.setattr(governance, "pledges", lambda: {governance._norm("Risky Co Limited"): {"pledged_pct_of_promoter": 60.0, "promoter_holding_pct": 50, "as_of": "x"},
                                                          governance._norm("Solid Co Limited"): {"pledged_pct_of_promoter": 0.0, "promoter_holding_pct": 60, "as_of": "x"}})
    monkeypatch.setattr(governance, "insider_activity", lambda s, c: {"insider_buy_value_180d": 1e7, "insider_sell_value_180d": 0, "pledges_created_180d": 0, "pledges_invoked_180d": 0} if "Solid" in c else {"pledges_invoked_180d": 1})
    risky = governance.assess("RISKY.NS", "Risky Co Limited", "in")
    solid = governance.assess("SOLID.NS", "Solid Co Limited", "in")
    assert len(risky["flags"]) == 2 and risky["score"] < 0.2
    assert solid["score"] > 0.7 and solid["strengths"]
    assert governance.assess("X", "Anything", "us")["available"] is False


# ---------------- regime ----------------

@pytest.mark.parametrize("shape,expected", [("up", "risk-on"), ("crash", "risk-off")])
def test_regime(monkeypatch, shape, expected):
    idx = pd.bdate_range(end="2026-10-08", periods=600)
    rng = np.random.default_rng(0)
    if shape == "up":
        c = 100 * np.exp(np.cumsum(np.full(600, 0.0008) + rng.normal(0, 0.008, 600)))
    else:
        r = np.r_[rng.normal(0.0005, 0.008, 560), rng.normal(-0.01, 0.04, 40)]
        c = 100 * np.exp(np.cumsum(r))
    monkeypatch.setattr(regime, "_index_symbol", lambda m: "^IDX")
    monkeypatch.setattr(regime.data, "daily", lambda syms, period="3y": (pd.DataFrame({"^IDX": c}, index=idx), None))
    rg = regime.detect("in")
    assert rg["regime"] == expected
    assert 0 < rg["momentum_scale"] <= 1


# ---------------- portfolio ----------------

def test_portfolio_caps():
    w = {"A": 10.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0, "G": 1.0, "H": 1.0, "I": 1.0, "J": 1.0}
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech", "E": "Bank", "F": "Bank", "G": "Energy", "H": "Energy", "I": "Health", "J": "Health"}
    out = portfolio._cap_weights(w, sectors)
    assert sum(out.values()) == pytest.approx(1.0)
    assert max(out.values()) <= portfolio.MAX_STOCK + 1e-6
    assert sum(v for k, v in out.items() if sectors[k] == "Tech") <= portfolio.MAX_SECTOR + 1e-6


# ---------------- costs in the tournament ----------------

def test_net_series_charges_turnover():
    idx = pd.bdate_range("2026-01-01", periods=6)
    cols = [f"S{i}" for i in range(20)]
    stable = pd.DataFrame([list(range(20))] * 6, index=idx, columns=cols, dtype=float)
    flipping = pd.DataFrame([list(range(20)) if i % 2 else list(range(20))[::-1] for i in range(6)], index=idx, columns=cols, dtype=float)
    fwd = pd.DataFrame(0.0, index=idx, columns=cols)
    n_stable = tournament._net_series(stable, fwd, 1, 0.01)
    n_flip = tournament._net_series(flipping, fwd, 1, 0.01)
    assert n_stable.iloc[2:].max() == pytest.approx(0.0)  # no turnover, no cost
    assert n_flip.iloc[2:].max() == pytest.approx(-0.01)  # full turnover each time costs a full round trip
