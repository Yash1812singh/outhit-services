"""Backtest engine, strategy rules, 13F parsing and the strategy endpoints."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from research import backtest, gurus, strategies
from research.main import app
from research.strategies import Context

client = TestClient(app)
INTERNAL = {"X-Internal-Token": "test-internal-token"}


def _days(n: int, end: str = "2026-09-30") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_buy_and_hold_matches_price_and_trades_the_next_day():
    idx = _days(10)
    px = pd.DataFrame({"A": [100.0] * 5 + [110.0] * 5}, index=idx)
    # Signal on day 4 (index 4, the day before the jump): executes at day 5's close, after the jump.
    eq, trades = backtest.simulate(px, {idx[4]: pd.Series({"A": 1.0})}, idx[0], cost_bps=0)
    assert eq.iloc[-1] == pytest.approx(1.0)  # bought at 110: missed the jump, no look-ahead
    assert trades[0]["date"] == idx[5] and trades[0]["signal"] == idx[4]
    eq2, _ = backtest.simulate(px, {idx[3]: pd.Series({"A": 1.0})}, idx[0], cost_bps=0)
    assert eq2.iloc[-1] == pytest.approx(1.1)  # bought at day 4's close (100), caught the jump


def test_costs_are_charged_on_turnover():
    idx = _days(6)
    px = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    targets = {idx[0]: pd.Series({"A": 1.0}), idx[2]: pd.Series({"B": 1.0})}
    eq, trades = backtest.simulate(px, targets, idx[0], cost_bps=10)
    # buy A (turnover 1), then switch A->B (turnover 2): 3 x 10 bps
    assert [t["turnover"] for t in trades] == pytest.approx([1.0, 2.0])
    assert eq.iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.002))


def test_partial_weights_leave_cash():
    idx = _days(5)
    px = pd.DataFrame({"A": [100.0, 100, 200, 200, 200]}, index=idx)
    eq, _ = backtest.simulate(px, {idx[0]: pd.Series({"A": 0.5})}, idx[0], cost_bps=0)
    assert eq.iloc[-1] == pytest.approx(1.5)


def test_signal_dates_are_period_ends_after_start():
    idx = _days(70, end="2026-03-31")
    d = backtest.signal_dates(idx, idx[0], "M")
    assert d[0] == idx[0]
    assert all(x.is_month_end or (x + pd.offsets.BDay(1)).month != x.month for x in d[1:])
    assert idx[-1] not in d  # no signal that could only execute after the data ends


def test_stats_on_steady_growth():
    idx = pd.bdate_range("2020-01-01", "2023-01-01")
    years = (idx[-1] - idx[0]).days / 365.25
    eq = pd.Series(np.exp(np.linspace(0, np.log(1.1) * years, len(idx))), index=idx)
    s = backtest.stats(eq, rf=0.0)
    assert s["cagr"] == pytest.approx(0.10, abs=0.002)
    assert s["max_drawdown"] == pytest.approx(0, abs=1e-9)
    assert s["positive_months"] == 1


def _curve(monthly: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(monthly) * 21)
    daily = np.repeat([(1 + m) ** (1 / 21) - 1 for m in monthly], 21)
    return pd.Series(np.cumprod(1 + daily), index=idx)


def test_verdict_levels():
    rng = np.random.default_rng(1)
    base = rng.normal(0.008, 0.04, 48)
    bench = _curve(list(base))
    good = _curve(list(base + 0.01))  # +1%/month every month
    noisy = _curve(list(base + rng.normal(0.002, 0.05, 48)))
    bad = _curve(list(base - 0.01))
    sb = backtest.stats(bench, 0)
    v = backtest.verdict(good, bench, backtest.stats(good, 0), sb, sb, stock_picking=False)
    assert v["level"] == "meaningful" and v["t_stat"] > 2
    assert backtest.verdict(bad, bench, backtest.stats(bad, 0), sb, sb, False)["level"] == "negative"
    assert backtest.verdict(noisy, bench, backtest.stats(noisy, 0), sb, sb, False)["level"] in ("modest", "marginal", "negative")
    capped = backtest.verdict(good, bench, backtest.stats(good, 0), sb, sb, stock_picking=True, momentum=True)
    assert capped["level"] == "modest" and any("recent winners" in n for n in capped["notes"])
    assert any("inflated" in n for n in capped["notes"])  # stock strategies explain survivorship bias


# ---------------------------------------------------------------------------
# Strategy rules on synthetic markets
# ---------------------------------------------------------------------------

def _price_ctx(prices: dict[str, np.ndarray]) -> Context:
    idx = _days(len(next(iter(prices.values()))))
    closes = pd.DataFrame(prices, index=idx)
    vols = pd.DataFrame(1e6, index=idx, columns=closes.columns)
    f = strategies.methods.features(closes, vols)
    s200 = closes.rolling(200, min_periods=150).mean()
    f |= {"sma50": closes.rolling(50, min_periods=40).mean(), "sma150": closes.rolling(150, min_periods=120).mean(), "sma200": s200,
          "sma200_rising": (s200 > s200.shift(21)).astype(float).where(s200.notna()),
          "vol_surge": vols.rolling(10).mean() / vols.rolling(100, min_periods=70).mean()}
    return Context("us", closes, f, {s: s for s in prices}, {}, closes.mean(axis=1))


def test_minervini_only_buys_confirmed_uptrends():
    n = 400
    t = np.arange(n)
    names = {f"UP{i}": 50 * np.exp(0.002 * t * (1 + i / 10)) for i in range(4)}
    names |= {f"DN{i}": 100 * np.exp(-0.001 * t) for i in range(4)}
    names |= {f"FL{i}": 100 + np.sin(t / 7) for i in range(4)}
    ctx = _price_ctx(names)
    w = strategies.minervini(ctx, ctx.closes.index[-1], {})
    assert len(w) and all(s.startswith("UP") for s in w.index)
    assert w.sum() <= 1 + 1e-9


def test_pick_keeps_cash_when_few_qualify():
    ctx = _price_ctx({f"S{i}": np.linspace(100, 100 + i, 300) for i in range(10)})
    t = ctx.closes.index[-1]
    score = pd.Series(range(10), index=ctx.closes.columns, dtype=float)
    elig = pd.Series([True, True] + [False] * 8, index=ctx.closes.columns)
    w = strategies._pick(ctx, t, score, elig, n=20)
    assert len(w) == 2 and w.sum() == pytest.approx(2 / 10)  # 2 names in 10 slots; 80% stays in cash


def test_antonacci_switches_to_bonds_when_stocks_lag_cash():
    idx = _days(400)
    up = np.linspace(100, 150, 400)
    down = np.linspace(150, 100, 400)
    flat = np.linspace(100, 104, 400)
    for spy, veu, expected in ((up, flat, "SPY"), (flat * 0.99, up, "VEU"), (down, down, "AGG")):
        closes = pd.DataFrame({"SPY": spy, "VEU": veu, "BIL": flat, "AGG": flat}, index=idx)
        ctx = Context("us", closes, {"ret_252": closes / closes.shift(252) - 1}, {}, {}, closes["SPY"])
        assert list(strategies.antonacci(ctx, idx[-1], {}).index) == [expected]


def test_faber_moves_assets_below_trend_to_cash():
    idx = _days(300)
    cols = ["VTI", "VEU", "IEF", "VNQ", "DBC", "BIL"]
    data = {c: np.linspace(100, 130, 300) for c in cols}
    data["DBC"] = np.linspace(130, 100, 300)  # falling: below its 10-month average
    closes = pd.DataFrame(data, index=idx)
    ctx = Context("us", closes, {"sma210": closes.rolling(210).mean()}, {}, {}, closes["VTI"])
    w = strategies.faber(ctx, idx[-1], {})
    assert "DBC" not in w and w["BIL"] == pytest.approx(0.2) and w.sum() == pytest.approx(1.0)


def test_custom_params_are_validated_and_idempotent():
    p = strategies.parse_custom({"factors": "momentum:2,value:1,bogus:3", "top_n": "15", "rebalance": "w", "trend_filter": "1"})
    assert p["factors"] == {"momentum": 2.0, "value": 1.0} and p["rebalance"] == "W" and p["trend_filter"] is True
    assert strategies.parse_custom(p) == p
    for bad in ({"factors": "bogus:1"}, {"top_n": "99"}, {"rebalance": "D"}, {"weighting": "cap"}):
        with pytest.raises(ValueError):
            strategies.parse_custom(bad)
    st = strategies.get("custom", {"factors": "quality:1"})
    assert st.needs == ("quality",) and strategies.momentum_biased(st, strategies.parse_custom({"factors": "quality:1"})) is False


def test_catalogue_has_every_kind():
    cat = strategies.catalogue()
    kinds = {s["kind"] for s in cat}
    assert kinds == {"stocks", "allocation", "guru", "custom"}
    assert all(s["markets"] == ["us"] for s in cat if s["kind"] == "guru")
    assert len({s["key"] for s in cat}) == len(cat)


# ---------------------------------------------------------------------------
# 13F filings
# ---------------------------------------------------------------------------

INFO_TABLE = b"""<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><cusip>037833100</cusip><value>600</value>
    <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
  <infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><cusip>037833100</cusip><value>200</value>
    <shrsOrPrnAmt><sshPrnamt>4</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
  <infoTable><nameOfIssuer>COCA COLA CO</nameOfIssuer><cusip>191216100</cusip><value>200</value>
    <shrsOrPrnAmt><sshPrnamt>20</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
  <infoTable><nameOfIssuer>SPDR S&amp;P 500</nameOfIssuer><cusip>78462F103</cusip><value>999</value>
    <shrsOrPrnAmt><sshPrnamt>5</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><putCall>Put</putCall></infoTable>
</informationTable>"""


class _Resp:
    def __init__(self, content=b"", js=None):
        self.content, self._js = content, js

    def json(self):
        return self._js


def test_13f_holdings_are_aggregated_and_options_skipped(monkeypatch):
    monkeypatch.setattr(gurus.data, "cached", lambda key, ttl, fn: fn())

    def fake_sec(url):
        if url.endswith("index.json"):
            return _Resp(js={"directory": {"item": [{"name": "primary_doc.xml"}, {"name": "infotable.xml"}]}})
        return _Resp(INFO_TABLE)

    monkeypatch.setattr(gurus, "_sec", fake_sec)
    h = gurus._holdings(1, "0000000000-26-000001")
    assert [(x["cusip"], x["value"], x["shares"]) for x in h] == [("037833100", 800, 14), ("191216100", 200, 20)]
    assert h[0]["issuer"] == "Apple Inc"


def _filing(filed: str, holdings: list[tuple[str, str, float, float]]) -> dict:
    total = sum(v for *_, v, _s in holdings)
    return {"filed": filed, "period": filed, "accession": filed, "mapped_weight": 1.0,
            "holdings": [{"cusip": c, "ticker": t, "issuer": t, "value": v, "shares": s, "weight": v / total} for c, t, v, s in holdings]}


def test_13f_weights_drop_tiny_positions_and_renormalise():
    f = _filing("2026-02-14", [("1", "AAA", 600, 1), ("2", "BBB", 397, 1), ("3", "CCC", 3, 1), ("4", None, 0, 1)])
    w = gurus._weights(f)
    assert set(w.index) == {"AAA", "BBB"} and w.sum() == pytest.approx(1.0)


def test_13f_copy_uses_only_filings_already_public():
    idx = _days(60, end="2026-05-29")
    closes = pd.DataFrame({"AAA": 10.0, "BBB": 20.0}, index=idx)
    old = _filing("2026-02-14", [("1", "AAA", 100, 10)])
    new = _filing("2026-05-15", [("2", "BBB", 100, 5)])
    st = gurus.strategy("buffett_13f", [new, old])
    ctx = Context("us", closes, {}, {}, {}, closes["AAA"])
    assert list(st.select(ctx, pd.Timestamp("2026-05-14"), {}).index) == ["AAA"]
    assert list(st.select(ctx, pd.Timestamp("2026-05-15"), {}).index) == ["BBB"]


def test_13f_changes():
    old = _filing("2026-02-14", [("1", "AAA", 100, 10), ("2", "BBB", 100, 10), ("3", "CCC", 100, 10)])
    new = _filing("2026-05-15", [("1", "AAA", 100, 15), ("2", "BBB", 100, 5), ("4", "DDD", 100, 1)])
    acts = {c["ticker"]: c["action"] for c in gurus.changes(new, old)}
    assert acts == {"AAA": "added", "BBB": "trimmed", "CCC": "sold", "DDD": "new"}
    assert gurus.changes(new, None) == []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_strategies_endpoint():
    d = client.get("/api/ai/strategies").json()
    assert len(d["strategies"]) >= 20 and {f["key"] for f in d["factors"]} >= {"momentum", "value", "quality"}


def test_backtest_endpoint_validates(monkeypatch):
    monkeypatch.setattr(backtest, "get", lambda *a: {"status": "running", "message": "x"})
    assert client.get("/api/ai/backtest/nope?market=in").status_code == 404
    assert client.get("/api/ai/backtest/buffett?market=xx").status_code == 422
    assert client.get("/api/ai/backtest/buffett_13f?market=in").status_code == 422
    assert client.get("/api/ai/backtest/custom?market=in&factors=bogus:1").status_code == 422
    assert client.get("/api/ai/backtest/custom?market=in&factors=value:1&top_n=10").json()["status"] == "running"


def test_targets_endpoint_is_internal(monkeypatch):
    monkeypatch.setattr(backtest, "targets", lambda *a: {"status": "ready", "weights": []})
    assert client.get("/internal/strategies/dalio/targets?market=in").status_code == 403
    assert client.get("/internal/strategies/dalio/targets?market=in", headers=INTERNAL).json()["status"] == "ready"
