from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from market_data.db import SessionLocal
from market_data.main import app
from market_data.models import MarketRefresh, StockQuote
from market_data.services import intraday, ranking
from market_data.tests.conftest import INTERNAL, user_headers

client = TestClient(app)


def _quote(symbol, sector, **data):
    base = dict(price=100.0, change_pct=1.0, volume=1e6, avg_volume_3m=1e6, market_cap=1e10, pe=20.0, forward_pe=18.0, pb=3.0,
                eps_ttm=5.0, eps_forward=5.5, dividend_yield_pct=1.0, change_52w_pct=10.0, high_52w=120.0, low_52w=80.0,
                sma50=98.0, sma200=95.0, analyst_rating="2.0 - Buy", market_state="REGULAR", delay_minutes=15)
    base.update(data)
    return StockQuote(symbol=symbol, market="zz", sector=sector, name=f"{symbol} Ltd", exchange="X", currency="INR",
                      market_cap=base["market_cap"], data=base)


def _universe():
    rows = []
    for i in range(40):
        sector = ["Technology", "Energy", "Healthcare", "Financial Services"][i % 4]
        rows.append(_quote(f"S{i}", sector, pe=8 + i, change_52w_pct=60 - 3 * i, market_cap=1e11 / (i + 1), sma50=100 - i * 0.5, sma200=95.0))
    rows.append(_quote("LOSS", "Technology", pe=-5.0, eps_ttm=-1.0, change_52w_pct=-40.0))
    rows.append(_quote("S0B", "Technology", pe=8.0))  # duplicate name check below uses a shared name
    rows[-1].name = "S0 Ltd"
    return rows


def test_score_market_ranks_cheap_strong_stocks_higher_and_dedupes_share_classes():
    df = ranking.score_market(ranking.frame(_universe()))
    assert df["name"].is_unique
    s = df.set_index("symbol")["score"]
    assert s.between(0, 100).all()
    assert s.loc["S0" if "S0" in s.index else "S0B"] > s.loc["S39"]
    assert df.set_index("symbol").loc["LOSS", "f_value"] == pytest.approx(0.1)
    assert set(df["size_tier"]) <= {"Large cap", "Mid cap", "Small cap"}


def test_quality_flags_negative_equity_and_one_off_margins():
    q, good, flags = ranking.quality({"bookValue": -3.3, "profitMargins": 0.82, "revenueGrowth": 0.06}, "Communication Services", {"pe": 3.8, "pb": -1})
    assert len(flags) == 2
    q, good, flags = ranking.quality({"returnOnEquity": 0.3, "debtToEquity": 10, "profitMargins": 0.2, "revenueGrowth": 0.2, "freeCashflow": 1e9}, "Technology")
    assert q > 0.9 and not flags and good


def test_quality_derives_roe_from_pe_and_pb_when_missing():
    q, good, _ = ranking.quality({}, "Industrials", {"pe": 10.0, "pb": 2.5})  # ROE = 2.5 / 10 = 25%
    assert q == pytest.approx(1.0)
    assert any("return on equity" in g for g in good)


def _session(day: str, path: list[float], volume: float = 1000) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:15", periods=len(path), freq="5min", tz="Asia/Kolkata")
    close = np.array(path)
    return pd.DataFrame({"Open": close, "High": close + 0.1, "Low": close - 0.1, "Close": close, "Volume": volume}, index=idx)


def test_orb_simulation_hits_target_on_clean_breakout():
    path = [100, 100.2, 99.9, 100.1, 100, 100.1] + [100.5, 101, 101.5, 102, 102.5, 103] + [103] * 6
    r = intraday._simulate(_session("2026-10-01", path))
    assert r is not None and r > 0


def test_orb_simulation_stops_out_on_failed_breakout():
    path = [100, 100.2, 99.9, 100.1, 100, 100.1] + [100.5, 99.5, 99.0, 98.5] + [98] * 6
    r = intraday._simulate(_session("2026-10-02", path))
    assert r is not None and r < 0


def test_today_detects_long_setup_with_relative_volume():
    quiet = [_session(f"2026-09-{d:02d}", [100] * 20, volume=500) for d in range(1, 22)]
    today = _session("2026-10-05", [100, 100.2, 99.9, 100.1, 100, 100.1, 100.6, 101, 101.4], volume=1000)
    t = intraday._today(quiet + [today])
    assert t["setup"].startswith("Long")
    assert t["relative_volume"] > 1.2
    assert t["stop_loss"] < t["entry"] < t["target"]


@pytest.fixture
def stored_market(monkeypatch):
    from market_data import markets
    from market_data.services import screener

    zz = markets.Market("zz", "Testland", "INR", ())
    monkeypatch.setitem(markets.MARKETS_BY_CODE, "zz", zz)
    monkeypatch.setitem(screener.MARKETS_BY_CODE, "zz", zz)
    with SessionLocal() as db:
        db.query(StockQuote).filter(StockQuote.market == "zz").delete()
        db.add_all(_universe())
        db.merge(MarketRefresh(market="zz", refreshed_at=datetime.now(timezone.utc), companies=42))
        db.commit()
    ranking.clear_cache()
    yield "zz"


def test_companies_endpoint_filters_sorts_and_pages(stored_market):
    r = client.get("/api/stocks/zz/companies", params={"sector": "Energy", "sort": "pe", "desc": False, "page_size": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 10 and len(body["rows"]) == 3
    pes = [row["pe"] for row in body["rows"]]
    assert pes == sorted(pes)
    assert client.get("/api/stocks/zz/companies", params={"q": "s1 "}).status_code == 200


def test_sectors_endpoint(stored_market):
    body = client.get("/api/stocks/zz/sectors").json()
    assert {s["sector"] for s in body["sectors"]} == {"Technology", "Energy", "Healthcare", "Financial Services"}
    assert all(0 <= s["strength"] <= 100 for s in body["sectors"])


def test_unknown_market_and_sector_are_rejected():
    assert client.get("/api/stocks/xx/companies").status_code == 404
    assert client.get("/api/stocks/in/picks", params={"sector": "Crypto"}).status_code == 422


def test_refresh_requires_gateway_token_and_login():
    # Not through the gateway (no internal token): forbidden.
    assert client.post("/api/stocks/in/refresh").status_code == 403
    assert client.post("/api/stocks/in/refresh", headers={"X-User-Id": "1"}).status_code == 403
    assert client.post("/api/stocks/in/refresh", headers={"X-Internal-Token": "wrong", "X-User-Id": "1"}).status_code == 403
    # Through the gateway but anonymous: unauthenticated.
    assert client.post("/api/stocks/in/refresh", headers=INTERNAL).status_code == 401


def test_refresh_runs_for_signed_in_users(monkeypatch):
    from market_data.services import screener

    calls = []
    monkeypatch.setattr(screener, "refresh_market", lambda code: calls.append(code) or 7)
    r = client.post("/api/stocks/IN/refresh", headers=user_headers(5))
    assert r.status_code == 200 and r.json() == {"market": "in", "companies": 7}
    assert calls == ["in"]


def test_breakout_without_volume_is_not_a_setup():
    busy = [_session(f"2026-09-{d:02d}", [100] * 20, volume=5000) for d in range(1, 22)]
    today = _session("2026-10-05", [100, 100.2, 99.9, 100.1, 100, 100.1, 100.6, 101, 101.4], volume=1000)
    t = intraday._today(busy + [today])
    assert t["side"] == 0
    assert "weak volume" in t["setup"]
