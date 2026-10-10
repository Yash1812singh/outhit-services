from fastapi.testclient import TestClient

from market_data.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"service": "market-data", "status": "ok"}


def test_asset_endpoint_returns_finite_json(fake_market):
    r = client.get("/api/market/asset/AAPL")
    assert r.status_code == 200
    assert "NaN" not in r.text and "Infinity" not in r.text
    assert -100 <= r.json()["score"] <= 100


def test_unknown_asset_is_404(monkeypatch):
    from market_data.services import market_data

    monkeypatch.setattr(market_data, "get_history", lambda symbols: {})
    assert client.get("/api/market/asset/NOPE").status_code == 404


def test_signals_endpoint_used_by_the_advisor(fake_market):
    r = client.get("/api/market/signals", params={"symbols": "aapl, BTC-USD,,SPY"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"AAPL", "BTC-USD", "SPY"}
    for s in body.values():
        assert set(s) >= {"score", "signal", "price", "volatility"}
        assert s["score"] is None or -100 <= s["score"] <= 100
    assert client.get("/api/market/signals").status_code == 422


def test_overview_groups_assets(fake_market):
    body = client.get("/api/market/overview").json()
    assert body["groups"] and body["benchmarks"]
    assert len(body["top_bullish"]) <= 5


def test_overview_reports_outage(monkeypatch):
    from market_data.services import market_data

    monkeypatch.setattr(market_data, "get_history", lambda symbols: {})
    assert client.get("/api/market/overview").status_code == 503
    assert client.get("/api/crypto/overview").status_code == 503


def test_crypto_overview(fake_market):
    r = client.get("/api/crypto/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["bitcoin"]["symbol"] == "BTC-USD"
    assert 0 <= body["in_uptrend"] <= len(body["coins"])
    scores = [c["score"] for c in body["coins"] if c["score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_predict_rejects_unsupported_horizon():
    assert client.get("/api/predict/in", params={"horizon": 5}).status_code == 422
