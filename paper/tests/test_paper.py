"""Paper trading: order maths, copy-trading rebalances, schedules and the API's ownership rules."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from paper import trading
from paper.db import Portfolio, SessionLocal
from paper.main import app

client = TestClient(app)
INTERNAL = {"X-Internal-Token": "test-internal-token"}


def user(uid: int = 1) -> dict:
    return {**INTERNAL, "X-User-Id": str(uid)}


def _new(market="in", cash=100_000.0, uid=1, **kw) -> dict:
    r = client.post("/api/paper/portfolios", json={"market": market, "cash": cash, **kw}, headers=user(uid))
    assert r.status_code == 201, r.text
    return r.json()


def _make_due(pid: int) -> None:
    """What the clock does: the next scheduled check arrives."""
    with SessionLocal() as db:
        db.get(Portfolio, pid).next_rebalance_at = trading.now()
        db.commit()


def _order(pid, uid=1, **body):
    return client.post(f"/api/paper/portfolios/{pid}/orders", json=body, headers=user(uid))


# ---------------- orders ----------------

def test_buy_then_sell_updates_cash_cost_and_realised_pnl(prices):
    pid = _new()["id"]
    r = _order(pid, symbol="tcs.ns", side="buy", qty=10).json()
    fee = 10 * 2000 * 20 / 1e4
    assert r["qty"] == 10 and r["fees"] == pytest.approx(fee) and r["cash"] == pytest.approx(100_000 - 20_000 - fee)
    prices["TCS.NS"] = 2200.0
    r = _order(pid, symbol="TCS.NS", side="sell", qty=4).json()
    avg = (20_000 + fee) / 10
    sell_fee = 4 * 2200 * 20 / 1e4
    assert r["realized_pnl"] == pytest.approx((2200 - avg) * 4 - sell_fee)
    d = client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()
    pos = d["positions"][0]
    assert pos["qty"] == 6 and pos["avg_cost"] == pytest.approx(avg)
    assert d["equity"] == pytest.approx(d["cash"] + 6 * 2200)
    assert d["stats"]["trades"] == 2 and d["stats"]["fees_paid"] == pytest.approx(fee + sell_fee)


def test_amount_orders_buy_whole_shares(prices):
    pid = _new()["id"]
    assert _order(pid, symbol="INFY.NS", side="buy", amount=5000).json()["qty"] == 3  # 3 x 1500 + fees fits, 4 doesn't
    assert _order(pid, symbol="INFY.NS", side="buy", amount=100).status_code == 422


@pytest.mark.parametrize("body,msg", [
    ({"symbol": "TCS.NS", "side": "buy", "qty": 1000}, "Not enough cash"),
    ({"symbol": "TCS.NS", "side": "sell", "qty": 1}, "can't sell"),
    ({"symbol": "AAPL", "side": "buy", "qty": 1}, "NSE/BSE"),
    ({"symbol": "NOPE.NS", "side": "buy", "qty": 1}, "No live price"),
])
def test_order_errors_are_readable(prices, body, msg):
    pid = _new()["id"]
    r = _order(pid, **body)
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_benchmark_return(prices):
    pid = _new()["id"]
    prices["NIFTYBEES.NS"] = 275.0
    d = client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()
    assert d["benchmark_return"] == pytest.approx(0.10) and d["vs_benchmark"] == pytest.approx(-0.10)
    assert d["curve"][0]["equity"] == 100_000


# ---------------- copy-trading ----------------

def test_copy_portfolio_buys_the_targets(prices, research):
    p = _new(cash=1_000_000, strategy="dalio")
    assert p["copying"] and p["name"].startswith("Copy: Ray Dalio")
    d = client.get(f"/api/paper/portfolios/{p['id']}", headers=user()).json()
    held = {x["symbol"]: x["qty"] for x in d["positions"]}
    assert held == {"TCS.NS": 248, "GOLDBEES.NS": 3731}  # floor(w * equity * 0.995 / price)
    assert d["status_message"].startswith("Rebalanced") and d["next_rebalance_at"]
    assert research.calls[-1] == ("/internal/strategies/dalio/targets", {"market": "in"})
    assert _order(p["id"], symbol="INFY.NS", side="buy", qty=1).status_code == 422  # copy portfolios are automatic


def test_rebalance_sells_what_the_strategy_dropped(prices, research):
    pid = _new(cash=1_000_000, strategy="dalio")["id"]
    research.targets = {"status": "ready", "signature": None, "weights": [{"symbol": "INFY.NS", "name": "Infosys", "weight": 1.0, "price": 1500.0}]}
    trading.rebalance(pid, force=True)
    d = client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()
    assert [x["symbol"] for x in d["positions"]] == ["INFY.NS"]
    sides = [t["side"] for t in d["trades"]]
    assert sides.count("sell") == 2 and d["cash"] >= 0


def test_guru_copy_waits_for_a_new_filing(prices, research):
    research.targets = {"status": "ready", "signature": "acc-1", "weights": [{"symbol": "AAPL", "name": "Apple", "weight": 1.0, "price": 200.0}]}
    pid = _new(market="us", cash=10_000, strategy="buffett_13f")["id"]
    n = len(client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()["trades"])
    _make_due(pid)
    assert trading.rebalance(pid) == "No new filing yet; holding the latest one."
    research.targets = research.targets | {"signature": "acc-2", "weights": [{"symbol": "KO", "name": "Coca-Cola", "weight": 1.0, "price": 60.0}]}
    _make_due(pid)
    assert trading.rebalance(pid).startswith("Rebalanced")
    d = client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()
    assert len(d["trades"]) == n + 2 and [x["symbol"] for x in d["positions"]] == ["KO"]
    assert "new 13F filing" in d["trades"][0]["reason"]


def test_copy_waits_while_research_is_computing(prices, research):
    research.targets = {"status": "running", "message": "Loading prices…"}
    p = _new(cash=50_000, strategy="dalio")
    d = client.get(f"/api/paper/portfolios/{p['id']}", headers=user()).json()
    assert d["positions"] == [] and "Loading prices" in d["status_message"]
    with SessionLocal() as db:
        assert trading.aware(db.get(Portfolio, p["id"]).next_rebalance_at) <= trading.now()  # still due: retried next minute


def test_custom_copy_keeps_params_and_name(prices, research):
    p = _new(cash=50_000, strategy="custom", params={"factors": "quality:1,value:1", "top_n": "10", "rebalance": "w"})
    assert p["strategy"]["name"] == "Custom: quality + value" and p["strategy"]["rebalance"] == "W"
    assert research.calls[-1][1] == {"market": "in", "factors": "quality:1,value:1", "top_n": "10", "rebalance": "w"}


def test_copy_rejects_unknown_or_wrong_market(prices, research):
    r = client.post("/api/paper/portfolios", json={"market": "in", "cash": 1000, "strategy": "buffett_13f"}, headers=user())
    assert r.status_code == 422 and "US" in r.json()["detail"]
    assert client.post("/api/paper/portfolios", json={"market": "in", "cash": 1000, "strategy": "nope"}, headers=user()).status_code == 422


def test_stop_copying_hands_over_control(prices, research):
    pid = _new(cash=1_000_000, strategy="dalio")["id"]
    assert client.post(f"/api/paper/portfolios/{pid}/stop-copying", headers=user()).status_code == 200
    assert _order(pid, symbol="INFY.NS", side="buy", qty=1).status_code == 201
    assert client.post(f"/api/paper/portfolios/{pid}/rebalance", headers=user()).status_code == 422


@pytest.mark.parametrize("freq,after,expected", [
    ("W", datetime(2026, 10, 7, tzinfo=timezone.utc), datetime(2026, 10, 12, 5, tzinfo=timezone.utc)),  # Wed -> Mon
    ("M", datetime(2026, 10, 10, tzinfo=timezone.utc), datetime(2026, 11, 2, 5, tzinfo=timezone.utc)),  # 1 Nov is a Sunday
    ("Q", datetime(2026, 11, 20, tzinfo=timezone.utc), datetime(2027, 1, 1, 5, tzinfo=timezone.utc)),
    ("A", datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2027, 1, 1, 5, tzinfo=timezone.utc)),
    ("F", datetime(2026, 10, 9, tzinfo=timezone.utc), datetime(2026, 10, 12, 5, tzinfo=timezone.utc)),  # Fri -> Mon
])
def test_next_run(freq, after, expected):
    assert trading.next_run(freq, "in", after) == expected


# ---------------- API rules ----------------

def test_login_required_and_portfolios_are_private(prices):
    assert client.get("/api/paper/portfolios").status_code == 403
    assert client.get("/api/paper/portfolios", headers=INTERNAL).status_code == 401
    pid = _new(uid=1)["id"]
    assert client.get(f"/api/paper/portfolios/{pid}", headers=user(2)).status_code == 404
    assert _order(pid, uid=2, symbol="TCS.NS", side="buy", qty=1).status_code == 404
    assert client.delete(f"/api/paper/portfolios/{pid}", headers=user(2)).status_code == 404
    assert [p["id"] for p in client.get("/api/paper/portfolios", headers=user(1)).json()] == [pid]
    assert client.delete(f"/api/paper/portfolios/{pid}", headers=user(1)).status_code == 204
    assert client.get("/api/paper/portfolios", headers=user(1)).json() == []


def test_portfolio_limit(prices):
    for _ in range(3):
        _new()
    r = client.post("/api/paper/portfolios", json={"market": "in", "cash": 1000}, headers=user())
    assert r.status_code == 422 and "up to 3" in r.json()["detail"]


def test_quote_endpoint(prices):
    assert client.get("/api/paper/quote?symbol=tcs.ns&market=in", headers=user()).json()["price"] == 2000.0
    assert client.get("/api/paper/quote?symbol=TCS&market=in", headers=user()).status_code == 422
    assert client.get("/api/paper/quote?symbol=ZZZZ&market=us", headers=user()).status_code == 404


def test_a_second_run_right_after_a_rebalance_does_nothing(prices, research):
    pid = _new(cash=1_000_000, strategy="dalio")["id"]
    n = len(client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()["trades"])
    assert trading.rebalance(pid) == "Not due yet."  # e.g. the scheduler picked it up while setup was running
    assert len(client.get(f"/api/paper/portfolios/{pid}", headers=user()).json()["trades"]) == n
