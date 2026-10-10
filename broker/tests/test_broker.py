"""Real-trading safety: connections, encryption, preview/confirm, limits, live switch, pause, and account views."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from broker import upstox
from broker.db import Connection, Order, SessionLocal
from broker.main import LIVE_ACK, app

client = TestClient(app)
INTERNAL = {"X-Internal-Token": "test-internal-token"}
SANDBOX_TOKEN = "sandbox-token-abcdefghijklmnopqrstuvwxyz"


def user(uid: int = 1) -> dict:
    return {**INTERNAL, "X-User-Id": str(uid)}


def connect_sandbox(uid: int = 1):
    r = client.post("/api/broker/connect/sandbox", json={"access_token": SANDBOX_TOKEN}, headers=user(uid))
    assert r.status_code == 200, r.text
    return r.json()


def connect_live(uid: int = 1, enable: bool = True):
    start = client.post("/api/broker/connect/live/start", json={"api_key": "my-api-key-123", "api_secret": "my-api-secret-456"}, headers=user(uid)).json()  # gitleaks:allow (fake test credentials)
    state = start["authorize_url"].split("state=")[1]
    r = client.post("/api/broker/connect/live/callback", json={"code": "abc", "state": state}, headers=user(uid))
    assert r.status_code == 200, r.text
    if enable:
        assert client.post("/api/broker/live-trading", json={"enable": True, "acknowledge": LIVE_ACK}, headers=user(uid)).status_code == 200
    return r.json()


def preview(uid: int = 1, **kw):
    body = {"symbol": "TCS", "side": "BUY", "qty": 2, "order_type": "MARKET", "product": "D"} | kw
    return client.post("/api/broker/orders/preview", json=body, headers=user(uid))


def confirm(p: dict, uid: int = 1):
    return client.post(f"/api/broker/orders/{p['preview_id']}/confirm", json={"nonce": p["nonce"]}, headers=user(uid))


# ---------------- connecting ----------------

def test_needs_login_and_starts_disconnected():
    assert client.get("/api/broker/status").status_code == 403
    s = client.get("/api/broker/status", headers=user()).json()
    assert s["connected"] is False and s["encryption_ready"] is True


def test_sandbox_token_is_checked_encrypted_and_never_returned(fake_broker):
    fake_broker.bad_tokens.add("bad-token-0123456789abcdefghij")
    r = client.post("/api/broker/connect/sandbox", json={"access_token": "bad-token-0123456789abcdefghij"}, headers=user())
    assert r.status_code == 422
    s = connect_sandbox()
    assert s["connected"] and s["mode"] == "sandbox" and not s["live_trading_enabled"]
    assert SANDBOX_TOKEN not in str(s)
    with SessionLocal() as db:
        c = db.scalar(select(Connection))
        assert c.access_token_enc and SANDBOX_TOKEN not in c.access_token_enc


def test_live_oauth_checks_state_and_sets_morning_expiry():
    start = client.post("/api/broker/connect/live/start", json={"api_key": "my-api-key-123", "api_secret": "my-api-secret-456"}, headers=user()).json()  # gitleaks:allow (fake test credentials)
    assert start["authorize_url"].startswith(upstox.AUTH_DIALOG) and "redirect_uri=http%3A%2F%2Flocalhost%3A3050%2Ftrade%2Fbroker%2Fcallback" in start["authorize_url"]
    assert client.post("/api/broker/connect/live/callback", json={"code": "abc", "state": "forged"}, headers=user()).status_code == 403
    assert client.post("/api/broker/connect/live/callback", json={"code": "abc", "state": "x"}, headers=user(2)).status_code == 403  # another user
    s = connect_live(enable=False)
    assert s["connected"] and s["mode"] == "live" and s["profile"]["user_name"] == "Test User"
    assert s["api_key"] == "my-a…-123" and "my-api-secret" not in str(s)
    exp = datetime.fromisoformat(str(s["token_expires_at"]).replace("Z", "+00:00")).astimezone(upstox.IST)
    assert (exp.hour, exp.minute) == (3, 30)


def test_token_expiry_is_next_330_am_ist():
    ist = upstox.IST
    assert upstox.token_expiry(datetime(2026, 10, 10, 15, 0, tzinfo=ist)) == datetime(2026, 10, 11, 3, 30, tzinfo=ist)
    assert upstox.token_expiry(datetime(2026, 10, 10, 2, 0, tzinfo=ist)) == datetime(2026, 10, 10, 3, 30, tzinfo=ist)


# ---------------- preview -> confirm ----------------

def test_sandbox_order_flow(fake_broker):
    connect_sandbox()
    p = preview().json()
    assert p["mode"] == "sandbox" and p["estimated_value"] == pytest.approx(2 * 2000 * 1.02) and "Test order" in p["warning"]
    assert fake_broker.placed == []  # preview never sends anything
    r = confirm(p)
    assert r.status_code == 200 and r.json()["order_ids"] == ["ORD1"]
    sent = fake_broker.placed[0]
    assert sent["sandbox"] is True and sent["instrument_key"] == "NSE_EQ|INE467B01029" and sent["side"] == "BUY" and sent["qty"] == 2
    assert confirm(p).status_code == 409  # single use: a double click can't send it twice
    orders = client.get("/api/broker/orders", headers=user()).json()
    assert [o["status"] for o in orders["sent_from_outhit"]] == ["placed"] and "sandbox" in orders["note"]


def test_preview_validation():
    connect_sandbox()
    assert "isn't an NSE equity" in preview(symbol="NOTREAL").json()["detail"]
    assert preview(order_type="LIMIT").status_code == 422  # needs a price
    assert preview(order_type="SL-M").status_code == 422  # needs a trigger
    assert "20% away" in preview(order_type="LIMIT", price=3000).json()["detail"]  # fat-finger guard
    p = preview(order_type="LIMIT", price=1999.97).json()
    assert p["price"] == 2000.0  # rounded to the 0.10 tick


def test_limits_per_order_and_per_day():
    connect_sandbox()
    assert "per-order limit" in preview(qty=30).json()["detail"]  # ~61k > 50k default
    for _ in range(4):
        assert confirm(preview(qty=24).json()).status_code == 200  # ~49k each
    r = preview(qty=24)
    assert r.status_code == 422 and "daily limit" in r.json()["detail"]
    r = client.put("/api/broker/limits", json={"max_order_value": 10_000_000, "max_daily_value": 10_000_000}, headers=user())
    assert r.status_code == 422
    assert client.put("/api/broker/limits", json={"max_order_value": 100_000, "max_daily_value": 50_000}, headers=user()).status_code == 422
    assert client.put("/api/broker/limits", json={"max_order_value": 100_000, "max_daily_value": 400_000}, headers=user()).status_code == 200
    assert preview(qty=24).status_code == 200


def test_expired_preview_is_refused():
    connect_sandbox()
    p = preview().json()
    with SessionLocal() as db:
        db.get(Order, p["preview_id"]).created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        db.commit()
    assert confirm(p).status_code == 410


def test_pause_blocks_orders_even_after_preview(fake_broker):
    connect_sandbox()
    p = preview().json()
    client.post("/api/broker/pause", json={"paused": True}, headers=user())
    assert confirm(p).status_code == 423 and fake_broker.placed == []
    assert preview().status_code == 423
    client.post("/api/broker/pause", json={"paused": False}, headers=user())
    assert preview().status_code == 200


def test_live_orders_need_the_switch(fake_broker):
    connect_live(enable=False)
    assert "Live trading is switched off" in preview().json()["detail"]
    r = client.post("/api/broker/live-trading", json={"enable": True, "acknowledge": "yes"}, headers=user())
    assert r.status_code == 422
    assert client.post("/api/broker/live-trading", json={"enable": True, "acknowledge": LIVE_ACK.lower()}, headers=user()).status_code == 200
    p = preview().json()
    assert p["live"] and "REAL ORDER" in p["warning"] and p["estimated_charges"] == 12.5
    assert confirm(p).status_code == 200
    assert fake_broker.placed[0]["sandbox"] is False and fake_broker.placed[0]["token"] == "live-token-abc"
    # switching off after a preview blocks the confirm
    p2 = preview().json()
    client.post("/api/broker/live-trading", json={"enable": False}, headers=user())
    assert confirm(p2).status_code == 409 and len(fake_broker.placed) == 1


def test_sandbox_cannot_enable_live_trading():
    connect_sandbox()
    assert client.post("/api/broker/live-trading", json={"enable": True, "acknowledge": LIVE_ACK}, headers=user()).status_code == 409


def test_broker_rejections_are_recorded(fake_broker):
    connect_sandbox()
    fake_broker.reject_next = "Insufficient funds"
    r = confirm(preview().json())
    assert r.status_code == 422 and "Insufficient funds" in r.json()["detail"]
    assert client.get("/api/broker/orders", headers=user()).json()["sent_from_outhit"][0]["status"] == "rejected"


def test_orders_are_private_and_cancellable(fake_broker):
    connect_sandbox(1)
    connect_sandbox(2)
    p = preview(1).json()
    assert confirm(p, uid=2).status_code == 404  # someone else's preview
    assert confirm(p).status_code == 200
    assert client.delete("/api/broker/orders/ORD1", headers=user(2)).status_code == 404
    assert client.delete("/api/broker/orders/ORD1", headers=user(1)).status_code == 200
    assert [c for c in fake_broker.cancelled if c != "0"] == ["ORD1"]  # "0" is the token check on connect


# ---------------- account ----------------

def test_funds_and_portfolio_need_a_live_connection():
    connect_sandbox()
    assert client.get("/api/broker/funds", headers=user()).status_code == 409
    assert client.get("/api/broker/portfolio", headers=user()).status_code == 409


def test_live_funds_portfolio_and_order_book():
    connect_live()
    f = client.get("/api/broker/funds", headers=user()).json()
    assert f["available"] == 25000 and {h["type"] for h in f["history"]} == {"deposit", "withdrawal"} and f["links"]["add_funds"]
    pf = client.get("/api/broker/portfolio", headers=user()).json()
    assert pf["holdings"][0]["symbol"] == "TCS.NS" and pf["value"] == 4000 and pf["pnl"] == 200
    assert client.get("/api/broker/orders", headers=user()).json()["broker_order_book"][0]["order_id"] == "ORD1"


def test_disconnect_wipes_credentials():
    connect_live()
    assert client.delete("/api/broker/connection", headers=user()).status_code == 204
    with SessionLocal() as db:
        assert db.scalar(select(Connection)) is None
    assert client.get("/api/broker/status", headers=user()).json()["connected"] is False


def test_instrument_search():
    assert client.get("/api/broker/instruments?q=tc", headers=user()).json()[0]["symbol"] == "TCS.NS"
