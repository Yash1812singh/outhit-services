import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import auth, main

client = TestClient(main.app)


def _register(email="ana@example.com", password="Str0ng!pass"):
    return client.post("/api/auth/register", json={"first_name": "Ana", "email": email, "password": password})


def _token(email: str) -> str:
    r = _register(email)
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


# ---------------- auth ----------------

def test_register_login_me():
    r = _register()
    assert r.status_code == 201
    assert r.json()["user"]["email"] == "ana@example.com"
    assert _register().status_code == 409
    assert _register("ANA@example.com").status_code == 409  # emails are case-insensitive

    bad = client.post("/api/auth/login", json={"email": "ana@example.com", "password": "wrong-pass"})
    assert bad.status_code == 401
    unknown = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "Str0ng!pass"})
    assert unknown.status_code == 401

    ok = client.post("/api/auth/login", json={"email": "ANA@example.com", "password": "Str0ng!pass"})
    assert ok.status_code == 200
    token = ok.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["first_name"] == "Ana"


def test_me_rejects_missing_or_forged_tokens():
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    import jwt

    forged = jwt.encode({"sub": "1"}, "some-other-secret-that-is-long-enough!!", algorithm="HS256")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_register_validates_input():
    assert _register("not-an-email").status_code == 422
    assert _register("short@example.com", password="short").status_code == 422


def test_passwords_are_hashed():
    _register("hash@example.com")
    with auth.SessionLocal() as db:
        u = db.query(auth.User).filter_by(email="hash@example.com").one()
    assert u.password_hash != "Str0ng!pass" and auth.verify_password("Str0ng!pass", u.password_hash)


# ---------------- routing ----------------

@pytest.mark.parametrize("path,service", [
    ("/api/ai/picks-track-record", "research"),
    ("/api/ai/picks/in", "research"),
    ("/api/ai/rockets/in", "research"),
    ("/api/ai/rockets/us/extra", "research"),
    ("/api/ai/research/in", "research"),
    ("/api/ai/regime/in", "research"),
    ("/api/ai/portfolio/in", "research"),
    ("/api/ai/funds", "research"),
    ("/api/ai/allocation", "research"),
    ("/api/ai/stream", "intelligence"),
    ("/api/ai/outlook/AAPL", "intelligence"),
    ("/api/ai/quotes", "intelligence"),
    ("/api/ai/model", "intelligence"),
    ("/api/crypto/overview", "market-data"),
    ("/api/crypto", "market-data"),
    ("/api/market/asset/AAPL", "market-data"),
    ("/api/stocks/in/companies", "market-data"),
    ("/api/global/overview", "market-data"),
    ("/api/predict/in", "market-data"),
    ("/api/advisor/plan", "advisor"),
    ("/api/ai/strategies", "research"),
    ("/api/ai/backtest/buffett", "research"),
    ("/api/ai/gurus/buffett_13f/filings", "research"),
    ("/api/paper/portfolios", "paper"),
    ("/api/paper/portfolios/3/orders", "paper"),
    ("/api/broker/status", "broker"),
    ("/api/broker/orders/preview", "broker"),
])
def test_routing_table(path, service, upstream):
    r = client.get(path, params={"symbols": "AAPL"})
    assert r.status_code == 200, r.text
    sent = upstream[-1]
    assert sent.url.host == f"{service}.test"
    assert sent.url.path == path
    assert sent.url.query == b"symbols=AAPL"


def test_prefixes_match_whole_segments_only(upstream):
    # "/api/ai/picksX" is not "/api/ai/picks": it falls through to the /api/ai catch-all (intelligence).
    assert client.get("/api/ai/picksX").json()["service"] == "intelligence"
    assert client.get("/api/cryptoX").status_code == 404
    assert client.get("/api/unknown/thing").status_code == 404
    assert upstream == [upstream[0]]  # only the first request reached a service


def test_stream_is_passed_through(upstream):
    r = client.get("/api/ai/stream", params={"symbols": "AAPL"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.text.startswith("event: update")


def test_post_body_is_forwarded(upstream):
    r = client.post("/api/advisor/plan", json={"age": 30})
    assert r.status_code == 200
    assert upstream[-1].method == "POST"
    assert upstream[-1].content == b'{"age":30}'


# ---------------- trusted headers ----------------

def test_client_supplied_trust_headers_are_stripped(upstream):
    r = client.get("/api/advisor/profile", headers={"X-User-Id": "999", "X-Internal-Token": "attacker-guess"})
    assert r.status_code == 200
    sent = upstream[-1].headers
    assert sent["x-internal-token"] == "test-internal-token"
    assert "x-user-id" not in sent  # anonymous: the spoofed id is gone


def test_valid_jwt_becomes_x_user_id(upstream):
    token = _token("proxy@example.com")
    uid = auth.user_id_from_request(type("R", (), {"headers": {"authorization": f"Bearer {token}"}})())
    client.get("/api/advisor/profile", headers={"Authorization": f"Bearer {token}", "X-User-Id": "999"})
    sent = upstream[-1].headers
    assert sent["x-user-id"] == str(uid) != "999"
    assert sent["x-internal-token"] == "test-internal-token"


def test_invalid_jwt_is_forwarded_as_anonymous(upstream):
    client.get("/api/advisor/profile", headers={"Authorization": "Bearer garbage"})
    sent = upstream[-1].headers
    assert "x-user-id" not in sent and "authorization" not in sent


def test_unreachable_service_returns_503(monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(main, "_client", httpx.AsyncClient(transport=httpx.MockTransport(down)))
    r = client.get("/api/crypto/overview")
    assert r.status_code == 503 and "market-data" in r.json()["detail"]


def test_health_aggregates_services(monkeypatch):
    def handler(request):
        if request.url.host == "advisor.test":
            return httpx.Response(200, json={"service": "advisor", "status": "ok", "ai_provider": "groq"})
        if request.url.host == "research.test":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"status": "ok"})

    for name in main.SERVICES:
        monkeypatch.setitem(main.SERVICES, name, f"http://{name}.test")
    monkeypatch.setattr(main, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["services"]["research"]["status"] == "down"
    assert body["services"]["market-data"]["status"] == "ok"
    assert body["ai_provider"] == "groq" and body["ai_advisor"] is True


# ---------------- rate limits ----------------

def test_auth_is_rate_limited():
    limit = main.settings.rate_limit_auth
    codes = [client.post("/api/auth/login", json={"email": "x@example.com", "password": "nope-nope"}).status_code for _ in range(limit)]
    assert set(codes) == {401}
    assert client.post("/api/auth/login", json={"email": "x@example.com", "password": "nope-nope"}).status_code == 429
    assert _register("late@example.com").status_code == 429  # register shares the auth budget
    assert client.get("/api/auth/me").status_code == 401  # other routes are unaffected


def test_chat_is_rate_limited(upstream, monkeypatch):
    monkeypatch.setattr(main.settings, "rate_limit_chat", 2)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert [client.post("/api/advisor/chat", json=body).status_code for _ in range(3)] == [200, 200, 429]


def test_stream_is_not_rate_limited(upstream, monkeypatch):
    monkeypatch.setattr(main.settings, "rate_limit_default", 1)
    assert all(client.get("/api/ai/stream", params={"symbols": "A"}).status_code == 200 for _ in range(3))
    assert client.get("/api/crypto/overview").status_code == 200
    assert client.get("/api/crypto/overview").status_code == 429
