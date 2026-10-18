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


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_logout_revokes_only_that_token():
    _register("out@example.com")
    login = lambda: client.post("/api/auth/login", json={"email": "out@example.com", "password": "Str0ng!pass"}).json()["access_token"]
    first, second = login(), login()
    assert client.post("/api/auth/logout", headers=_bearer(first)).status_code == 204
    assert client.get("/api/auth/me", headers=_bearer(first)).status_code == 401
    assert client.get("/api/auth/me", headers=_bearer(second)).status_code == 200  # other sessions stay signed in
    assert client.post("/api/auth/logout").status_code == 204  # logging out without a token is harmless


def test_revoked_token_is_forwarded_as_anonymous(upstream):
    token = _token("revoked@example.com")
    client.post("/api/auth/logout", headers=_bearer(token))
    client.get("/api/paper/portfolios", headers=_bearer(token))
    assert "x-user-id" not in upstream[-1].headers


@pytest.fixture
def google(monkeypatch):
    """Enable Google sign-in and stand in for Google's signature check."""
    monkeypatch.setattr(auth.settings, "google_client_id", "test-client.apps.googleusercontent.com")
    tokens: dict[str, dict] = {}

    def verify(credential: str) -> dict:
        if credential not in tokens:
            raise auth.jwt.InvalidSignatureError("bad signature")
        return tokens[credential]

    monkeypatch.setattr(auth, "verify_google_token", verify)
    return tokens


def test_google_config_reports_client_id(google):
    assert client.get("/api/auth/config").json() == {"google_client_id": "test-client.apps.googleusercontent.com"}


def test_google_disabled_without_client_id(monkeypatch):
    monkeypatch.setattr(auth.settings, "google_client_id", "")
    assert client.get("/api/auth/config").json() == {"google_client_id": None}
    assert client.post("/api/auth/google", json={"credential": "x" * 40}).status_code == 503


def test_google_sign_up_then_sign_in(google):
    google["cred-new-user-0000000000"] = {"sub": "g-1", "email": "Gina@Gmail.com", "email_verified": True, "given_name": "Gina", "family_name": "Lee"}
    r = client.post("/api/auth/google", json={"credential": "cred-new-user-0000000000"})
    assert r.status_code == 200, r.text
    user = r.json()["user"]
    assert user["email"] == "gina@gmail.com" and user["first_name"] == "Gina"
    assert user["google_linked"] and not user["has_password"]
    again = client.post("/api/auth/google", json={"credential": "cred-new-user-0000000000"}).json()["user"]
    assert again["id"] == user["id"]
    # No password was set, so password login explains what to do instead.
    pw = client.post("/api/auth/login", json={"email": "gina@gmail.com", "password": "whatever-123"})
    assert pw.status_code == 401 and "Google" in pw.json()["detail"]


def test_google_links_existing_password_account(google):
    uid = _register("link@example.com").json()["user"]["id"]
    google["cred-link-000000000000000"] = {"sub": "g-2", "email": "link@example.com", "email_verified": True}
    r = client.post("/api/auth/google", json={"credential": "cred-link-000000000000000"})
    assert r.json()["user"]["id"] == uid and r.json()["user"]["google_linked"] and r.json()["user"]["has_password"]
    assert client.post("/api/auth/login", json={"email": "link@example.com", "password": "Str0ng!pass"}).status_code == 200


def test_google_rejects_bad_or_unverified_tokens(google):
    assert client.post("/api/auth/google", json={"credential": "forged-credential-00000000"}).status_code == 401
    google["cred-unverified-000000000"] = {"sub": "g-3", "email": "u@example.com", "email_verified": False}
    assert client.post("/api/auth/google", json={"credential": "cred-unverified-000000000"}).status_code == 401


def test_verify_google_token_checks_audience_and_issuer(monkeypatch):
    """The real verifier, with a locally generated key standing in for Google's."""
    import time

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(auth._google_keys, "get_signing_key_from_jwt", lambda _t: type("K", (), {"key": key.public_key()})())
    monkeypatch.setattr(auth.settings, "google_client_id", "ours.apps.googleusercontent.com")
    base = {"sub": "1", "email": "a@b.com", "exp": int(time.time()) + 60, "iss": "https://accounts.google.com", "aud": "ours.apps.googleusercontent.com"}
    sign = lambda **kw: jwt.encode({**base, **kw}, key, algorithm="RS256")

    assert auth.verify_google_token(sign())["sub"] == "1"
    for bad in ({"aud": "someone-else"}, {"iss": "https://evil.example"}, {"exp": int(time.time()) - 600}):
        with pytest.raises(jwt.PyJWTError):
            auth.verify_google_token(sign(**bad))


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
