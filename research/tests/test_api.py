"""research endpoints and its HTTP client for the intelligence service."""

from fastapi.testclient import TestClient

from research import intel, recommend
from research.main import app

client = TestClient(app)


class FakeClient:
    """Stands in for outhit_common.client.ServiceClient."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def get(self, path, ttl=0, timeout=60, params=None):
        self.calls.append(("GET", path, params))
        return self.responses.get(path)

    def post(self, path, json=None, timeout=60):
        self.calls.append(("POST", path, json))
        return self.responses.get(path)


def test_health():
    assert client.get("/health").json() == {"service": "research", "status": "ok"}


def test_picks_track_record_endpoint():
    recommend.record("in", "short_term", {"bullish": [{"symbol": "A.NS", "price": 10.0, "score": 50}], "bearish": []})
    r = client.get("/api/ai/picks-track-record")
    assert r.status_code == 200
    body = r.json()
    assert body["picks_logged"] == 1
    assert body["horizons"]["short_term"]["5d"] == {"evaluated": 0}


def test_input_validation():
    assert client.get("/api/ai/rockets/in", params={"kind": "moonshot"}).status_code == 422
    assert client.get("/api/ai/portfolio/in", params={"risk": "yolo"}).status_code == 422
    assert client.get("/api/ai/portfolio/in", params={"amount": -5}).status_code == 422


def test_ai_view_and_price_come_from_the_intelligence_service(monkeypatch):
    fake = FakeClient({
        "/internal/outlooks": {"X": {"score": 30.0, "label": "Bullish", "reputation": {"score": 70, "grade": "Good"}, "news_7d": 4,
                                     "components": [{"key": "news", "value": 0.4}, {"key": "events", "value": -0.1}]}},
        "/api/ai/quotes": {"X": {"price": 123.5}},
    })
    monkeypatch.setattr(intel, "_client", fake)
    view = recommend._ai_view("X")
    assert view == {"score": 30.0, "label": "Bullish", "reputation": {"score": 70, "grade": "Good"}, "news": 0.4, "events": -0.1, "news_7d": 4}
    assert recommend._ai_view("UNKNOWN") is None
    assert recommend._price("X", 1.0) == 123.5
    assert recommend._price("UNKNOWN", 1.0) == 1.0
    assert ("GET", "/api/ai/quotes", {"symbols": "X"}) in fake.calls


def test_intel_degrades_when_intelligence_is_down(monkeypatch):
    monkeypatch.setattr(intel, "_client", FakeClient({}))
    assert intel.latest(["X"]) == {}
    assert intel.quote("X") is None


def test_track_posts_symbols(monkeypatch):
    fake = FakeClient({})
    monkeypatch.setattr(intel, "_client", fake)
    intel.track(["A.NS", "B.NS"])
    assert fake.calls == [("POST", "/internal/track", ["A.NS", "B.NS"])]


def test_real_client_sends_the_internal_token(monkeypatch):
    """ServiceClient calls are authenticated with the shared internal token (no network: httpx is faked)."""
    import httpx

    from outhit_common import client as client_mod

    seen = {}

    def fake_get(url, params=None, timeout=None, headers=None):
        seen.update(url=url, params=params, headers=headers)
        return httpx.Response(200, json={"X": {"price": 1.0}}, request=httpx.Request("GET", url))

    monkeypatch.setattr(client_mod.httpx, "get", fake_get)
    c = client_mod.ServiceClient("http://intelligence.test/", "intelligence")
    assert c.get("/api/ai/quotes", params={"symbols": "X"}) == {"X": {"price": 1.0}}
    assert seen["url"] == "http://intelligence.test/api/ai/quotes"
    assert seen["headers"]["X-Internal-Token"] == "test-internal-token"
