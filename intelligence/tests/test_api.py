"""Service endpoints, including the internal ones other services call with the shared token."""

import pytest
from fastapi.testclient import TestClient

from intelligence import outlook, registry
from intelligence.main import app
from intelligence.tests.conftest import INTERNAL

client = TestClient(app)


@pytest.fixture
def one_outlook(monkeypatch):
    monkeypatch.setitem(outlook._latest, "TEST", {"symbol": "TEST", "score": 25.0, "label": "Bullish", "confidence": "Medium", "components": []})
    yield


def test_health():
    assert client.get("/health").json()["service"] == "intelligence"


def test_internal_outlooks_require_the_internal_token(one_outlook):
    assert client.get("/internal/outlooks").status_code == 403
    assert client.get("/internal/outlooks", headers={"X-Internal-Token": "nope"}).status_code == 403
    r = client.get("/internal/outlooks", headers=INTERNAL)
    assert r.status_code == 200 and r.json()["TEST"]["score"] == 25.0
    r = client.get("/internal/outlooks", params={"symbols": "test,OTHER"}, headers=INTERNAL)
    assert list(r.json()) == ["TEST"]


def test_internal_track_requires_the_internal_token(monkeypatch):
    seen = []
    monkeypatch.setattr(registry, "track", lambda syms: seen.append(syms) or len(syms))
    assert client.post("/internal/track", json=["aapl"]).status_code == 403
    assert seen == []
    r = client.post("/internal/track", json=["aapl", "infy.ns"], headers=INTERNAL)
    assert r.status_code == 200 and r.json() == {"added": 2}
    assert seen == [["AAPL", "INFY.NS"]]


def test_track_record_endpoint_is_public():
    r = client.get("/api/ai/track-record")
    assert r.status_code == 200 and r.json()["predictions_logged"] == 0
