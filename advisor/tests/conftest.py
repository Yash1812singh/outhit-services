import os
import shutil
import tempfile

# Must run before any advisor module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-advisor-test-")
os.environ["ADVISOR_DATABASE_URL"] = f"sqlite:///{_TMP}/advisor.db"
os.environ["ADVISOR_ANTHROPIC_API_KEY"] = ""
os.environ["ADVISOR_GROQ_API_KEY"] = ""

import pytest  # noqa: E402

INTERNAL = {"X-Internal-Token": "test-internal-token"}


def user_headers(user_id: int = 1) -> dict:
    return {**INTERNAL, "X-User-Id": str(user_id)}


def fake_signal(symbol: str) -> dict:
    """Deterministic score in [-60, 80) so some stocks qualify as bullish satellites and some don't."""
    score = float(sum(map(ord, symbol)) % 140 - 60)
    label = "Bullish" if score >= 15 else "Bearish" if score <= -15 else "Neutral"
    return {"score": score, "signal": label, "price": 100.0, "volatility": 0.2}


class FakeServiceClient:
    """Stands in for outhit_common.client.ServiceClient; answers from a dict of path -> response (or callable)."""

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[tuple] = []

    def get(self, path, ttl=0, timeout=60, params=None):
        self.calls.append((path, params))
        r = self.responses.get(path)
        return r(params) if callable(r) else r

    def post(self, path, json=None, timeout=60):
        self.calls.append((path, json))
        r = self.responses.get(path)
        return r(json) if callable(r) else r


@pytest.fixture
def fake_market(monkeypatch):
    """The market-data service, faked: /api/market/signals scores every requested symbol."""
    from advisor import clients

    def signals(params):
        return {s: fake_signal(s) for s in params["symbols"].split(",")}

    fake = FakeServiceClient({"/api/market/signals": signals})
    monkeypatch.setattr(clients, "market", fake)
    return fake


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    from advisor.db import engine

    engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)
