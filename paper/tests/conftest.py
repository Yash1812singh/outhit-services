import os
import shutil
import tempfile

# Must run before any paper module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-paper-test-")
os.environ["PAPER_DATABASE_URL"] = f"sqlite:///{_TMP}/paper.db"
os.environ["PAPER_ENABLE_SCHEDULER"] = "false"
os.environ["PAPER_MAX_PORTFOLIOS_PER_USER"] = "3"

import pytest  # noqa: E402

from paper import trading  # noqa: E402
from paper.db import Base, engine, init_db  # noqa: E402

PRICES = {"TCS.NS": 2000.0, "INFY.NS": 1500.0, "NIFTYBEES.NS": 250.0, "GOLDBEES.NS": 80.0, "AAPL": 200.0, "KO": 60.0, "SPY": 600.0}
STRATEGIES = {
    "dalio": {"key": "dalio", "name": "All Weather", "person": "Ray Dalio", "kind": "allocation", "rebalance": "Q", "markets": ["in", "us"]},
    "buffett_13f": {"key": "buffett_13f", "name": "Berkshire Hathaway portfolio", "person": "Warren Buffett", "kind": "guru", "rebalance": "F", "markets": ["us"]},
    "custom": {"key": "custom", "name": "Build your own", "person": "You", "kind": "custom", "rebalance": "M", "markets": ["in", "us"]},
}


class FakeResearch:
    """Answers /internal/strategies/{key}/targets from a settable dict."""

    def __init__(self):
        self.targets: dict | None = {"status": "ready", "signature": None, "weights": [
            {"symbol": "TCS.NS", "name": "TCS", "weight": 0.5, "price": 2000.0},
            {"symbol": "GOLDBEES.NS", "name": "Gold ETF", "weight": 0.3, "price": 80.0}]}
        self.calls: list[tuple[str, dict]] = []

    def get(self, path, ttl=0, timeout=60, params=None):
        self.calls.append((path, params or {}))
        if path == "/api/ai/strategies":
            return {"strategies": list(STRATEGIES.values())}
        return self.targets


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield


@pytest.fixture
def prices(monkeypatch):
    p = dict(PRICES)
    monkeypatch.setattr(trading, "quotes", lambda syms: {s: {"price": p[s], "change_pct": 0.5, "trading": True} for s in syms if s in p})
    return p


@pytest.fixture
def research(monkeypatch):
    fake = FakeResearch()
    monkeypatch.setattr(trading, "research", fake)
    monkeypatch.setattr(trading, "rebalance_in_background", lambda pid, force=False: trading.rebalance(pid, force))
    return fake


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)
