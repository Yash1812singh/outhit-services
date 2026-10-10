import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

# Must run before any market_data module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-market-data-test-")
os.environ["MARKET_DATA_DATABASE_URL"] = f"sqlite:///{_TMP}/market_data.db"
os.environ["MARKET_DATA_ENABLE_REFRESHER"] = "false"

INTERNAL = {"X-Internal-Token": "test-internal-token"}


def user_headers(user_id: int = 1) -> dict:
    return {**INTERNAL, "X-User-Id": str(user_id)}


def make_prices(drift: float, vol: float = 0.15, days: int = 1000, seed: int = 0, calendar: bool = False) -> pd.DataFrame:
    """Synthetic daily closes with a given annual drift and volatility."""
    rng = np.random.default_rng(seed)
    ppy = 365 if calendar else 252
    rets = rng.normal(drift / ppy, vol / np.sqrt(ppy), days)
    end = pd.Timestamp("2026-10-08")
    index = pd.date_range(end=end, periods=days, freq="D" if calendar else "B")
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": close, "Volume": 1_000_000}, index=index)


@pytest.fixture
def fake_market(monkeypatch):
    """Replace Yahoo with deterministic synthetic history for every symbol."""
    from market_data.services import market_data

    def get_history(symbols):
        return {s.upper(): make_prices(0.12, seed=sum(map(ord, s)) % 1000, calendar=s.endswith("-USD")) for s in symbols}

    monkeypatch.setattr(market_data, "get_history", get_history)
    return get_history


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)
