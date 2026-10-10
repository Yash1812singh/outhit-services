import os
import shutil
import tempfile

# Must run before any broker module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-broker-test-")
os.environ["BROKER_DATABASE_URL"] = f"sqlite:///{_TMP}/broker.db"
os.environ["BROKER_ENCRYPTION_KEY"] = "c2VjcmV0LWtleS1mb3ItdGVzdHMtb25seS0wMDAwMDA="  # 32 url-safe base64 bytes
os.environ["BROKER_REDIRECT_URI"] = "http://localhost:3050/trade/broker/callback"

import pytest  # noqa: E402

from broker import main, upstox  # noqa: E402
from broker.db import Base, engine, init_db  # noqa: E402

INSTRUMENTS = {
    "TCS": {"symbol": "TCS", "name": "TATA CONSULTANCY SERV LT", "key": "NSE_EQ|INE467B01029", "tick": 0.1, "lot": 1},
    "INFY": {"symbol": "INFY", "name": "INFOSYS LIMITED", "key": "NSE_EQ|INE009A01021", "tick": 0.05, "lot": 1},
}
PRICES = {"TCS.NS": 2000.0, "INFY.NS": 1500.0}


class FakeUpstox:
    """Records calls instead of talking to Upstox."""

    placed: list[dict] = []
    cancelled: list[str] = []
    bad_tokens: set[str] = set()
    reject_next: str | None = None

    def __init__(self, token, sandbox):
        self.token, self.sandbox = token, sandbox

    def _auth(self):
        if self.token in FakeUpstox.bad_tokens:
            raise upstox.BrokerError("Upstox rejected the access token. Reconnect your account.", 401)

    def place_order(self, **kw):
        self._auth()
        if FakeUpstox.reject_next:
            msg, FakeUpstox.reject_next = FakeUpstox.reject_next, None
            raise upstox.BrokerError(f"Upstox: {msg}", 422)
        FakeUpstox.placed.append(kw | {"sandbox": self.sandbox, "token": self.token})
        return [f"ORD{len(FakeUpstox.placed)}"]

    def cancel_order(self, order_id):
        self._auth()
        FakeUpstox.cancelled.append(order_id)

    def charges(self, *a):
        return None if self.sandbox else {"charges": {"total": 12.5}}

    def _live(self):
        if self.sandbox:
            raise upstox.BrokerError("Not available in Upstox sandbox mode.", 409)

    def funds(self):
        self._live()
        return {"equity": {"available_margin": 25000.0, "used_margin": 1000.0, "payin_amount": 0, "notional_cash": 24000.0}}

    def payins(self):
        self._live()
        return [{"amount": 10000, "mode": "UPI", "status": "SUCCESS", "created_at": "2026-10-01T10:00:00"}]

    def payouts(self):
        self._live()
        return [{"amount": 2000, "mode": "IMPS", "status": "SUCCESS", "created_at": "2026-10-05T10:00:00"}]

    def holdings(self):
        self._live()
        return [{"trading_symbol": "TCS", "company_name": "TCS", "quantity": 2, "average_price": 1900.0, "last_price": 2000.0, "pnl": 200.0}]

    def positions(self):
        self._live()
        return []

    def order_book(self):
        self._live()
        return [{"order_id": "ORD1", "trading_symbol": "TCS", "transaction_type": "BUY", "quantity": 1, "status": "complete"}]

    def logout(self):
        pass


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield


@pytest.fixture(autouse=True)
def fake_broker(monkeypatch):
    FakeUpstox.placed, FakeUpstox.cancelled, FakeUpstox.bad_tokens, FakeUpstox.reject_next = [], [], set(), None
    monkeypatch.setattr(main, "Upstox", FakeUpstox)
    monkeypatch.setattr(upstox, "lookup", lambda s: INSTRUMENTS.get(s.strip().upper().removesuffix(".NS")))
    monkeypatch.setattr(upstox, "search", lambda q: [x for k, x in INSTRUMENTS.items() if k.startswith(q.upper())])
    monkeypatch.setattr(upstox, "exchange_code", lambda code, key, secret, uri: {"access_token": f"live-token-{code}", "user_id": "AB1234", "user_name": "Test User", "email": "t@example.com"})

    class Quotes:
        def get(self, path, ttl=0, timeout=60, params=None):
            s = (params or {}).get("symbols")
            return {s: {"price": PRICES[s], "trading": True}} if s in PRICES else {}

    monkeypatch.setattr(main, "intelligence", Quotes())
    return FakeUpstox


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)
