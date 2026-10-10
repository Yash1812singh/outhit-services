"""Thin Upstox API client (v2/v3 REST) and the NSE instrument list.

Hosts follow Upstox's official SDK: live data on api.upstox.com, live orders on api-hft.upstox.com,
and everything in sandbox on api-sandbox.upstox.com. Sandbox only supports placing, modifying and
cancelling orders; funds, holdings and the order book need a live connection.
"""

import gzip
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

log = logging.getLogger(__name__)

LIVE_HOST, LIVE_ORDER_HOST, SANDBOX_HOST = "https://api.upstox.com", "https://api-hft.upstox.com", "https://api-sandbox.upstox.com"
AUTH_DIALOG = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
INSTRUMENTS_FILE = Path("data") / "nse_instruments.json"
IST = timezone(timedelta(hours=5, minutes=30))


class BrokerError(Exception):
    def __init__(self, message: str, status: int = 502, code: str | None = None):
        super().__init__(message)
        self.status, self.code = status, code


def _error(r: httpx.Response) -> BrokerError:
    try:
        body = r.json()
        errs = body.get("errors") or []
        msg = "; ".join(e.get("message", "") for e in errs if e.get("message")) or body.get("message") or r.text[:200]
        code = errs[0].get("errorCode") if errs else None
    except Exception:
        msg, code = r.text[:200] or r.reason_phrase, None
    if r.status_code == 401:
        return BrokerError(f"Upstox rejected the access token ({msg}). Reconnect your account.", 401, code)
    return BrokerError(f"Upstox: {msg}", 422 if r.status_code < 500 else 502, code)


def token_expiry(issued: datetime | None = None) -> datetime:
    """Upstox access tokens expire at 3:30 AM IST the next morning."""
    t = (issued or datetime.now(timezone.utc)).astimezone(IST)
    expiry = t.replace(hour=3, minute=30, second=0, microsecond=0)
    if t >= expiry:
        expiry += timedelta(days=1)
    return expiry.astimezone(timezone.utc)


def authorize_url(api_key: str, redirect_uri: str, state: str) -> str:
    return f"{AUTH_DIALOG}?{urlencode({'response_type': 'code', 'client_id': api_key, 'redirect_uri': redirect_uri, 'state': state})}"


def exchange_code(code: str, api_key: str, api_secret: str, redirect_uri: str) -> dict:
    r = httpx.post(TOKEN_URL, timeout=30, headers={"accept": "application/json"},
                   data={"code": code, "client_id": api_key, "client_secret": api_secret, "redirect_uri": redirect_uri, "grant_type": "authorization_code"})
    if r.status_code != 200:
        raise _error(r)
    data = r.json()
    if not data.get("access_token"):
        raise BrokerError("Upstox did not return an access token.")
    return data


class Upstox:
    def __init__(self, token: str, sandbox: bool):
        self.token, self.sandbox = token, sandbox

    def _url(self, path: str, order: bool) -> str:
        if self.sandbox:
            return SANDBOX_HOST + path
        return (LIVE_ORDER_HOST if order else LIVE_HOST) + path

    def _req(self, method: str, path: str, order: bool = False, **kw):
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.token}"}
        if "json" in kw:
            headers["Content-Type"] = "application/json"
        try:
            r = httpx.request(method, self._url(path, order), headers=headers, timeout=20, **kw)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Couldn't reach Upstox: {exc.__class__.__name__}") from exc
        if r.status_code >= 400:
            raise _error(r)
        body = r.json()
        if body.get("status") == "error":
            raise _error(r)
        return body.get("data")

    def _live_only(self, what: str) -> None:
        if self.sandbox:
            raise BrokerError(f"{what} isn't available in Upstox sandbox mode; connect your live account to see it.", 409)

    # orders (sandbox + live)
    def place_order(self, *, instrument_key: str, side: str, qty: int, order_type: str, product: str,
                    price: float = 0, trigger_price: float = 0, tag: str | None = None) -> list[str]:
        body = {"quantity": qty, "product": product, "validity": "DAY", "price": price, "instrument_token": instrument_key,
                "order_type": order_type, "transaction_type": side, "disclosed_quantity": 0, "trigger_price": trigger_price,
                "is_amo": False, "slice": True}
        if tag:
            body["tag"] = tag[:40]
        if order_type in ("MARKET", "SL-M"):
            body["market_protection"] = -1  # Upstox's automatic price protection
        data = self._req("POST", "/v3/order/place", order=True, json=body) or {}
        return [str(x) for x in data.get("order_ids", [])]

    def cancel_order(self, order_id: str) -> None:
        self._req("DELETE", "/v3/order/cancel", order=True, params={"order_id": order_id})

    # account (live only)
    def profile(self) -> dict:
        self._live_only("Your profile")
        return self._req("GET", "/v2/user/profile") or {}

    def funds(self) -> dict:
        self._live_only("Funds")
        return self._req("GET", "/v2/user/get-funds-and-margin", params={"segment": "SEC"}) or {}

    def payins(self) -> list:
        self._live_only("Deposits")
        return self._req("GET", "/v2/user/payments/payin") or []

    def payouts(self) -> list:
        self._live_only("Withdrawals")
        return self._req("GET", "/v2/user/payments/payout") or []

    def holdings(self) -> list:
        self._live_only("Holdings")
        return self._req("GET", "/v2/portfolio/long-term-holdings") or []

    def positions(self) -> list:
        self._live_only("Positions")
        return self._req("GET", "/v2/portfolio/short-term-positions") or []

    def order_book(self) -> list:
        self._live_only("The order book")
        return self._req("GET", "/v2/order/retrieve-all") or []

    def charges(self, instrument_key: str, qty: int, product: str, side: str, price: float) -> dict | None:
        if self.sandbox:
            return None
        try:
            return self._req("GET", "/v2/charges/brokerage", params={"instrument_token": instrument_key, "quantity": qty, "product": product,
                                                                     "transaction_type": side, "price": price})
        except BrokerError:
            return None

    def logout(self) -> None:
        if not self.sandbox:
            try:
                self._req("DELETE", "/v2/logout")
            except BrokerError:
                pass


# ---------------------------------------------------------------------------
# NSE instruments: "TCS" / "TCS.NS" -> NSE_EQ|INE467B01029
# ---------------------------------------------------------------------------

_instruments: dict[str, dict] = {}
_loaded_at = [0.0]
_lock = threading.Lock()


def instruments() -> dict[str, dict]:
    with _lock:
        if _instruments and time.time() - _loaded_at[0] < 24 * 3600:
            return _instruments
        rows = None
        if INSTRUMENTS_FILE.exists() and time.time() - INSTRUMENTS_FILE.stat().st_mtime < 24 * 3600:
            rows = json.loads(INSTRUMENTS_FILE.read_text())
        else:
            try:
                r = httpx.get(INSTRUMENTS_URL, timeout=60)
                r.raise_for_status()
                rows = [{"symbol": x["trading_symbol"], "name": x.get("name") or x["trading_symbol"], "key": x["instrument_key"],
                         "tick": (x.get("tick_size") or 5) / 100, "lot": x.get("lot_size") or 1, "isin": x.get("isin")}
                        for x in json.loads(gzip.decompress(r.content))
                        if x.get("segment") == "NSE_EQ" and x.get("instrument_type") in ("EQ", "BE")]
                INSTRUMENTS_FILE.parent.mkdir(exist_ok=True)
                INSTRUMENTS_FILE.write_text(json.dumps(rows))
            except Exception as exc:
                log.warning("instrument list download failed: %s", exc)
                if INSTRUMENTS_FILE.exists():
                    rows = json.loads(INSTRUMENTS_FILE.read_text())
        if rows:
            _instruments.clear()
            _instruments.update({x["symbol"]: x for x in rows})
            _loaded_at[0] = time.time()
        return _instruments


def lookup(symbol: str) -> dict | None:
    s = symbol.strip().upper().removesuffix(".NS")
    return instruments().get(s)


def search(q: str, limit: int = 10) -> list[dict]:
    q = q.strip().upper().removesuffix(".NS")
    if not q:
        return []
    ins = instruments()
    starts = [x for s, x in ins.items() if s.startswith(q)]
    named = [x for x in ins.values() if q in x["name"].upper() and not x["symbol"].startswith(q)]
    return sorted(starts, key=lambda x: (len(x["symbol"]), x["symbol"]))[:limit] + named[: max(0, limit - len(starts))]
