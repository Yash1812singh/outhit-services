"""broker service: real buy/sell orders and your funds, through your own Upstox account.

Safety model:
- Outhit never holds money. Orders go to your Upstox account; deposits and withdrawals happen on Upstox.
- Test mode (Upstox sandbox) by default. Live orders need a live connection AND the live-trading switch.
- Every order is previewed, then confirmed once (single-use, expires in 2 minutes).
- Per-order and per-day value limits, and a pause switch, are enforced here before anything is sent.
- API secrets and access tokens are encrypted at rest and never returned to the browser.
"""

import logging
import math
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from broker import upstox, vault
from broker.config import get_settings
from broker.db import Connection, Order, aware, get_db, init_db, now
from broker.upstox import IST, BrokerError, Upstox
from outhit_common.client import ServiceClient
from outhit_common.internal import require_user
from outhit_common.web import CleanJSONResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("broker")
settings = get_settings()
Path("data").mkdir(exist_ok=True)
intelligence = ServiceClient(settings.intelligence_url, "intelligence")
LIVE_ACK = "ENABLE LIVE TRADING"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if not settings.encryption_key:
        log.warning("BROKER_ENCRYPTION_KEY is not set: broker connections are disabled.")
    yield


app = FastAPI(title="Outhit broker", version="1.0.0", lifespan=lifespan, default_response_class=CleanJSONResponse)


@app.get("/health")
def health() -> dict:
    return {"service": "broker", "status": "ok", "encryption": bool(settings.encryption_key)}


def _fail(exc: Exception):
    if isinstance(exc, BrokerError):
        raise HTTPException(exc.status, str(exc)) from exc
    if isinstance(exc, vault.VaultError):
        raise HTTPException(503, str(exc)) from exc
    raise exc


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _conn(db: Session, user: int) -> Connection | None:
    return db.scalar(select(Connection).where(Connection.user_id == user))


def _connected(c: Connection | None) -> bool:
    return bool(c and c.access_token_enc and c.token_expires_at and aware(c.token_expires_at) > now())


def _client(c: Connection | None) -> Upstox:
    if not _connected(c):
        raise HTTPException(409, "Connect your Upstox account first (or reconnect: live logins expire every morning at 3:30 AM IST).")
    try:
        return Upstox(vault.open_(c.access_token_enc), sandbox=c.mode == "sandbox")
    except vault.VaultError as exc:
        _fail(exc)


def _status(c: Connection | None) -> dict:
    if c is None:
        return {"connected": False, "mode": None, "encryption_ready": bool(settings.encryption_key),
                "redirect_uri": settings.redirect_uri, "limits": {"max_order_value": settings.default_max_order_value,
                                                                 "max_daily_value": settings.default_max_daily_value}}
    try:
        key = vault.mask(vault.open_(c.api_key_enc)) if c.api_key_enc else None
    except vault.VaultError:
        key = "unreadable"
    return {
        "connected": _connected(c), "mode": c.mode, "broker": c.broker, "encryption_ready": bool(settings.encryption_key),
        "token_expires_at": aware(c.token_expires_at), "api_key": key, "profile": c.profile or {},
        "live_trading_enabled": c.live_trading_enabled and c.mode == "live", "paused": c.paused,
        "awaiting_login": c.mode == "live" and not _connected(c) and bool(c.oauth_state),
        "limits": {"max_order_value": c.max_order_value, "max_daily_value": c.max_daily_value,
                   "cap_order_value": settings.cap_max_order_value, "cap_daily_value": settings.cap_max_daily_value},
        "redirect_uri": settings.redirect_uri,
    }


def _new_conn(user: int) -> Connection:
    return Connection(user_id=user, mode="sandbox", max_order_value=settings.default_max_order_value, max_daily_value=settings.default_max_daily_value)


@app.get("/api/broker/status")
def status(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    return _status(_conn(db, user))


class SandboxConnect(BaseModel):
    access_token: str = Field(min_length=20, max_length=4000, description="Sandbox token from account.upstox.com/developer/apps (Sandbox tab)")


@app.post("/api/broker/connect/sandbox")
async def connect_sandbox(body: SandboxConnect, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    token = body.access_token.strip()
    # Check the token: cancelling a non-existent order fails with "invalid token" (401) only if the token is bad.
    try:
        await run_in_threadpool(Upstox(token, sandbox=True).cancel_order, "0")
    except BrokerError as exc:
        if exc.status == 401:
            raise HTTPException(422, "Upstox rejected this sandbox token. Generate a new one in the Sandbox tab of your Upstox developer apps.") from exc
    c = _conn(db, user) or _new_conn(user)
    try:
        c.access_token_enc = vault.seal(token)
    except vault.VaultError as exc:
        _fail(exc)
    c.mode, c.broker, c.live_trading_enabled = "sandbox", "upstox", False
    c.api_key_enc = c.api_secret_enc = c.oauth_state = None
    c.token_expires_at = now() + timedelta(days=30)
    c.profile = {"user_name": "Upstox sandbox", "note": "Test mode: orders are simulated by Upstox; no real money moves."}
    db.add(c)
    db.commit()
    return _status(c)


class LiveStart(BaseModel):
    api_key: str = Field(min_length=8, max_length=200)
    api_secret: str = Field(min_length=8, max_length=200)


@app.post("/api/broker/connect/live/start")
def connect_live_start(body: LiveStart, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    """Save your Upstox app's key and secret, and get the Upstox login URL."""
    c = _conn(db, user) or _new_conn(user)
    try:
        c.api_key_enc, c.api_secret_enc = vault.seal(body.api_key.strip()), vault.seal(body.api_secret.strip())
    except vault.VaultError as exc:
        _fail(exc)
    c.mode, c.broker = "live", "upstox"
    c.access_token_enc, c.token_expires_at, c.live_trading_enabled = None, None, False
    c.oauth_state, c.oauth_state_at = secrets.token_urlsafe(24), now()
    db.add(c)
    db.commit()
    return {"authorize_url": upstox.authorize_url(body.api_key.strip(), settings.redirect_uri, c.oauth_state), "redirect_uri": settings.redirect_uri}


@app.post("/api/broker/connect/live/login")
def connect_live_login(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    """Log in again (each morning) using the saved key and secret."""
    c = _conn(db, user)
    if c is None or c.mode != "live" or not c.api_key_enc:
        raise HTTPException(409, "Save your Upstox API key and secret first.")
    c.oauth_state, c.oauth_state_at = secrets.token_urlsafe(24), now()
    db.commit()
    try:
        return {"authorize_url": upstox.authorize_url(vault.open_(c.api_key_enc), settings.redirect_uri, c.oauth_state)}
    except vault.VaultError as exc:
        _fail(exc)


class LiveCallback(BaseModel):
    code: str = Field(min_length=1, max_length=500)
    state: str = Field(min_length=1, max_length=200)


@app.post("/api/broker/connect/live/callback")
async def connect_live_callback(body: LiveCallback, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    if c is None or not c.oauth_state or not secrets.compare_digest(c.oauth_state, body.state):
        raise HTTPException(403, "This login link doesn't match your session. Start the Upstox connection again.")
    if now() - aware(c.oauth_state_at) > timedelta(minutes=15):
        raise HTTPException(403, "The Upstox login took too long. Start again.")
    c.oauth_state = None
    db.commit()
    try:
        data = await run_in_threadpool(upstox.exchange_code, body.code, vault.open_(c.api_key_enc), vault.open_(c.api_secret_enc), settings.redirect_uri)
        c.access_token_enc = vault.seal(data["access_token"])
    except (BrokerError, vault.VaultError) as exc:
        db.commit()
        _fail(exc)
    c.token_expires_at = upstox.token_expiry()
    c.profile = {k: data.get(k) for k in ("user_id", "user_name", "email", "broker", "exchanges", "products") if data.get(k) is not None}
    db.commit()
    return _status(c)


class LiveSwitch(BaseModel):
    enable: bool
    acknowledge: str = ""


@app.post("/api/broker/live-trading")
def live_trading(body: LiveSwitch, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    if body.enable:
        if c is None or c.mode != "live" or not _connected(c):
            raise HTTPException(409, "Connect and log in to your live Upstox account first.")
        if body.acknowledge.strip().upper() != LIVE_ACK:
            raise HTTPException(422, f'Type "{LIVE_ACK}" to confirm that real money will be used.')
        c.live_trading_enabled, c.live_enabled_at = True, now()
    elif c is not None:
        c.live_trading_enabled = False
    db.commit()
    return _status(c)


class Limits(BaseModel):
    max_order_value: float = Field(gt=0)
    max_daily_value: float = Field(gt=0)


@app.put("/api/broker/limits")
def set_limits(body: Limits, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    if c is None:
        raise HTTPException(409, "Connect a broker first.")
    if body.max_order_value > settings.cap_max_order_value or body.max_daily_value > settings.cap_max_daily_value:
        raise HTTPException(422, f"Limits can be at most ₹{settings.cap_max_order_value:,.0f} per order and ₹{settings.cap_max_daily_value:,.0f} per day.")
    if body.max_order_value > body.max_daily_value:
        raise HTTPException(422, "The per-order limit can't be above the daily limit.")
    c.max_order_value, c.max_daily_value = body.max_order_value, body.max_daily_value
    db.commit()
    return _status(c)


class Pause(BaseModel):
    paused: bool


@app.post("/api/broker/pause")
def pause(body: Pause, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    """Emergency stop: no orders can be sent while paused."""
    c = _conn(db, user)
    if c is None:
        raise HTTPException(409, "Connect a broker first.")
    c.paused = body.paused
    db.commit()
    return _status(c)


@app.delete("/api/broker/connection", status_code=204)
async def disconnect(user: int = Depends(require_user), db: Session = Depends(get_db)) -> None:
    c = _conn(db, user)
    if c is None:
        return
    if _connected(c) and c.mode == "live":
        try:
            await run_in_threadpool(Upstox(vault.open_(c.access_token_enc), sandbox=False).logout)
        except vault.VaultError:
            pass
    db.delete(c)  # wipes the encrypted key, secret and token
    db.commit()


# ---------------------------------------------------------------------------
# Orders: preview -> confirm
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0, le=1_000_000)
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET"
    product: Literal["D", "I"] = "D"
    price: float | None = Field(default=None, gt=0)
    trigger_price: float | None = Field(default=None, gt=0)


def _round_tick(v: float, tick: float) -> float:
    return round(round(v / tick) * tick, 2) if tick > 0 else round(v, 2)


def _day_start_utc() -> datetime:
    t = datetime.now(timezone.utc).astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return t.astimezone(timezone.utc)


def _used_today(db: Session, user: int, mode: str) -> float:
    return db.scalar(select(func.coalesce(func.sum(Order.est_value), 0.0)).where(
        Order.user_id == user, Order.mode == mode, Order.status == "placed", Order.sent_at >= _day_start_utc())) or 0.0


def _check_can_trade(c: Connection | None) -> None:
    if c is None or not _connected(c):
        raise HTTPException(409, "Connect your Upstox account first (live logins expire every morning at 3:30 AM IST).")
    if c.paused:
        raise HTTPException(423, "Trading is paused. Resume it to send orders.")
    if c.mode == "live" and not c.live_trading_enabled:
        raise HTTPException(409, "Live trading is switched off. Turn it on above to place real orders.")


@app.get("/api/broker/instruments")
async def instruments(q: str = Query(min_length=1, max_length=30), user: int = Depends(require_user)) -> list[dict]:
    rows = await run_in_threadpool(upstox.search, q)
    return [{"symbol": f"{x['symbol']}.NS", "name": x["name"], "tick": x["tick"]} for x in rows]


@app.post("/api/broker/orders/preview")
async def preview(body: OrderRequest, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    _check_can_trade(c)
    ins = await run_in_threadpool(upstox.lookup, body.symbol)
    if ins is None:
        raise HTTPException(422, f"{body.symbol.upper()} isn't an NSE equity Upstox can trade. Use a symbol like TCS or TCS.NS.")
    if body.order_type in ("LIMIT", "SL") and not body.price:
        raise HTTPException(422, "Limit and stop-loss limit orders need a price.")
    if body.order_type in ("SL", "SL-M") and not body.trigger_price:
        raise HTTPException(422, "Stop-loss orders need a trigger price.")
    sym = f"{ins['symbol']}.NS"
    q = (intelligence.get("/api/ai/quotes", ttl=15, timeout=30, params={"symbols": sym}) or {}).get(sym) or {}
    last = q.get("price")
    price = _round_tick(body.price, ins["tick"]) if body.price else 0.0
    trigger = _round_tick(body.trigger_price, ins["tick"]) if body.trigger_price else 0.0
    ref = price or trigger or last
    if not ref:
        raise HTTPException(422, f"No price available for {sym} right now; use a limit order.")
    if last and price and abs(price / last - 1) > 0.2:
        raise HTTPException(422, f"Your price ₹{price:,.2f} is more than 20% away from the last price ₹{last:,.2f}. Check for a typo.")
    est = body.qty * ref * (1.02 if body.order_type == "MARKET" and body.side == "BUY" else 1)  # room for market slippage
    if est > c.max_order_value:
        raise HTTPException(422, f"This order (~₹{est:,.0f}) is above your per-order limit of ₹{c.max_order_value:,.0f}. Change it under Safety limits.")
    used = _used_today(db, user, c.mode)
    if used + est > c.max_daily_value:
        raise HTTPException(422, f"This order would take today's total to ₹{used + est:,.0f}, above your daily limit of ₹{c.max_daily_value:,.0f}.")
    client = _client(c)
    charges = await run_in_threadpool(client.charges, ins["key"], body.qty, body.product, body.side, ref)
    o = Order(user_id=user, mode=c.mode, nonce=secrets.token_urlsafe(24), status="preview", symbol=sym, name=ins["name"], instrument_key=ins["key"],
              side=body.side, qty=body.qty, order_type=body.order_type, product=body.product, price=price, trigger_price=trigger,
              ref_price=ref, est_value=est)
    db.add(o)
    db.commit()
    total_charges = None
    if isinstance(charges, dict):
        total_charges = (charges.get("charges") or {}).get("total")
    return {
        "preview_id": o.id, "nonce": o.nonce, "expires_in": settings.preview_ttl_seconds, "mode": c.mode, "live": c.mode == "live",
        "symbol": sym, "name": ins["name"], "side": body.side, "qty": body.qty, "order_type": body.order_type, "product": body.product,
        "price": price or None, "trigger_price": trigger or None, "last_price": last, "market_open": q.get("trading"),
        "estimated_value": est, "estimated_charges": total_charges, "used_today": used, "daily_limit": c.max_daily_value,
        "warning": ("REAL ORDER: this uses real money in your Upstox account." if c.mode == "live"
                    else "Test order: sent to the Upstox sandbox; no real money moves."),
    }


class Confirm(BaseModel):
    nonce: str = Field(min_length=10, max_length=64)


@app.post("/api/broker/orders/{preview_id}/confirm")
async def confirm(preview_id: int, body: Confirm, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    o = db.get(Order, preview_id)
    if o is None or o.user_id != user or not secrets.compare_digest(o.nonce, body.nonce):
        raise HTTPException(404, "Order preview not found.")
    if o.status != "preview":
        raise HTTPException(409, f"This order was already {o.status}.")
    if now() - aware(o.created_at) > timedelta(seconds=settings.preview_ttl_seconds):
        o.status = "expired"
        db.commit()
        raise HTTPException(410, "This preview expired (prices move). Review the order again.")
    c = _conn(db, user)
    _check_can_trade(c)  # re-check: the switch, pause or login may have changed since the preview
    if c.mode != o.mode:
        raise HTTPException(409, "Your broker mode changed since the preview. Review the order again.")
    if _used_today(db, user, c.mode) + o.est_value > c.max_daily_value:
        raise HTTPException(422, "This order would now exceed your daily limit.")
    o.status = "sending"  # claim it, so a double-click can't send it twice
    db.commit()
    client = _client(c)
    try:
        ids = await run_in_threadpool(client.place_order, instrument_key=o.instrument_key, side=o.side, qty=o.qty, order_type=o.order_type,
                                      product=o.product, price=o.price, trigger_price=o.trigger_price, tag=f"outhit-{o.id}")
        o.status, o.broker_order_ids, o.message = "placed", ids, None
    except BrokerError as exc:
        o.status, o.message = ("rejected" if exc.status in (409, 422) else "error"), str(exc)[:500]
    o.sent_at = now()
    db.commit()
    log.info("order %s user=%s mode=%s %s %s x%s -> %s %s", o.id, user, o.mode, o.side, o.symbol, o.qty, o.status, o.broker_order_ids)
    if o.status != "placed":
        raise HTTPException(422, o.message or "Upstox didn't accept the order.")
    return {"status": "placed", "order_ids": o.broker_order_ids, "mode": o.mode, "symbol": o.symbol, "side": o.side, "qty": o.qty,
            "message": "Order sent to Upstox" + (" sandbox (test)." if o.mode == "sandbox" else ". Check its status in Orders.")}


@app.delete("/api/broker/orders/{order_id}")
async def cancel(order_id: str, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    if c is None or not _connected(c):
        raise HTTPException(409, "Connect your Upstox account first.")
    mine = [o for o in db.scalars(select(Order).where(Order.user_id == user, Order.status == "placed")).all() if order_id in (o.broker_order_ids or [])]
    if c.mode == "sandbox" and not mine:
        raise HTTPException(404, "Order not found.")
    try:
        await run_in_threadpool(_client(c).cancel_order, order_id)
    except BrokerError as exc:
        _fail(exc)
    for o in mine:
        o.status = "cancelled"
    db.commit()
    return {"status": "cancel requested", "order_id": order_id}


def _local_orders(db: Session, user: int, mode: str, limit: int = 100) -> list[dict]:
    rows = db.scalars(select(Order).where(Order.user_id == user, Order.mode == mode, Order.status != "preview")
                      .order_by(Order.created_at.desc()).limit(limit)).all()
    return [{"id": o.id, "order_ids": o.broker_order_ids, "symbol": o.symbol, "name": o.name, "side": o.side, "qty": o.qty, "order_type": o.order_type,
             "product": o.product, "price": o.price or None, "trigger_price": o.trigger_price or None, "est_value": o.est_value,
             "status": o.status, "message": o.message, "sent_at": aware(o.sent_at), "created_at": aware(o.created_at)} for o in rows]


@app.get("/api/broker/orders")
async def orders(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    c = _conn(db, user)
    if c is None:
        raise HTTPException(409, "Connect a broker first.")
    out = {"mode": c.mode, "sent_from_outhit": _local_orders(db, user, c.mode), "broker_order_book": None, "note": None}
    if c.mode == "sandbox":
        out["note"] = "Upstox's sandbox doesn't report order status; this is the list of test orders Outhit sent."
    elif _connected(c):
        try:
            book = await run_in_threadpool(_client(c).order_book)
            out["broker_order_book"] = [{k: x.get(k) for k in ("order_id", "trading_symbol", "transaction_type", "quantity", "filled_quantity",
                                                                 "pending_quantity", "order_type", "product", "price", "average_price",
                                                                 "trigger_price", "status", "status_message", "order_timestamp", "tag")} for x in book]
        except BrokerError as exc:
            out["note"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Account: holdings, positions, funds (live only)
# ---------------------------------------------------------------------------

@app.get("/api/broker/portfolio")
async def portfolio(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    client = _client(_conn(db, user))
    try:
        holdings, positions = await run_in_threadpool(client.holdings), await run_in_threadpool(client.positions)
    except BrokerError as exc:
        _fail(exc)
    h = [{"symbol": f"{x.get('trading_symbol') or x.get('tradingsymbol')}.NS", "name": x.get("company_name"), "qty": x.get("quantity"),
          "avg_price": x.get("average_price"), "last_price": x.get("last_price"), "pnl": x.get("pnl"),
          "day_change_pct": x.get("day_change_percentage"), "value": (x.get("quantity") or 0) * (x.get("last_price") or 0)} for x in holdings]
    p = [{"symbol": f"{x.get('trading_symbol') or x.get('tradingsymbol')}.NS", "qty": x.get("quantity"), "product": x.get("product"),
          "avg_price": x.get("average_price"), "last_price": x.get("last_price"), "pnl": x.get("pnl")} for x in positions]
    return {"holdings": h, "positions": p, "invested": sum((x["qty"] or 0) * (x["avg_price"] or 0) for x in h),
            "value": sum(x["value"] for x in h), "pnl": sum(x["pnl"] or 0 for x in h)}


@app.get("/api/broker/funds")
async def funds(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    """Cash and margin from Upstox, plus deposit/withdrawal history. Money moves only on Upstox itself."""
    client = _client(_conn(db, user))
    links = {"add_funds": settings.upstox_web_url, "withdraw": settings.upstox_web_url, "help": settings.upstox_funds_help_url}
    try:
        f = await run_in_threadpool(client.funds)
    except BrokerError as exc:
        _fail(exc)
    eq = f.get("equity") or {}
    history: list[dict] = []
    for kind, fn in (("deposit", client.payins), ("withdrawal", client.payouts)):
        try:
            rows = await run_in_threadpool(fn)
        except BrokerError:
            continue
        for r in rows if isinstance(rows, list) else []:
            history.append({"type": kind} | {k: r.get(k) for k in ("amount", "mode", "status", "bank_name", "transaction_id", "created_at", "eta", "reason")})
    history.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return {"available": eq.get("available_margin"), "used": eq.get("used_margin"), "payin_today": eq.get("payin_amount"),
            "withdrawable": eq.get("notional_cash"), "history": history[:100], "links": links,
            "note": "Deposits and withdrawals happen securely on Upstox. Outhit never holds your money."}
