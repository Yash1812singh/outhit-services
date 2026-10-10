"""Paper trading: virtual orders at live prices, valuations, and copy-trading strategies.

Orders fill at the latest live price (delayed quotes when the market is closed) and pay realistic
costs. Copy portfolios ask the research service for the strategy's target holdings on its schedule
and trade to match them, exactly as the backtest assumes, so you can see whether it works going forward.
"""

import logging
import math
import re
import threading
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from outhit_common.client import ServiceClient
from paper.config import get_settings
from paper.db import Portfolio, Position, SessionLocal, Snapshot, Trade, aware, now

log = logging.getLogger(__name__)
settings = get_settings()
research = ServiceClient(settings.research_url, "research")
intelligence = ServiceClient(settings.intelligence_url, "intelligence")

FEES_BPS = {"in": 20, "us": 5}  # per side: brokerage + taxes/charges + slippage
CURRENCY = {"in": "INR", "us": "USD"}
BENCHMARK = {"in": "NIFTYBEES.NS", "us": "SPY"}
RUN_HOUR_UTC = {"in": 5, "us": 15}  # mid-morning in Mumbai / New York, when the market is open
SYMBOL_RE = {"in": re.compile(r"^[A-Z0-9&\-]{1,20}\.(NS|BO)$"), "us": re.compile(r"^[A-Z0-9\-]{1,10}$")}
CASH_BUFFER = 0.995  # keep 0.5% aside for costs when rebalancing
_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


class TradeError(ValueError):
    pass


def _lock(pid: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(pid, threading.Lock())


def valid_symbol(market: str, symbol: str) -> bool:
    return bool(SYMBOL_RE[market].match(symbol))


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def quotes(symbols: list[str]) -> dict[str, dict]:
    """Latest quotes from the intelligence service: {symbol: {price, change_pct, trading, name?}}."""
    syms = sorted({s for s in symbols if s})
    out: dict[str, dict] = {}
    for i in range(0, len(syms), 50):
        chunk = syms[i : i + 50]
        data = intelligence.get("/api/ai/quotes", ttl=20, timeout=60, params={"symbols": ",".join(chunk)}) or {}
        for s, q in data.items():
            if isinstance(q, dict) and q.get("price") and math.isfinite(q["price"]) and q["price"] > 0:
                out[s] = q
    return out


def strategies() -> dict[str, dict]:
    data = research.get("/api/ai/strategies", ttl=3600, timeout=30) or {}
    return {s["key"]: s for s in data.get("strategies", [])}


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def execute(db: Session, p: Portfolio, symbol: str, name: str, side: str, qty: float, price: float, reason: str, market_open: bool = True) -> Trade:
    if qty <= 0:
        raise TradeError("Quantity must be at least 1 share.")
    notional = qty * price
    fee = notional * FEES_BPS[p.market] / 1e4
    pos = next((x for x in p.positions if x.symbol == symbol), None)
    realized = None
    if side == "buy":
        if notional + fee > p.cash + 1e-6:
            raise TradeError(f"Not enough cash: this order needs {notional + fee:,.2f} {p.currency}, you have {p.cash:,.2f}.")
        p.cash -= notional + fee
        if pos is None:
            pos = Position(symbol=symbol, name=name or symbol, qty=0, avg_cost=0)
            p.positions.append(pos)
        pos.avg_cost = (pos.qty * pos.avg_cost + notional + fee) / (pos.qty + qty)
        pos.qty += qty
    else:
        if pos is None or pos.qty < qty - 1e-9:
            raise TradeError(f"You hold {pos.qty if pos else 0:g} shares of {symbol}; you can't sell {qty:g}.")
        p.cash += notional - fee
        realized = (price - pos.avg_cost) * qty - fee
        pos.qty -= qty
        if pos.qty <= 1e-9:
            p.positions.remove(pos)
    t = Trade(portfolio_id=p.id, symbol=symbol, name=name or symbol, side=side, qty=qty, price=price, fees=fee,
              realized_pnl=realized, reason=reason, market_open=market_open)
    db.add(t)
    return t


def manual_order(db: Session, p: Portfolio, symbol: str, side: str, qty: int | None, amount: float | None) -> Trade:
    symbol = symbol.strip().upper()
    if p.copying:
        raise TradeError("This portfolio copies a strategy automatically. Stop copying first to trade it yourself.")
    if not valid_symbol(p.market, symbol):
        hint = "an NSE/BSE symbol like TCS.NS" if p.market == "in" else "a US ticker like AAPL"
        raise TradeError(f"Use {hint} in this portfolio.")
    q = quotes([symbol]).get(symbol)
    if not q:
        raise TradeError(f"No live price for {symbol}. Check the symbol.")
    price = q["price"]
    if qty is None:
        if not amount or amount <= 0:
            raise TradeError("Give a quantity or an amount.")
        qty = math.floor(amount / (price * (1 + FEES_BPS[p.market] / 1e4)))
        if qty < 1:
            raise TradeError(f"{amount:,.2f} is less than one share ({price:,.2f}).")
    pos = next((x for x in p.positions if x.symbol == symbol), None)
    name = pos.name if pos else (q.get("name") or symbol)
    with _lock(p.id):
        t = execute(db, p, symbol, name, side, float(qty), price, "Your order", bool(q.get("trading", True)))
        db.commit()
    return t


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def value(p: Portfolio, q: dict[str, dict] | None = None) -> dict:
    q = q if q is not None else quotes([x.symbol for x in p.positions] + [p.benchmark_symbol])
    positions = []
    invested = 0.0
    for x in p.positions:
        quote = q.get(x.symbol)
        price = quote["price"] if quote else x.avg_cost
        val = x.qty * price
        invested += val
        positions.append({"symbol": x.symbol, "name": x.name, "qty": x.qty, "avg_cost": x.avg_cost, "price": price, "value": val,
                          "pnl": val - x.qty * x.avg_cost, "pnl_pct": price / x.avg_cost - 1 if x.avg_cost else None,
                          "change_pct": quote.get("change_pct") if quote else None, "stale": quote is None})
    equity = p.cash + invested
    for pos in positions:
        pos["weight"] = pos["value"] / equity if equity else 0
    bq = q.get(p.benchmark_symbol)
    bench_ret = bq["price"] / p.benchmark_start - 1 if bq and p.benchmark_start else None
    ret = equity / p.initial_cash - 1
    return {"equity": equity, "cash": p.cash, "invested": invested, "pnl": equity - p.initial_cash, "return": ret,
            "benchmark_return": bench_ret, "vs_benchmark": ret - bench_ret if bench_ret is not None else None,
            "benchmark_price": bq["price"] if bq else None,
            "positions": sorted(positions, key=lambda r: -r["value"])}


def snapshot(db: Session, p: Portfolio, v: dict | None = None) -> None:
    v = v or value(p)
    today = date.today()
    snap = db.scalar(select(Snapshot).where(Snapshot.portfolio_id == p.id, Snapshot.day == today))
    if snap is None:
        db.add(Snapshot(portfolio_id=p.id, day=today, equity=v["equity"], benchmark=v["benchmark_price"]))
    else:
        snap.equity, snap.benchmark = v["equity"], v["benchmark_price"] or snap.benchmark


# ---------------------------------------------------------------------------
# Copy-trading
# ---------------------------------------------------------------------------

def next_run(freq: str, market: str, after: datetime | None = None) -> datetime:
    after = after or now()
    hour = RUN_HOUR_UTC[market]
    d = after.date()
    if freq in ("F", "D"):
        nxt = d + timedelta(days=1)
    elif freq == "W":
        nxt = d + timedelta(days=7 - d.weekday())  # next Monday
    else:
        months = {"M": 1, "Q": 3, "A": 12}[freq]
        m0 = (d.month - 1) // months * months  # start of the current period (0-based month)
        y, m = divmod(m0 + months, 12)
        nxt = date(d.year + y, m + 1, 1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return datetime(nxt.year, nxt.month, nxt.day, hour, tzinfo=timezone.utc)


def rebalance(pid: int, force: bool = False) -> str:
    """Trade a copy portfolio to the strategy's current target holdings. Returns a status message."""
    with _lock(pid), SessionLocal() as db:
        p = db.get(Portfolio, pid)
        if p is None or not p.copying:
            return "Not a copy portfolio."
        if not force and p.last_rebalance_at and p.next_rebalance_at and aware(p.next_rebalance_at) > now():
            return "Not due yet."  # another run (scheduler or setup) already did it while we waited for the lock
        params = {"market": p.market} | {k: v for k, v in (p.strategy_params or {}).items()}
        res = research.get(f"/internal/strategies/{p.strategy_key}/targets", timeout=600, params=params)
        if not res:
            p.status_message = "Waiting for the research service; will retry shortly."
            db.commit()
            return p.status_message
        if res.get("status") == "running":
            p.status_message = f"Preparing today's holdings: {res.get('message', 'working…')}"
            db.commit()
            return p.status_message
        if res.get("status") == "error":
            p.status_message = f"Strategy failed: {res.get('message')}. Retrying in an hour."
            p.next_rebalance_at = now() + timedelta(hours=1)
            db.commit()
            return p.status_message
        if p.rebalance == "F" and not force and p.positions and res.get("signature") == p.last_signature:
            p.next_rebalance_at = next_run("F", p.market)
            p.status_message = "No new filing yet; holding the latest one."
            db.commit()
            return p.status_message

        targets = {w["symbol"]: w for w in res["weights"]}
        q = quotes(list(targets) + [x.symbol for x in p.positions] + [p.benchmark_symbol])
        price = lambda s: (q.get(s) or {}).get("price") or (targets.get(s) or {}).get("price")
        equity = p.cash + sum(x.qty * (price(x.symbol) or x.avg_cost) for x in p.positions)
        want = {s: math.floor(w["weight"] * equity * CASH_BUFFER / price(s)) for s, w in targets.items() if price(s)}
        reason = (f"Copying {p.strategy_name}: new 13F filing" if p.rebalance == "F" else f"Copying {p.strategy_name}: scheduled rebalance")
        sells = buys = 0
        for x in list(p.positions):
            diff = x.qty - want.get(x.symbol, 0)
            if diff > 0 and price(x.symbol):
                execute(db, p, x.symbol, x.name, "sell", diff, price(x.symbol), reason, bool((q.get(x.symbol) or {}).get("trading", True)))
                sells += 1
        held = {x.symbol: x.qty for x in p.positions}
        for s, n in sorted(want.items(), key=lambda kv: -targets[kv[0]]["weight"]):
            diff = n - held.get(s, 0)
            if diff <= 0:
                continue
            cost = price(s) * (1 + FEES_BPS[p.market] / 1e4)
            diff = min(diff, math.floor(p.cash / cost))
            if diff >= 1:
                execute(db, p, s, targets[s]["name"], "buy", diff, price(s), reason, bool((q.get(s) or {}).get("trading", True)))
                buys += 1
        if not targets:
            note = "The strategy holds cash right now (its rules found nothing to buy)."
        else:
            note = f"{buys} buys, {sells} sells to match {len(targets)} holdings"
        p.last_rebalance_at = now()
        p.last_signature = res.get("signature")
        p.next_rebalance_at = next_run(p.rebalance or "M", p.market)
        p.status_message = f"Rebalanced {p.last_rebalance_at:%d %b %Y}: {note}. Next: {p.next_rebalance_at:%d %b %Y}."
        db.flush()
        snapshot(db, p)
        db.commit()
        return p.status_message


def rebalance_in_background(pid: int, force: bool = False) -> None:
    threading.Thread(target=lambda: _safe(rebalance, pid, force), daemon=True, name=f"rebalance:{pid}").start()


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        log.exception("%s%s failed", fn.__name__, args)


def due_rebalances() -> None:
    with SessionLocal() as db:
        due = db.scalars(select(Portfolio.id).where(Portfolio.copying.is_(True), Portfolio.next_rebalance_at <= now())).all()
    for pid in due:
        _safe(rebalance, pid)


def snapshot_all() -> None:
    with SessionLocal() as db:
        ps = db.scalars(select(Portfolio)).all()
        q = quotes([x.symbol for p in ps for x in p.positions] + [p.benchmark_symbol for p in ps])
        for p in ps:
            snapshot(db, p, value(p, q))
        db.commit()


def curve(db: Session, p: Portfolio, v: dict) -> list[dict]:
    snaps = db.scalars(select(Snapshot).where(Snapshot.portfolio_id == p.id).order_by(Snapshot.day)).all()
    pts = [{"day": str(s.day), "equity": s.equity,
            "benchmark": p.initial_cash * s.benchmark / p.benchmark_start if s.benchmark and p.benchmark_start else None} for s in snaps]
    today = str(date.today())
    live = {"day": today, "equity": v["equity"],
            "benchmark": p.initial_cash * (1 + v["benchmark_return"]) if v["benchmark_return"] is not None else None}
    if pts and pts[-1]["day"] == today:
        pts[-1] = live
    else:
        pts.append(live)
    start = {"day": str(aware(p.created_at).date()), "equity": p.initial_cash, "benchmark": p.initial_cash}
    return [start] + [x for x in pts if x["day"] > start["day"]] if pts[0]["day"] > start["day"] else pts
