"""paper service: virtual portfolios to test strategies with live prices, and one-click copy-trading."""

import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from outhit_common.internal import require_user
from outhit_common.web import CleanJSONResponse
from paper import scheduler, trading
from paper.config import get_settings
from paper.db import Portfolio, Trade, aware, get_db, init_db, now

logging.basicConfig(level=logging.INFO)
settings = get_settings()
Path("data").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.enable_scheduler:
        scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="Outhit paper trading", version="1.0.0", lifespan=lifespan, default_response_class=CleanJSONResponse)


@app.get("/health")
def health() -> dict:
    return {"service": "paper", "status": "ok"}


class NewPortfolio(BaseModel):
    market: Literal["in", "us"] = "in"
    cash: float = Field(gt=0, le=1e10)
    name: str | None = Field(default=None, max_length=100)
    strategy: str | None = Field(default=None, max_length=40, description="Copy this strategy (key from /api/ai/strategies)")
    params: dict[str, str] = Field(default_factory=dict, description="Custom-strategy settings")


class Order(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    side: Literal["buy", "sell"]
    qty: int | None = Field(default=None, gt=0, le=10_000_000)
    amount: float | None = Field(default=None, gt=0, le=1e10)


def _owned(db: Session, pid: int, user: int) -> Portfolio:
    p = db.get(Portfolio, pid)
    if p is None or p.user_id != user:
        raise HTTPException(404, "Portfolio not found")
    return p


def _summary(p: Portfolio, v: dict) -> dict:
    return {
        "id": p.id, "name": p.name, "market": p.market, "currency": p.currency, "created_at": aware(p.created_at),
        "copying": p.copying, "strategy": {"key": p.strategy_key, "name": p.strategy_name, "params": p.strategy_params, "rebalance": p.rebalance} if p.strategy_key else None,
        "initial_cash": p.initial_cash, "positions_count": len(p.positions), "status_message": p.status_message,
        "next_rebalance_at": aware(p.next_rebalance_at), "last_rebalance_at": aware(p.last_rebalance_at),
        "benchmark_symbol": p.benchmark_symbol, "fees_bps": trading.FEES_BPS[p.market],
    } | {k: v[k] for k in ("equity", "cash", "invested", "pnl", "return", "benchmark_return", "vs_benchmark")}


@app.get("/api/paper/portfolios")
def list_portfolios(user: int = Depends(require_user), db: Session = Depends(get_db)) -> list[dict]:
    ps = db.scalars(select(Portfolio).where(Portfolio.user_id == user).order_by(Portfolio.created_at.desc())).all()
    q = trading.quotes([x.symbol for p in ps for x in p.positions] + [p.benchmark_symbol for p in ps])
    return [_summary(p, trading.value(p, q)) for p in ps]


@app.post("/api/paper/portfolios", status_code=201)
def create_portfolio(body: NewPortfolio, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    count = db.scalar(select(func.count()).select_from(Portfolio).where(Portfolio.user_id == user))
    if count >= settings.max_portfolios_per_user:
        raise HTTPException(422, f"You can have up to {settings.max_portfolios_per_user} paper portfolios. Delete one first.")
    st = None
    if body.strategy:
        st = trading.strategies().get(body.strategy)
        if st is None:
            raise HTTPException(422 if trading.strategies() else 503, "Unknown strategy" if trading.strategies() else "Strategy list unavailable; try again shortly.")
        if body.market not in st["markets"]:
            raise HTTPException(422, f"{st['name']} is available for: {', '.join(m.upper() for m in st['markets'])}")
    bench = trading.BENCHMARK[body.market]
    bq = trading.quotes([bench]).get(bench)
    name = body.name or (f"Copy: {st['person']} – {st['name']}" if st and st["key"] != "custom" else "My strategy" if st else f"My {body.market.upper()} portfolio")
    p = Portfolio(user_id=user, name=name[:100], market=body.market, currency=trading.CURRENCY[body.market], initial_cash=body.cash,
                  cash=body.cash, benchmark_symbol=bench, benchmark_start=bq["price"] if bq else None, copying=st is not None)
    if st:
        p.strategy_key, p.strategy_name, p.rebalance = st["key"], f"{st['person']} – {st['name']}", st["rebalance"]
        p.strategy_params = body.params if st["key"] == "custom" else {}
        if st["key"] == "custom":
            p.strategy_name = "Custom: " + " + ".join(f.split(":")[0].replace("_", " ") for f in body.params.get("factors", "momentum,quality").split(","))
            p.rebalance = body.params.get("rebalance", "M").upper() if body.params.get("rebalance", "M").upper() in ("W", "M", "Q") else "M"
        p.next_rebalance_at = now()
        p.status_message = "Setting up: working out the strategy's holdings for today…"
    db.add(p)
    db.commit()
    if st:
        trading.rebalance_in_background(p.id)
    return _summary(p, trading.value(p, {bench: bq} if bq else {}))


@app.get("/api/paper/portfolios/{pid}")
def portfolio_detail(pid: int, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    p = _owned(db, pid, user)
    v = trading.value(p)
    trades = db.scalars(select(Trade).where(Trade.portfolio_id == p.id).order_by(Trade.at.desc(), Trade.id.desc()).limit(300)).all()
    closed = [t for t in trades if t.realized_pnl is not None]
    return _summary(p, v) | {
        "positions": v["positions"],
        "curve": trading.curve(db, p, v),
        "trades": [{"at": aware(t.at), "symbol": t.symbol, "name": t.name, "side": t.side, "qty": t.qty, "price": t.price, "fees": t.fees,
                    "realized_pnl": t.realized_pnl, "reason": t.reason, "market_open": t.market_open} for t in trades],
        "stats": {"trades": len(trades), "fees_paid": sum(t.fees for t in trades), "realized_pnl": sum(t.realized_pnl or 0 for t in closed),
                  "winning_sells": sum(1 for t in closed if t.realized_pnl > 0), "closed_sells": len(closed)},
    }


@app.post("/api/paper/portfolios/{pid}/orders", status_code=201)
async def place_order(pid: int, body: Order, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    p = _owned(db, pid, user)
    try:
        t = await run_in_threadpool(trading.manual_order, db, p, body.symbol, body.side, body.qty, body.amount)
    except trading.TradeError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"symbol": t.symbol, "side": t.side, "qty": t.qty, "price": t.price, "fees": t.fees, "realized_pnl": t.realized_pnl,
            "market_open": t.market_open, "cash": p.cash}


@app.post("/api/paper/portfolios/{pid}/rebalance")
def rebalance_now(pid: int, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    p = _owned(db, pid, user)
    if not p.copying:
        raise HTTPException(422, "Only copy portfolios rebalance automatically.")
    if p.last_rebalance_at and now() - aware(p.last_rebalance_at) < timedelta(minutes=10):
        raise HTTPException(429, "Rebalanced a few minutes ago; please wait.")
    p.status_message = "Rebalancing now…"
    db.commit()
    trading.rebalance_in_background(p.id, force=True)
    return {"status_message": p.status_message}


@app.post("/api/paper/portfolios/{pid}/stop-copying")
def stop_copying(pid: int, user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    p = _owned(db, pid, user)
    p.copying, p.next_rebalance_at = False, None
    p.status_message = "Stopped copying; the positions are now yours to trade."
    db.commit()
    return {"status_message": p.status_message}


@app.delete("/api/paper/portfolios/{pid}", status_code=204)
def delete_portfolio(pid: int, user: int = Depends(require_user), db: Session = Depends(get_db)) -> None:
    db.delete(_owned(db, pid, user))
    db.commit()


@app.get("/api/paper/quote")
def quote(symbol: str = Query(min_length=1, max_length=30), market: Literal["in", "us"] = "in", user: int = Depends(require_user)) -> dict:
    s = symbol.strip().upper()
    if not trading.valid_symbol(market, s):
        raise HTTPException(422, "Use an NSE/BSE symbol like TCS.NS" if market == "in" else "Use a US ticker like AAPL")
    q = trading.quotes([s]).get(s)
    if not q:
        raise HTTPException(404, f"No live price for {s}")
    return {"symbol": s, "price": q["price"], "change_pct": q.get("change_pct"), "trading": q.get("trading"),
            "delayed_minutes": q.get("delayed_minutes"), "fees_bps": trading.FEES_BPS[market]}
