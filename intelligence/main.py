"""intelligence service: live prices, news (own sentiment model + LLM), events, reputation and fused outlooks."""

import asyncio
import json
import logging
from pathlib import Path
import math
import signal
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from intelligence import events, ledger, live, news, outlook, registry, reputation, scheduler
from intelligence.config import get_settings
from intelligence.db import init_db
from intelligence.sentiment import model as sentiment
from outhit_common.internal import require_internal

logging.basicConfig(level=logging.INFO)
settings = get_settings()
Path("data").mkdir(exist_ok=True)
MAX_STREAM_SYMBOLS = 300
_shutting_down = threading.Event()


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class CleanJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, separators=(",", ":"), default=str).encode()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # End open live streams as soon as the server is asked to stop, so restarts never hang
    # waiting for browsers to disconnect.
    for sig in ((signal.SIGINT, signal.SIGTERM) if threading.current_thread() is threading.main_thread() else ()):
        previous = signal.getsignal(sig)

        def handler(signum, frame, previous=previous):
            _shutting_down.set()
            if callable(previous):
                previous(signum, frame)

        signal.signal(sig, handler)
    if settings.enable_scheduler:
        scheduler.start()
    yield
    _shutting_down.set()
    scheduler.stop()


app = FastAPI(
    title="Outhit intelligence",
    description="Outhit's own AI: minute-by-minute prices, news sentiment (own trained model), corporate-event detection, reputation scoring and fused bullish/bearish outlooks with a live track record. Educational use only.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=CleanJSONResponse,
)


def _symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()][:MAX_STREAM_SYMBOLS]


@app.get("/api/ai/health")
def health() -> dict:
    return {"status": "ok", "tracked_symbols": len(live.tracked()), "live_version": live.version}


@app.get("/api/ai/model")
def model_card() -> dict:
    return {
        "sentiment": sentiment.metrics(),
        "outlook_weights": {"quant": outlook.QUANT_WEIGHT, "technical_fallback": outlook.TECH_WEIGHT, **outlook.WEIGHTS},
        "events": [{"key": e.key, "label": e.label, "impact": e.impact, "controversy": e.controversy} for e in events.EVENTS],
    }


class SentimentIn(BaseModel):
    text: str = Field(min_length=3, max_length=1000)


@app.post("/api/ai/sentiment")
def analyse_text(body: SentimentIn) -> dict:
    s = sentiment.score([body.text])[0]
    return {**s, "explanation": sentiment.explain(body.text), "events": events.tag(body.text)}


@app.get("/api/ai/quotes")
async def quotes(symbols: str = Query(min_length=1)) -> dict:
    syms = _symbols(symbols)
    live.subscribe(syms)
    got = live.get(syms)
    missing = [s for s in syms if s not in got][:50]
    if missing:  # never-seen symbols: fetch now instead of waiting for the next minute's poll
        await run_in_threadpool(live.poll, missing)
        got = live.get(syms)
    return got


@app.get("/api/ai/stream")
async def stream(request: Request, symbols: str = Query(min_length=1)) -> StreamingResponse:
    """Server-Sent Events. Sends `quotes` (and `outlooks` where available) for the requested
    symbols whenever the minute poll produces new data. Symbols stay tracked while connected."""
    syms = _symbols(symbols)
    live.subscribe(syms)
    first = live.get(syms)
    if len(first) < len(syms):  # never-seen symbols: fetch now so the page fills immediately
        await run_in_threadpool(live.poll, [s for s in syms if s not in first])

    async def gen():
        seen = -1
        beat = 0
        while not await request.is_disconnected() and not _shutting_down.is_set():
            live.subscribe(syms)
            if live.version != seen:
                seen = live.version
                payload = {"quotes": live.get(syms), "outlooks": {s: {k: o[k] for k in ("score", "label", "confidence")} for s, o in outlook.latest(syms).items()}}
                yield f"event: update\ndata: {json.dumps(_clean(payload), default=str)}\n\n"
            beat += 1
            if beat % 15 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/ai/outlooks")
async def outlooks(market: str = "in") -> dict:
    companies = await run_in_threadpool(registry.by_market, market.lower())
    if not companies:
        await run_in_threadpool(registry.bootstrap)
        companies = await run_in_threadpool(registry.by_market, market.lower())
    syms = [c.symbol for c in companies]
    ready = outlook.latest(syms)
    rows = [ready[s] for s in syms if s in ready]
    rows.sort(key=lambda o: o["score"], reverse=True)
    return {
        "market": market,
        "tracked": len(syms),
        "ready": len(rows),
        "outlooks": [{k: v for k, v in o.items() if k != "components"} | {"components": {c["key"]: c["value"] for c in o["components"]}} for o in rows],
        "track_record": ledger.track_record(),
    }


@app.get("/api/ai/outlook/{symbol}")
async def outlook_detail(symbol: str) -> dict:
    symbol = symbol.upper()
    c = await run_in_threadpool(registry.ensure, symbol)
    if c is None:
        raise HTTPException(404, f"Couldn't find a listed company for '{symbol}'.")
    live.subscribe([symbol])
    if not live.get([symbol]):
        await run_in_threadpool(live.poll, [symbol])
    if c.news_checked_at is None:
        await run_in_threadpool(news.refresh, symbol)
    o = await run_in_threadpool(outlook.compute, symbol)
    rep = await run_in_threadpool(reputation.score, symbol)
    items = news.recent(symbol, days=30)
    return {
        "outlook": o,
        "quote": live.get([symbol]).get(symbol),
        "reputation": rep,
        "news": [
            {"title": i.title, "source": i.source, "url": i.url, "published_at": i.published_at.isoformat(), "sentiment": i.sentiment,
             "label": i.label, "confidence": i.confidence, "events": i.events}
            for i in items[:40]
        ],
        "history": ledger.history(symbol),
    }


@app.get("/api/ai/track-record")
def track_record() -> dict:
    return ledger.track_record()


# ---------------------------------------------------------------------------
# Internal endpoints (called by other services through the internal token)
# ---------------------------------------------------------------------------

from fastapi import Body, Depends  # noqa: E402


@app.get("/internal/outlooks", dependencies=[Depends(require_internal)])
def internal_outlooks(symbols: str | None = None) -> dict:
    """Latest outlook per symbol (all tracked symbols if none given)."""
    syms = _symbols(symbols) if symbols else None
    return outlook.latest(syms)


@app.post("/internal/track", dependencies=[Depends(require_internal)])
async def internal_track(symbols: list[str] = Body(...)) -> dict:
    """Start tracking news, reputation and live prices for these symbols (e.g. recommended stocks)."""
    n = await run_in_threadpool(registry.track, [s.upper() for s in symbols][:200])
    return {"added": n}


@app.get("/health")
def service_health() -> dict:
    return {"service": "intelligence", "status": "ok", "tracked_symbols": len(live.tracked())}
