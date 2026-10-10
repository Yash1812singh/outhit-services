from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from market_data.services import market_data
from market_data.services import signals as sig
from market_data.universe import ASSET_CLASS_LABELS, BENCHMARKS, UNIVERSE, lookup

router = APIRouter(prefix="/api/market", tags=["market"])


def _overview() -> dict:
    assets = UNIVERSE + [b for b in BENCHMARKS if b.symbol not in {a.symbol for a in UNIVERSE}]
    history = market_data.get_history([a.symbol for a in assets])
    if not history:
        raise HTTPException(503, "Market data is temporarily unavailable. Please try again shortly.")

    groups = []
    for key, label in ASSET_CLASS_LABELS.items():
        rows = [sig.summary(sig.analyse(a, history[a.symbol]), include_backtest=True) for a in UNIVERSE if a.asset_class == key and a.symbol in history]
        rows.sort(key=lambda r: r["score"] if r["score"] is not None else -999, reverse=True)
        groups.append({"key": key, "label": label, "assets": rows})

    benchmarks = [sig.summary(sig.analyse(b, history[b.symbol])) for b in BENCHMARKS if b.symbol in history]
    ranked = [r for g in groups for r in g["assets"] if r["score"] is not None]
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {
        "benchmarks": benchmarks,
        "groups": groups,
        "top_bullish": ranked[:5],
        "top_bearish": sorted(ranked, key=lambda r: r["score"])[:5],
    }


@router.get("/overview")
async def overview() -> dict:
    return await run_in_threadpool(_overview)


@router.get("/asset/{symbol}")
async def asset(symbol: str) -> dict:
    def work() -> dict:
        try:
            frame = market_data.get_one(symbol)
        except market_data.MarketDataError as exc:
            raise HTTPException(404, str(exc)) from exc
        return sig.detail(sig.analyse(lookup(symbol), frame))

    return await run_in_threadpool(work)


@router.get("/search")
async def search(q: str = Query(min_length=1, max_length=64)) -> list[dict]:
    return await run_in_threadpool(market_data.search, q)


@router.get("/signals")
async def signals(symbols: str = Query(min_length=1, max_length=4000)) -> dict:
    """Score and label for several symbols at once (used by the advisor service)."""

    def work() -> dict:
        syms = [x.strip().upper() for x in symbols.split(",") if x.strip()][:100]
        hist = market_data.get_history(syms)
        out = {}
        for s in syms:
            if s in hist:
                a = sig.analyse(lookup(s), hist[s])
                out[s] = {"score": round(a.score, 1) if a.ready else None, "signal": sig.label_for(a.score) if a.ready else "Insufficient data",
                          "price": float(a.close.iloc[-1]), "volatility": sig.ind.annual_volatility(a.close)}
        return out

    return await run_in_threadpool(work)
