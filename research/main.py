"""research service: method tournament, recommendations (long/short/intraday/F&O), rockets & growers,
portfolio builder, mutual funds, ETF/crypto/commodity allocation, and the pick ledger."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from outhit_common.internal import require_internal
from outhit_common.web import CleanJSONResponse
from research import allocation, backtest, funds, gurus, portfolio, recommend, regime, rockets, scheduler, strategies, tournament
from research.config import get_settings
from research.db import init_db

logging.basicConfig(level=logging.INFO)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.enable_scheduler:
        scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="Outhit research", version="2.0.0", lifespan=lifespan, default_response_class=CleanJSONResponse)


@app.get("/health")
def health() -> dict:
    return {"service": "research", "status": "ok"}


# ---------------------------------------------------------------------------
# Research & recommendations
# ---------------------------------------------------------------------------



HORIZONS = {"long_term", "short_term", "intraday", "fo"}


def _slim_tournament(t: dict) -> dict:
    if t.get("status") != "ready":
        return t
    out = {k: v for k, v in t.items() if k not in ("horizons", "companies", "factors_now")}
    out["horizons"] = {h: {k: v for k, v in x.items() if k != "ranking"} | {"top": x.get("ranking", [])[:10], "bottom": x.get("ranking", [])[-5:]}
                       for h, x in t["horizons"].items()}
    return out


@app.get("/api/ai/research/{market}")
async def research(market: str) -> dict:
    """Method tournament results (runs in the background on first call; poll until status=ready)."""
    t = await run_in_threadpool(tournament.get, market.lower())
    out = _slim_tournament(t)
    if t.get("status") == "ready":
        names = t["companies"]
        for x in out["horizons"].values():
            for r in x.get("top", []) + x.get("bottom", []):
                r["name"] = names.get(r["symbol"], {}).get("name")
    return out


@app.get("/api/ai/picks/{market}")
async def picks(market: str, horizon: str = "long_term") -> dict:
    if horizon not in HORIZONS:
        raise HTTPException(422, f"horizon must be one of {sorted(HORIZONS)}")
    if horizon == "fo" and market.lower() not in ("in", "us"):
        raise HTTPException(422, "F&O analysis is available for India (NSE) and the US.")
    return await run_in_threadpool(recommend.get, market.lower(), horizon)


@app.get("/api/ai/funds")
def fund_categories() -> list[dict]:
    return [{"key": k, "label": v[0]} for k, v in funds.CATEGORIES.items()]


@app.get("/api/ai/funds/{category}")
async def fund_ranking(category: str) -> dict:
    if category not in funds.CATEGORIES:
        raise HTTPException(404, "Unknown fund category")
    return await run_in_threadpool(funds.get, category)


@app.get("/api/ai/allocation")
async def asset_allocation() -> dict:
    return await run_in_threadpool(allocation.get)


@app.get("/api/ai/picks-track-record")
def picks_track_record() -> dict:
    return recommend.track_record()


@app.get("/api/ai/regime/{market}")
async def market_regime(market: str) -> dict:
    return await run_in_threadpool(regime.detect, market.lower())


@app.get("/api/ai/portfolio/{market}")
async def build_portfolio(market: str, amount: float = Query(100000, gt=0, le=1e12), risk: str = "balanced") -> dict:
    if risk not in portfolio.SPLITS:
        raise HTTPException(422, f"risk must be one of {list(portfolio.SPLITS)}")
    try:
        return await run_in_threadpool(portfolio.build, market.lower(), amount, risk)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/ai/rockets/{market}")
async def rocket_picks(market: str, kind: str = "rocket") -> dict:
    """Stocks most likely to be the market's biggest winners (kind=rocket, 6 months) or steady growers (kind=grower, 12 months)."""
    if kind not in rockets.KINDS:
        raise HTTPException(422, f"kind must be one of {list(rockets.KINDS)}")
    return await run_in_threadpool(rockets.get, market.lower(), kind)


# ---------------------------------------------------------------------------
# Strategies, backtests and investor portfolios
# ---------------------------------------------------------------------------

CUSTOM_PARAMS = ("factors", "top_n", "rebalance", "weighting", "trend_filter", "regime_filter")


def _strategy_args(key: str, market: str, request: Request) -> dict:
    market = market.lower()
    if market not in strategies.CURRENCY:
        raise HTTPException(422, "market must be in or us")
    params = {k: v for k, v in request.query_params.items() if k in CUSTOM_PARAMS}
    try:
        st = strategies.get(key, params)
    except KeyError:
        raise HTTPException(404, "Unknown strategy") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if market not in st.markets:
        raise HTTPException(422, f"{st.name} is available for: {', '.join(m.upper() for m in st.markets)}")
    return params


@app.get("/api/ai/strategies")
def strategy_catalogue() -> dict:
    """Famous investors' strategies, classic portfolios and real 13F portfolios, plus custom-builder factors."""
    return {
        "strategies": strategies.catalogue(),
        "factors": [{"key": k, "label": v[0]} for k, v in strategies.FACTORS.items()],
        "default_cost_bps": strategies.DEFAULT_COST_BPS,
    }


@app.get("/api/ai/backtest/{key}")
async def run_backtest(key: str, request: Request, market: str = "in", years: int = Query(5, ge=1, le=10),
                       cost_bps: float | None = Query(None, ge=0, le=200)) -> dict:
    """Backtest a strategy (runs in the background; poll until status=ready). Custom strategies take
    factors=momentum:1,value:1&top_n=20&rebalance=M&weighting=equal&trend_filter=1&regime_filter=1."""
    params = _strategy_args(key, market, request)
    return await run_in_threadpool(backtest.get, key, market.lower(), years, cost_bps, params)


@app.get("/api/ai/gurus/{key}/filings")
async def guru_filings(key: str) -> list[dict]:
    """Recent 13F filings of one investor: holdings and what changed."""
    if key not in gurus.GURUS:
        raise HTTPException(404, "Unknown investor")
    fl = await run_in_threadpool(gurus.filings, key, 2)
    return [{k: f[k] for k in ("filed", "period", "mapped_weight")} | {"holdings": f["holdings"][:40], "changes": gurus.changes(f, fl[i + 1] if i + 1 < len(fl) else None)}
            for i, f in enumerate(fl)]


@app.get("/internal/strategies/{key}/targets", dependencies=[Depends(require_internal)])
async def strategy_targets(key: str, request: Request, market: str = "in") -> dict:
    """Today's target weights for copy-trading (background job; poll until status=ready)."""
    params = _strategy_args(key, market, request)
    return await run_in_threadpool(backtest.targets, key, market.lower(), params)
