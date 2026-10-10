from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from outhit_common.internal import require_user
from market_data.markets import MACRO, MARKETS, MARKETS_BY_CODE, SECTORS
from market_data.services import intraday, market_data, ranking, screener
from market_data.services import signals as sig
from market_data.universe import Asset

router = APIRouter(prefix="/api", tags=["global markets & stocks"])


def native(value):
    """Convert numpy scalars (which FastAPI can't encode) to plain Python values."""
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


async def _run(fn, *args):
    try:
        return native(await run_in_threadpool(fn, *args))
    except ranking.NoData as exc:
        raise HTTPException(404, str(exc)) from exc


def _market(code: str) -> str:
    code = code.lower()
    if code not in MARKETS_BY_CODE:
        raise HTTPException(404, f"Unknown market '{code}'. See /api/stocks/markets.")
    return code


def _sector(sector: str | None) -> str | None:
    if sector and sector not in SECTORS and sector != "Unclassified":
        raise HTTPException(422, f"Unknown sector. Use one of: {', '.join(SECTORS)}")
    return sector or None


@router.get("/global/overview")
async def global_overview() -> dict:
    def work():
        history = market_data.get_history([m.symbol for m in MACRO])
        groups: dict[str, list] = {}
        for m in MACRO:
            if m.symbol not in history:
                continue
            a = sig.analyse(Asset(m.symbol, m.name, "index_fund", "GLOBAL", ""), history[m.symbol])
            row = sig.summary(a)
            row.pop("currency")
            row["unit"] = m.unit
            row.update(m.extra)
            groups.setdefault(m.group, []).append(row)
        if not groups:
            raise HTTPException(503, "Market data is temporarily unavailable.")
        return {"groups": [{"name": k, "assets": v} for k, v in groups.items()]}

    return await run_in_threadpool(work)


@router.get("/stocks/markets")
def markets() -> list[dict]:
    out = []
    for m in MARKETS:
        st = screener.status(m.code)
        out.append({
            "code": m.code,
            "name": m.name,
            "currency": m.currency,
            "index": m.index,
            "companies": st.companies if st else None,
            "refreshed_at": st.refreshed_at.isoformat() if st else None,
        })
    return out


@router.get("/stocks/sectors-list")
def sectors_list() -> list[str]:
    return SECTORS


@router.get("/stocks/{market}/companies")
async def companies(
    market: str,
    sector: str | None = None,
    q: str | None = Query(default=None, max_length=64),
    sort: str = "market_cap",
    desc: bool = True,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> dict:
    code, sec = _market(market), _sector(sector)
    return await _run(ranking.browse, code, sec, q, sort, desc, page, page_size)


@router.get("/stocks/{market}/sectors")
async def sectors(market: str) -> dict:
    return await _run(ranking.sectors, _market(market))


@router.get("/stocks/{market}/picks")
async def picks(
    market: str,
    horizon: Literal["long_term", "swing", "intraday"] = "long_term",
    sector: str | None = None,
    size: Literal["large", "mid", "small", "large_mid", "all"] = "large_mid",
) -> dict:
    code, sec = _market(market), _sector(sector)
    if horizon == "long_term":
        return await _run(ranking.long_term_picks, code, sec, size)
    if horizon == "swing":
        return await _run(ranking.swing_picks, code)
    return await _run(intraday.scan, code)


@router.post("/stocks/{market}/refresh")
async def refresh(market: str, _: int = Depends(require_user)) -> dict:
    code = _market(market)
    n = await run_in_threadpool(screener.refresh_market, code)
    ranking.clear_cache()
    return {"market": code, "companies": n}


@router.get("/predict/{market}")
async def predict(market: str, horizon: int = 21) -> dict:
    """ML forecast of which stocks will beat the market over the next 1 or 3 months.
    The first call starts training (~30 s) and returns status "training"; poll until "ready"."""
    from market_data.services import predictor

    if horizon not in predictor.HORIZONS:
        raise HTTPException(422, "horizon must be 21 (1 month) or 63 (3 months)")
    return native(await run_in_threadpool(predictor.get, _market(market), horizon))


@router.get("/stocks/quality/{symbol}")
async def quality(symbol: str) -> dict:
    """Per-company quality check (ROE, debt, margins, growth, cash flow) with strengths and red flags."""

    def work():
        q = screener.get_quote(symbol)
        f = ranking.fundamentals(symbol)
        row = q.data if q else {}
        score, strengths, flags = ranking.quality(f, q.sector if q else "", row)
        return {
            "symbol": symbol,
            "quality": None if score is None else round(score * 100),
            "strengths": strengths,
            "red_flags": flags,
            "fundamentals": {k: f.get(k) for k in ("returnOnEquity", "debtToEquity", "profitMargins", "revenueGrowth", "earningsGrowth", "freeCashflow", "industry")},
        }

    return native(await run_in_threadpool(work))
