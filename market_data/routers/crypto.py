"""Crypto: its own module inside market-data, with a wider coin list than the curated universe."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from market_data.services import indicators as ind
from market_data.services import market_data
from market_data.services import signals as sig
from market_data.universe import CRYPTO, Asset

router = APIRouter(prefix="/api/crypto", tags=["crypto"])

COINS = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche", "LINK-USD": "Chainlink", "DOT-USD": "Polkadot", "TRX-USD": "Tron",
    "LTC-USD": "Litecoin", "BCH-USD": "Bitcoin Cash", "XLM-USD": "Stellar", "ATOM-USD": "Cosmos", "NEAR-USD": "NEAR",
    "UNI7083-USD": "Uniswap", "AAVE-USD": "Aave", "HBAR-USD": "Hedera", "ETC-USD": "Ethereum Classic", "FIL-USD": "Filecoin",
    "ALGO-USD": "Algorand", "ICP-USD": "Internet Computer", "XMR-USD": "Monero",
}


def _overview() -> dict:
    hist = market_data.get_history(list(COINS))
    if not hist:
        raise HTTPException(503, "Crypto data is temporarily unavailable.")
    rows = []
    for sym, name in COINS.items():
        if sym not in hist:
            continue
        a = sig.analyse(Asset(sym, name, CRYPTO, "GLOBAL", "USD"), hist[sym])
        s = sig.summary(a, include_backtest=True)
        c = a.close
        s["max_drawdown_1y"] = ind.max_drawdown(c.iloc[-365:])
        s["above_200d"] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
        rows.append(s)
    rows.sort(key=lambda r: r["score"] if r["score"] is not None else -999, reverse=True)
    btc = next((r for r in rows if r["symbol"] == "BTC-USD"), None)
    return {
        "coins": rows,
        "in_uptrend": sum(r["above_200d"] for r in rows),
        "bitcoin": btc,
        "note": "Crypto trades 24/7 and is far more volatile than stocks; most coins fall 70-90% in bear markets. Keep it a small part of a portfolio.",
    }


@router.get("/overview")
async def overview() -> dict:
    return await run_in_threadpool(_overview)
