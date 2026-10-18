"""market-data service: stocks (screener, sectors, picks, predictions), crypto, global markets and asset analysis."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from market_data.config import get_settings
from market_data.db import Base, engine
from market_data import models  # noqa: F401  (register tables)
from market_data.routers import crypto, market, stocks
from market_data.services import predictor, screener
from outhit_common.web import CleanJSONResponse

logging.basicConfig(level=logging.INFO)
settings = get_settings()
Path("data").mkdir(exist_ok=True)
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.enable_refresher:
        screener.start_refresher()
        predictor.start_learning_loop()
    yield
    screener.stop_refresher()
    predictor.stop_learning_loop()


app = FastAPI(title="Outhit market-data", version="2.0.0", lifespan=lifespan, default_response_class=CleanJSONResponse)
app.include_router(market.router)
app.include_router(stocks.router)
app.include_router(crypto.router)


@app.get("/health")
def health() -> dict:
    return {"service": "market-data", "status": "ok"}
