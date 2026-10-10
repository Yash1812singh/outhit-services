"""advisor service: financial health check, risk profile, plan, goal projection, saved profiles and the AI chat."""

import logging
from pathlib import Path

from fastapi import FastAPI

from advisor.db import Base, engine
from advisor.routers import advisor
from outhit_common.web import CleanJSONResponse

logging.basicConfig(level=logging.INFO)
Path("data").mkdir(exist_ok=True)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Outhit advisor", version="2.0.0", default_response_class=CleanJSONResponse)
app.include_router(advisor.router)


@app.get("/health")
def health() -> dict:
    from advisor.config import get_settings

    s = get_settings()
    return {"service": "advisor", "status": "ok", "ai_provider": "anthropic" if s.anthropic_api_key else "groq" if s.groq_api_key else None}
