from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from advisor import ai_advisor, planner
from advisor.db import FinancialProfile, get_db
from advisor.schemas import AdvisorInput, ChatRequest, ChatResponse, ProjectionRequest
from outhit_common.internal import optional_user, require_user

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


def _profile(db: Session, user_id: int) -> FinancialProfile | None:
    return db.query(FinancialProfile).filter(FinancialProfile.user_id == user_id).one_or_none()


def _save_profile(db: Session, user_id: int, body: AdvisorInput) -> None:
    row = _profile(db, user_id)
    if row:
        row.data = body.model_dump()
    else:
        db.add(FinancialProfile(user_id=user_id, data=body.model_dump()))
    db.commit()


@router.post("/health")
def health(body: AdvisorInput) -> dict:
    return planner.financial_health(body)


@router.post("/risk-profile")
def risk_profile(body: AdvisorInput) -> dict:
    return planner.risk_profile(body)


@router.post("/plan")
async def plan(body: AdvisorInput, user: int | None = Depends(optional_user), db: Session = Depends(get_db)) -> dict:
    """Full plan. Signed-in users have their answers saved for next time and for the AI chat."""
    if user:
        _save_profile(db, user, body)
    return await run_in_threadpool(planner.build_plan, body)


@router.post("/project")
def project(body: ProjectionRequest) -> dict:
    return planner.project_goal(
        initial=body.initial,
        monthly=body.monthly,
        years=body.years,
        annual_return=body.annual_return,
        annual_vol=body.annual_volatility,
        target=body.target,
        step_up=body.annual_step_up,
        inflation=body.inflation,
    )


@router.get("/profile")
def get_profile(user: int = Depends(require_user), db: Session = Depends(get_db)) -> dict | None:
    row = _profile(db, user)
    return row.data if row else None


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, user: int = Depends(require_user), db: Session = Depends(get_db)) -> ChatResponse:
    if body.messages[-1].role != "user":
        raise HTTPException(422, "The last message must be from the user")
    row = _profile(db, user)
    profile = row.data if row else None
    try:
        reply = await run_in_threadpool(ai_advisor.chat, [m.model_dump() for m in body.messages], profile)
    except ai_advisor.AdvisorUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return ChatResponse(reply=reply)
