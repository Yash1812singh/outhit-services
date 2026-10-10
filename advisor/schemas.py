from typing import Literal

from pydantic import BaseModel, Field


class AdvisorInput(BaseModel):
    """Everything the planner needs. Amounts are in the investor's home currency."""

    age: int = Field(ge=16, le=100)
    country: Literal["IN", "US"] = "IN"

    monthly_income: float = Field(ge=0)
    monthly_expenses: float = Field(ge=0, description="Living costs, excluding loan repayments")
    monthly_debt_payment: float = Field(default=0, ge=0, description="Total EMIs / loan repayments per month")
    current_savings: float = Field(default=0, ge=0, description="Cash and liquid savings")
    high_interest_debt: float = Field(default=0, ge=0, description="Credit-card or personal-loan balance")
    has_health_insurance: bool = False
    has_life_insurance: bool = False
    dependents: int = Field(default=0, ge=0, le=20)

    investment_horizon_years: int = Field(ge=1, le=50)
    experience: Literal["none", "beginner", "intermediate", "advanced"] = "beginner"
    drawdown_reaction: Literal["sell_all", "sell_some", "hold", "buy_more"] = "hold"
    income_stability: Literal["unstable", "stable", "very_stable"] = "stable"
    include_crypto: bool = False

    lump_sum: float = Field(default=0, ge=0, description="Amount available to invest now")
    monthly_investment: float = Field(default=0, ge=0)
    annual_step_up: float = Field(default=0.0, ge=0, le=0.5, description="Yearly increase of the monthly investment")
    goal_amount: float | None = Field(default=None, ge=0)


class ProjectionRequest(BaseModel):
    initial: float = Field(ge=0)
    monthly: float = Field(ge=0)
    years: int = Field(ge=1, le=50)
    annual_return: float = Field(default=0.10, ge=-0.5, le=0.5)
    annual_volatility: float = Field(default=0.15, ge=0, le=1.5)
    target: float | None = Field(default=None, ge=0)
    annual_step_up: float = Field(default=0.0, ge=0, le=0.5)
    inflation: float = Field(default=0.05, ge=0, le=0.3)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class ChatResponse(BaseModel):
    reply: str
