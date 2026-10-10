import numpy as np

from advisor import planner
from advisor.schemas import AdvisorInput


def profile(**overrides) -> AdvisorInput:
    base = dict(
        age=30, country="IN", monthly_income=100_000, monthly_expenses=50_000, monthly_debt_payment=0,
        current_savings=400_000, high_interest_debt=0, has_health_insurance=True, has_life_insurance=False,
        dependents=0, investment_horizon_years=15, experience="intermediate", drawdown_reaction="hold",
        income_stability="stable", include_crypto=False, lump_sum=300_000, monthly_investment=20_000,
    )
    base.update(overrides)
    return AdvisorInput(**base)


def test_healthy_profile_is_ready_to_invest():
    h = planner.financial_health(profile())
    assert h["score"] >= 85
    assert h["actions"][-1]["priority"] == "good"


def test_debt_and_missing_emergency_fund_are_flagged_first():
    h = planner.financial_health(profile(high_interest_debt=80_000, current_savings=10_000, has_health_insurance=False))
    priorities = [a["priority"] for a in h["actions"]]
    assert priorities[0] == "critical"
    titles = " ".join(a["title"] for a in h["actions"])
    assert "high-interest debt" in titles and "emergency fund" in titles and "health insurance" in titles
    assert all(a["priority"] != "good" for a in h["actions"])


def test_short_horizon_caps_risk():
    r = planner.risk_profile(profile(investment_horizon_years=2, drawdown_reaction="buy_more", experience="advanced", age=22))
    assert r["level"] <= 1
    assert r["notes"]


def test_panic_seller_is_never_aggressive():
    r = planner.risk_profile(profile(drawdown_reaction="sell_all", experience="none", age=22, investment_horizon_years=30, income_stability="very_stable"))
    assert r["level"] <= 2


def test_bucket_weights_sum_to_one_and_respect_crypto_choice():
    for level in range(5):
        for country in ("IN", "US"):
            for crypto in (True, False):
                w = planner.bucket_weights(level, country, crypto)
                assert np.isclose(sum(w.values()), 1.0)
                if not crypto:
                    assert w["crypto"] == 0


def test_plan_reserves_foundations_and_allocates_the_rest(fake_market):
    p = profile(high_interest_debt=50_000, current_savings=100_000, lump_sum=500_000, include_crypto=True, drawdown_reaction="buy_more")
    plan = planner.build_plan(p)
    gap = planner.financial_health(p)["metrics"]["emergency_gap"]
    assert plan["reserved"]["debt_payoff"] == 50_000
    assert plan["reserved"]["emergency_fund"] == gap
    assert np.isclose(plan["investable_lump_sum"], 500_000 - 50_000 - gap)
    assert np.isclose(sum(h["weight"] for h in plan["holdings"]), 1.0)
    assert all(h["weight"] <= 0.05 + 1e-9 for h in plan["holdings"] if h["role"] == "satellite")


def test_projection_target_solver_hits_75_percent():
    out = planner.project_goal(initial=100_000, monthly=10_000, years=10, annual_return=0.1, annual_vol=0.15, target=3_000_000)
    assert out["pessimistic_final"] < out["median_final"] < out["optimistic_final"]
    needed = out["monthly_needed_for_75pct"]
    check = planner.project_goal(initial=100_000, monthly=needed * 1.001, years=10, annual_return=0.1, annual_vol=0.15, target=3_000_000)
    assert check["prob_reach_target"] >= 0.75


def test_zero_volatility_projection_matches_compound_interest():
    out = planner.project_goal(initial=1000, monthly=0, years=10, annual_return=0.08, annual_vol=0.0)
    assert np.isclose(out["median_final"], 1000 * 1.08**10, rtol=1e-6)


def test_small_lump_sum_pays_debt_before_emergency_fund(fake_market):
    p = profile(high_interest_debt=40_000, current_savings=150_000, lump_sum=100_000, monthly_expenses=45_000)
    plan = planner.build_plan(p)
    assert plan["reserved"]["debt_payoff"] == 40_000
    assert plan["reserved"]["emergency_fund"] == 60_000
    assert plan["investable_lump_sum"] == 0
    assert any("Before starting monthly investments" in s for s in plan["steps"])


def test_plan_fetches_signals_from_market_data_and_picks_bullish_satellites(fake_market):
    plan = planner.build_plan(profile(drawdown_reaction="buy_more", experience="advanced", investment_horizon_years=25))
    (path, params), = fake_market.calls
    assert path == "/api/market/signals"
    requested = set(params["symbols"].split(","))
    assert {h["symbol"] for h in plan["holdings"]} <= requested
    satellites = [h for h in plan["holdings"] if h["role"] == "satellite"]
    assert satellites, "a growth profile with bullish stocks should get satellite picks"
    assert all(h["score"] >= planner.BULLISH_AT and h["signal"] == "Bullish" for h in satellites)
    assert len(satellites) <= 4


def test_plan_survives_market_data_outage(monkeypatch):
    from advisor import clients
    from advisor.tests.conftest import FakeServiceClient

    monkeypatch.setattr(clients, "market", FakeServiceClient())  # every call returns None
    plan = planner.build_plan(profile())
    assert plan["holdings"] and all(h["role"] == "core" for h in plan["holdings"])
    assert all(h["signal"] == "Data unavailable" and h["score"] is None for h in plan["holdings"])
    assert np.isclose(sum(h["weight"] for h in plan["holdings"]), 1.0)
