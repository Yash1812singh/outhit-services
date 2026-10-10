"""Personal financial planning: health check, risk profile, allocation and goal projection.

The ordering of advice follows standard financial-planning practice: build an
emergency fund and clear expensive debt before investing, then size risk to
both the person's capacity (horizon, age, income stability) and tolerance
(experience, how they react to losses).
"""

import numpy as np

from advisor import clients
from advisor.schemas import AdvisorInput
from advisor.universe import EQUITY_IN, EQUITY_US, UNIVERSE, lookup

BULLISH_AT, BEARISH_AT = 15, -15  # same thresholds as the market-data signal engine

# ---------------------------------------------------------------------------
# Financial health
# ---------------------------------------------------------------------------

EMERGENCY_TARGET_MONTHS = 6


def financial_health(p: AdvisorInput) -> dict:
    income = max(p.monthly_income, 1e-9)
    outgoings = p.monthly_expenses + p.monthly_debt_payment
    surplus = p.monthly_income - outgoings
    savings_rate = surplus / income
    emergency_months = p.current_savings / outgoings if outgoings > 0 else float("inf")
    dti = p.monthly_debt_payment / income
    emergency_target = EMERGENCY_TARGET_MONTHS * outgoings
    emergency_gap = max(0.0, emergency_target - p.current_savings)

    score = 0
    score += 25 if savings_rate >= 0.20 else 15 if savings_rate >= 0.10 else 5 if savings_rate > 0 else 0
    score += 25 if emergency_months >= 6 else 15 if emergency_months >= 3 else 5 if emergency_months >= 1 else 0
    score += 20 if dti <= 0.20 else 10 if dti <= 0.36 else 3 if dti <= 0.50 else 0
    score += 15 if p.high_interest_debt <= 0 else 0
    score += 10 if p.has_health_insurance else 0
    score += 5 if (p.has_life_insurance or p.dependents == 0) else 0

    actions: list[dict] = []
    if surplus <= 0:
        actions.append({
            "priority": "critical",
            "title": "Spend less than you earn",
            "detail": f"Your outgoings exceed your income by {-surplus:,.0f} a month. List every expense and cut non-essentials until you have a surplus. Investing is not possible until this is fixed.",
        })
    if p.high_interest_debt > 0:
        actions.append({
            "priority": "critical",
            "title": "Pay off high-interest debt first",
            "detail": f"Clear {p.high_interest_debt:,.0f} of credit-card or personal-loan debt before investing. Paying off debt at 15-40% interest is a guaranteed return that no investment can reliably beat.",
        })
    if emergency_gap > 0:
        actions.append({
            "priority": "high",
            "title": "Build an emergency fund",
            "detail": f"Keep {EMERGENCY_TARGET_MONTHS} months of expenses ({emergency_target:,.0f}) in a savings account or liquid fund. You are {emergency_gap:,.0f} short (about {emergency_months:.1f} months covered now).",
        })
    if not p.has_health_insurance:
        actions.append({
            "priority": "high",
            "title": "Get health insurance",
            "detail": "One hospital stay can wipe out years of savings. Buy a health policy for yourself and your family before investing in volatile assets.",
        })
    if p.dependents > 0 and not p.has_life_insurance:
        cover = p.monthly_income * 12 * 10
        actions.append({
            "priority": "high",
            "title": "Get term life insurance",
            "detail": f"With {p.dependents} dependent(s), a pure term plan of roughly 10x annual income ({cover:,.0f}) protects them. Avoid insurance-plus-investment products; they usually have high costs.",
        })
    if dti > 0.36:
        actions.append({
            "priority": "medium",
            "title": "Reduce your debt load",
            "detail": f"{dti:.0%} of your income goes to loan repayments. Above 36% is considered stretched; avoid new loans and prepay the highest-rate one.",
        })
    if 0 < savings_rate < 0.20:
        actions.append({
            "priority": "medium",
            "title": "Raise your savings rate toward 20%",
            "detail": f"You save {savings_rate:.0%} of income. Automating a transfer on payday is the most reliable way to raise it.",
        })
    if surplus > 0 and p.high_interest_debt <= 0 and emergency_gap <= 0:
        actions.append({
            "priority": "good",
            "title": "You're ready to invest",
            "detail": f"Your foundations are in place. A monthly investment of up to {surplus * 0.8:,.0f} (80% of your surplus) is sustainable.",
        })

    grade = "Excellent" if score >= 85 else "Good" if score >= 65 else "Fair" if score >= 45 else "Needs attention"
    return {
        "score": score,
        "grade": grade,
        "metrics": {
            "monthly_surplus": surplus,
            "savings_rate": savings_rate,
            "emergency_months": None if emergency_months == float("inf") else emergency_months,
            "emergency_target": emergency_target,
            "emergency_gap": emergency_gap,
            "debt_to_income": dti,
        },
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# Risk profile
# ---------------------------------------------------------------------------

REACTION_POINTS = {"sell_all": 0, "sell_some": 8, "hold": 17, "buy_more": 25}
EXPERIENCE_POINTS = {"none": 0, "beginner": 5, "intermediate": 10, "advanced": 15}
STABILITY_POINTS = {"unstable": 0, "stable": 5, "very_stable": 8}

PROFILES = ["Conservative", "Moderately conservative", "Balanced", "Growth", "Aggressive"]


def risk_profile(p: AdvisorInput) -> dict:
    tolerance = (REACTION_POINTS[p.drawdown_reaction] + EXPERIENCE_POINTS[p.experience]) * 50 / 40

    horizon = p.investment_horizon_years
    capacity = 0
    capacity += 22 if horizon >= 10 else 16 if horizon >= 5 else 8 if horizon >= 3 else 0
    capacity += 12 if p.age < 30 else 10 if p.age < 40 else 7 if p.age < 50 else 4 if p.age < 60 else 1
    capacity += STABILITY_POINTS[p.income_stability]
    outgoings = p.monthly_expenses + p.monthly_debt_payment
    months = p.current_savings / outgoings if outgoings > 0 else 12
    capacity += 8 if months >= 6 else 4 if months >= 3 else 0

    score = round(tolerance + capacity)
    idx = 0 if score < 25 else 1 if score < 45 else 2 if score < 65 else 3 if score < 80 else 4

    notes = []
    # Money needed soon must not sit in volatile assets, whatever the appetite.
    if horizon < 3 and idx > 1:
        idx = 1
        notes.append("Capped at Moderately conservative because you need this money within 3 years.")
    if tolerance < 15 and idx > 2:
        idx = 2
        notes.append("Capped at Balanced because you said you would sell after a big fall; a riskier portfolio would likely make you lock in losses.")

    return {
        "score": score,
        "tolerance": round(tolerance),
        "capacity": capacity,
        "profile": PROFILES[idx],
        "level": idx,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

# Strategic weights per profile: equity, bonds, gold, reit, crypto.
TEMPLATES = [
    {"equity": 0.20, "bonds": 0.65, "gold": 0.10, "reit": 0.05, "crypto": 0.00},
    {"equity": 0.40, "bonds": 0.45, "gold": 0.10, "reit": 0.05, "crypto": 0.00},
    {"equity": 0.60, "bonds": 0.25, "gold": 0.08, "reit": 0.05, "crypto": 0.02},
    {"equity": 0.75, "bonds": 0.12, "gold": 0.06, "reit": 0.04, "crypto": 0.03},
    {"equity": 0.85, "bonds": 0.00, "gold": 0.05, "reit": 0.03, "crypto": 0.07},
]
# Share of the equity sleeve that may go to individual stocks (rest in index funds).
STOCK_SHARE = [0.0, 0.0, 0.2, 0.3, 0.4]
DOMESTIC_SHARE = {"IN": 0.7, "US": 0.8}
MAX_SINGLE_STOCK = 0.05

# Long-run (return, volatility) per bucket in the investor's home currency.
ASSUMPTIONS = {
    "IN": {"equity_dom": (0.11, 0.18), "equity_intl": (0.09, 0.16), "bonds": (0.065, 0.04), "gold": (0.07, 0.15), "reit": (0.08, 0.18), "crypto": (0.10, 0.65)},
    "US": {"equity_dom": (0.08, 0.16), "equity_intl": (0.07, 0.17), "bonds": (0.045, 0.05), "gold": (0.05, 0.15), "reit": (0.07, 0.19), "crypto": (0.10, 0.65)},
}
BUCKETS = ["equity_dom", "equity_intl", "bonds", "gold", "reit", "crypto"]
CORR = np.array([
    # eq_dom eq_int bonds gold  reit  crypto
    [1.00, 0.75, 0.10, 0.05, 0.60, 0.30],
    [0.75, 1.00, 0.10, 0.05, 0.55, 0.30],
    [0.10, 0.10, 1.00, 0.20, 0.20, 0.00],
    [0.05, 0.05, 0.20, 1.00, 0.10, 0.10],
    [0.60, 0.55, 0.20, 0.10, 1.00, 0.20],
    [0.30, 0.30, 0.00, 0.10, 0.20, 1.00],
])

BUCKET_LABELS = {
    "equity_dom": "Domestic equity",
    "equity_intl": "International equity",
    "bonds": "Bonds / debt",
    "gold": "Gold",
    "reit": "Real estate (REITs)",
    "crypto": "Crypto",
}


def bucket_weights(level: int, country: str, wants_crypto: bool) -> dict[str, float]:
    t = dict(TEMPLATES[level])
    if not wants_crypto:
        t["equity"] += t.pop("crypto")
        t["crypto"] = 0.0
    dom = DOMESTIC_SHARE[country]
    return {
        "equity_dom": t["equity"] * dom,
        "equity_intl": t["equity"] * (1 - dom),
        "bonds": t["bonds"],
        "gold": t["gold"],
        "reit": t["reit"],
        "crypto": t["crypto"],
    }


def portfolio_stats(weights: dict[str, float], country: str) -> tuple[float, float]:
    a = ASSUMPTIONS[country]
    w = np.array([weights[b] for b in BUCKETS])
    mu = np.array([a[b][0] for b in BUCKETS])
    vol = np.array([a[b][1] for b in BUCKETS])
    cov = np.outer(vol, vol) * CORR
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


# Core holdings per bucket: these are held for diversification regardless of
# short-term signal. The signal only decides whether to buy all at once or stagger.
CORE = {
    "IN": {
        "equity_dom": [("NIFTYBEES.NS", 0.75), ("JUNIORBEES.NS", 0.25)],
        "equity_intl": [("SPY", 0.7), ("QQQ", 0.3)],
        "bonds": [("EBBETF0431.NS", 0.6), ("GILT5YBEES.NS", 0.4)],
        "gold": [("GOLDBEES.NS", 1.0)],
        "reit": [("EMBASSY.NS", 1.0)],
        "crypto": [("BTC-USD", 0.7), ("ETH-USD", 0.3)],
    },
    "US": {
        "equity_dom": [("VTI", 0.75), ("QQQ", 0.25)],
        "equity_intl": [("VXUS", 1.0)],
        "bonds": [("BND", 0.7), ("IEF", 0.3)],
        "gold": [("GLD", 1.0)],
        "reit": [("VNQ", 1.0)],
        "crypto": [("BTC-USD", 0.7), ("ETH-USD", 0.3)],
    },
}

BUCKET_NOTES = {
    ("IN", "bonds"): "For Indian investors, PPF, EPF/VPF, high-quality debt mutual funds or bank FDs are good alternatives for this slice.",
    ("IN", "gold"): "Sovereign Gold Bonds (when available) add 2.5% interest on top of gold's return and are tax-free at maturity.",
    ("US", "bonds"): "Treasury I-bonds or a high-yield savings account also work for the short-term part of this slice.",
}


def build_plan(p: AdvisorInput) -> dict:
    health = financial_health(p)
    profile = risk_profile(p)
    weights = bucket_weights(profile["level"], p.country, p.include_crypto)
    exp_return, exp_vol = portfolio_stats(weights, p.country)

    # Money that must go to foundations before it can be invested: expensive
    # debt first (the guaranteed "return" is highest), then the emergency fund.
    reserve_for_debt = min(p.lump_sum, p.high_interest_debt)
    reserve_for_emergency = min(p.lump_sum - reserve_for_debt, health["metrics"]["emergency_gap"])
    investable_lump = max(0.0, p.lump_sum - reserve_for_emergency - reserve_for_debt)
    debt_left = p.high_interest_debt - reserve_for_debt
    emergency_left = health["metrics"]["emergency_gap"] - reserve_for_emergency
    surplus = max(0.0, health["metrics"]["monthly_surplus"])
    monthly = min(p.monthly_investment, surplus) if surplus > 0 else 0.0

    stock_class = EQUITY_IN if p.country == "IN" else EQUITY_US
    stock_share = STOCK_SHARE[profile["level"]]

    needed = {s for bucket in CORE[p.country].values() for s, _ in bucket}
    stock_universe = [a.symbol for a in UNIVERSE if a.asset_class == stock_class and a.region == p.country]
    needed.update(stock_universe)
    signals = clients.market.get("/api/market/signals", ttl=900, params={"symbols": ",".join(sorted(needed))}) or {}

    def holding(symbol: str, weight: float, bucket: str, role: str) -> dict:
        s = signals.get(symbol) or {}
        score = s.get("score")
        label = s.get("signal") if score is not None else "Data unavailable"
        if role == "core":
            if score is not None and score <= BEARISH_AT:
                timing = "Signal is weak right now: stagger the lump sum over 3-6 months instead of buying all at once."
            else:
                timing = "Core long-term holding; invest the lump sum now and continue monthly."
        else:
            timing = "Satellite pick chosen for its currently strong trend and momentum. Review quarterly; trim if the signal turns bearish."
        return {
            "symbol": symbol,
            "name": lookup(symbol).name,
            "bucket": bucket,
            "role": role,
            "weight": weight,
            "lump_sum_amount": round(investable_lump * weight, 2),
            "monthly_amount": round(monthly * weight, 2),
            "signal": label,
            "score": None if score is None else round(score, 1),
            "currency": lookup(symbol).currency,
            "note": timing,
        }

    holdings: list[dict] = []
    for bucket, bw in weights.items():
        if bw <= 0:
            continue
        core_w = bw
        if bucket == "equity_dom" and stock_share > 0:
            ranked = sorted(
                (s for s in stock_universe if (signals.get(s) or {}).get("score") is not None and signals[s]["score"] >= BULLISH_AT),
                key=lambda s: signals[s]["score"],
                reverse=True,
            )
            picks = ranked[:4]
            if picks:
                # Never more than 5% of the portfolio in any single stock.
                stock_w = min(bw * stock_share, MAX_SINGLE_STOCK * len(picks))
                core_w = bw - stock_w
                for sym in picks:
                    holdings.append(holding(sym, stock_w / len(picks), bucket, "satellite"))
        for sym, share in CORE[p.country][bucket]:
            holdings.append(holding(sym, core_w * share, bucket, "core"))

    buckets = [
        {
            "key": b,
            "label": BUCKET_LABELS[b],
            "weight": w,
            "amount": round(investable_lump * w, 2),
            "monthly": round(monthly * w, 2),
            "note": BUCKET_NOTES.get((p.country, b)),
        }
        for b, w in weights.items()
        if w > 0
    ]

    steps = []
    if reserve_for_debt > 0:
        steps.append(f"Use {reserve_for_debt:,.0f} of your lump sum to pay off high-interest debt.")
    if reserve_for_emergency > 0:
        steps.append(f"Park {reserve_for_emergency:,.0f} in a savings account or liquid fund as your emergency fund.")
    foundations_left = debt_left + emergency_left
    if foundations_left > 0 and surplus > 0:
        what = " and ".join(x for x in (
            f"{debt_left:,.0f} of remaining high-interest debt" if debt_left > 0 else "",
            f"{emergency_left:,.0f} to finish your emergency fund" if emergency_left > 0 else "",
        ) if x)
        months = foundations_left / surplus
        steps.append(f"Before starting monthly investments, put your monthly surplus ({surplus:,.0f}) toward {what}: about {months:.0f} month(s). The projection below assumes investing starts now, so it is slightly optimistic.")
    if investable_lump > 0:
        steps.append(f"Invest the remaining {investable_lump:,.0f} across the holdings below.")
    if monthly > 0:
        then = "Then set up" if foundations_left > 0 else "Set up"
        steps.append(f"{then} an automatic monthly investment of {monthly:,.0f}, split the same way.")
    if p.monthly_investment > monthly:
        steps.append(f"Your planned monthly investment ({p.monthly_investment:,.0f}) is more than your monthly surplus ({surplus:,.0f}); the plan uses {monthly:,.0f}.")
    steps.append("Rebalance once a year back to these target weights; don't react to day-to-day moves.")

    return {
        "currency": "INR" if p.country == "IN" else "USD",
        "health": health,
        "risk_profile": profile,
        "expected_annual_return": exp_return,
        "expected_volatility": exp_vol,
        "worst_year_estimate": exp_return - 2 * exp_vol,
        "investable_lump_sum": investable_lump,
        "monthly_investment": monthly,
        "reserved": {"emergency_fund": reserve_for_emergency, "debt_payoff": reserve_for_debt},
        "buckets": buckets,
        "holdings": holdings,
        "steps": steps,
        "projection": project_goal(
            initial=investable_lump,
            monthly=monthly,
            years=max(1, p.investment_horizon_years),
            annual_return=exp_return,
            annual_vol=exp_vol,
            target=p.goal_amount,
            step_up=p.annual_step_up,
        ),
    }


# ---------------------------------------------------------------------------
# Goal projection
# ---------------------------------------------------------------------------


def project_goal(
    initial: float,
    monthly: float,
    years: int,
    annual_return: float,
    annual_vol: float,
    target: float | None = None,
    step_up: float = 0.0,
    inflation: float = 0.05,
    paths: int = 5000,
    seed: int = 11,
) -> dict:
    """Monte Carlo of a lump sum plus monthly contributions (optionally rising
    each year). Final wealth is linear in the monthly amount for a fixed set
    of paths, which lets us solve for the contribution needed to hit a target."""
    months = years * 12
    mu_m = np.log1p(annual_return) / 12 - 0.5 * (annual_vol**2) / 12
    sd_m = annual_vol / np.sqrt(12)
    rng = np.random.default_rng(seed)
    growth = np.exp(rng.normal(mu_m, sd_m, size=(paths, months)))

    contrib = np.array([(1 + step_up) ** (m // 12) for m in range(months)])  # per unit of monthly
    # Value at the end of each month from the lump sum (G) and from one unit of monthly contribution (S).
    G = np.cumprod(growth, axis=1)
    S = np.zeros((paths, months))
    acc = np.zeros(paths)
    for m in range(months):
        acc = (acc + contrib[m]) * growth[:, m]
        S[:, m] = acc
    wealth = initial * G + monthly * S

    year_ends = [12 * y - 1 for y in range(1, years + 1)]
    pct = np.percentile(wealth[:, year_ends], [10, 50, 90], axis=0)
    contributed = initial + monthly * np.cumsum(contrib)
    timeline = [
        {
            "year": y,
            "contributed": float(contributed[12 * y - 1]),
            "p10": float(pct[0, i]),
            "p50": float(pct[1, i]),
            "p90": float(pct[2, i]),
        }
        for i, y in enumerate(range(1, years + 1))
    ]

    final = wealth[:, -1]
    out = {
        "years": years,
        "assumed_return": annual_return,
        "assumed_volatility": annual_vol,
        "total_contributed": float(contributed[-1]),
        "median_final": float(np.median(final)),
        "pessimistic_final": float(np.percentile(final, 10)),
        "optimistic_final": float(np.percentile(final, 90)),
        "median_final_in_todays_money": float(np.median(final) / (1 + inflation) ** years),
        "inflation_assumed": inflation,
        "timeline": timeline,
        "target": target,
        "prob_reach_target": None,
        "monthly_needed_for_75pct": None,
    }
    if target and target > 0:
        out["prob_reach_target"] = float((final >= target).mean())
        g, s = G[:, -1], S[:, -1]
        lo, hi = 0.0, max(target / months, 1.0) * 4
        for _ in range(60):
            mid = (lo + hi) / 2
            if ((initial * g + mid * s) >= target).mean() >= 0.75:
                hi = mid
            else:
                lo = mid
        out["monthly_needed_for_75pct"] = float(hi)
    return out
