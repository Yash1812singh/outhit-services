"""Turns the research engines into ranked, explained recommendations per horizon,
and logs every pick so its real-world record can be measured.

Long term (years)
  Fundamentals carry most of the weight, because price-only methods showed no
  out-of-sample edge at a 1-year horizon in our tournaments:
    fundamentals (value, momentum, growth, trend, analysts, dividend)   0.55
    quality check (ROE, debt, margins, growth, cash flow)                0.15
    price methods at 1 year / 3 months, weighted by their verdict        up to 0.30 / 0.15
    reputation 0.10, recent news 0.05
  Companies with 2+ red flags are dropped; max 3 per sector.

Short term (1-4 weeks)
  The tournament's adaptive ensemble at 1 month and 1 week (weighted by how
  well it did out-of-sample), plus news sentiment and corporate events.
  Each idea has a volatility-based stop-loss and a 2:1 target.

F&O
  Defined-risk option strategies for the strongest short-term ideas and the
  main indices (options.py). If the short-term ranking has no proven edge, the
  direction is treated as neutral rather than guessed.
"""

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import select

from research import backend, data, earnings, governance, intel, intraday_lab, jobs, options, regime, tournament
from research.db import RecommendationLog, SessionLocal, aware, now

log = logging.getLogger(__name__)
TTL = 1800
VERDICT_WEIGHT = {"meaningful": 1.0, "modest": 0.67, "none": 0.15, "negative": 0.0}
BENCHMARK = {"in": "^NSEI", "us": "^GSPC"}
_tracked: set[str] = set()  # picks already handed to the news/outlook tracker


def _pending(t: dict) -> dict | None:
    return t if t.get("status") != "ready" else None


def _rank_map(h: dict) -> dict[str, dict]:
    return {r["symbol"]: r for r in h.get("ranking", [])} if h.get("available") else {}


def _price(sym: str, fallback: float | None) -> float | None:
    q = intel.quote(sym)
    return q["price"] if q else fallback


def _ai_view(sym: str) -> dict | None:
    o = intel.latest([sym]).get(sym)
    if not o:
        return None
    comps = {c["key"]: c["value"] for c in o["components"]}
    return {"score": o["score"], "label": o["label"], "reputation": o["reputation"], "news": comps.get("news"), "events": comps.get("events"), "news_7d": o["news_7d"]}


def _quality(sym: str) -> dict:
    return backend._get(f"/api/stocks/quality/{sym}", ttl=24 * 3600, timeout=60) or {}


# ---------------------------------------------------------------------------
# Long term
# ---------------------------------------------------------------------------

def _method_ic(t: dict, horizon: str, method: str) -> float | None:
    for m in t["horizons"].get(horizon, {}).get("methods", []):
        if m["method"] == method:
            return m["holdout"].get("ic")
    return None


def evidence_weight(t: dict, method: str, horizon: str = "63") -> float:
    """Scale a factor's weight by how it did on the out-of-sample holdout in this market:
    IC 0 -> 0.3x, IC +0.03 -> 0.6x, IC +0.07 -> 1.0x (capped at 1.3x); negative IC -> 0."""
    ic = _method_ic(t, horizon, method)
    if ic is None:
        return 0.5
    return float(np.clip(0.3 + 10 * ic, 0, 1.3))


def _pct(t: dict, factor: str, sym: str) -> float | None:
    v = t.get("factors_now", {}).get(factor, {}).get("pct", {}).get(sym)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def long_term(market: str, progress=lambda m: None) -> dict:
    t = tournament.get(market)
    if _pending(t):
        return t
    rg = regime.detect(market)
    uni = data.universe(market, tournament.UNIVERSE).set_index("symbol")
    h252, h63 = t["horizons"]["252"], t["horizons"]["63"]
    r252, r63 = _rank_map(h252), _rank_map(h63)
    fund = uni["score"].rank(pct=True)

    weights = {
        "quality_history": 0.20 * evidence_weight(t, "quality"),
        "momentum": 0.15 * evidence_weight(t, "momentum_12_1") * rg["momentum_scale"],
        "value": 0.10 * evidence_weight(t, "value_ey"),
        "fundamentals": 0.15,
        "price_methods_1y": 0.15 * VERDICT_WEIGHT[h252.get("verdict_level", "none")],
        "price_methods_3m": 0.08 * VERDICT_WEIGHT[h63.get("verdict_level", "none")],
        "reputation": 0.06,
        "news": 0.04,
    }
    late = {"quality_check": 0.10, "analyst_revisions": 0.07, "governance": 0.07 if market == "in" else 0.0}

    def value_of(sym: str, k: str, ai: dict | None) -> float | None:
        return {
            "quality_history": _pct(t, "quality", sym),
            "momentum": _pct(t, "mom_12_1", sym),
            "value": _pct(t, "earnings_yield", sym),
            "fundamentals": float(fund[sym]) if sym in fund and pd.notna(fund[sym]) else None,
            "price_methods_1y": r252.get(sym, {}).get("percentile"),
            "price_methods_3m": r63.get(sym, {}).get("percentile"),
            "reputation": (ai["reputation"]["score"] / 100) if ai else None,
            "news": (0.5 + 0.5 * ai["news"]) if ai and ai["news"] is not None else None,
        }[k]

    def blend(parts: dict) -> float:
        tw = sum(w for w, _ in parts.values())
        return sum(w * (0.5 if v is None else v) for w, v in parts.values()) / tw if tw else 0.5

    rows = []
    for sym in uni.index:
        if pd.isna(uni.at[sym, "score"]):
            continue
        ai = _ai_view(sym)
        parts = {k: (w, value_of(sym, k, ai)) for k, w in weights.items()}
        rows.append({"symbol": sym, "parts": parts, "pre": blend(parts)})
    rows.sort(key=lambda r: r["pre"], reverse=True)
    cands = rows[:50]
    syms = [c["symbol"] for c in cands]
    progress("Checking quality, analyst revisions and governance for the top candidates…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        quals = dict(zip(syms, ex.map(_quality, syms)))
    revs = earnings.revisions_many(syms)
    with ThreadPoolExecutor(max_workers=4) as ex:
        govs = dict(zip(syms, ex.map(lambda s: governance.assess(s, uni.at[s, "name"], market), syms)))

    picks, per_sector = [], {}
    for c in sorted(cands, key=lambda c: c["pre"], reverse=True):
        sym = c["symbol"]
        q, rv, gv = quals.get(sym) or {}, revs.get(sym), govs.get(sym) or {}
        flags = list(q.get("red_flags") or []) + list(gv.get("flags") or [])
        if len(flags) >= 2:
            continue
        c["parts"]["quality_check"] = (late["quality_check"], (q.get("quality") / 100) if q.get("quality") is not None else None)
        c["parts"]["analyst_revisions"] = (late["analyst_revisions"], (0.5 + 0.5 * rv["score"]) if rv else None)
        if late["governance"]:
            c["parts"]["governance"] = (late["governance"], gv.get("score"))
        score = 100 * blend(c["parts"])
        c["final"] = score
        c["q"], c["rv"], c["gv"], c["flags"] = q, rv, gv, flags
    ranked = sorted([c for c in cands if "final" in c], key=lambda c: c["final"], reverse=True)
    for c in ranked:
        sym = c["symbol"]
        sector = uni.at[sym, "sector"]
        if per_sector.get(sector, 0) >= 3:
            continue
        per_sector[sector] = per_sector.get(sector, 0) + 1
        u = uni.loc[sym]
        q, rv, gv = c["q"], c["rv"], c["gv"]
        reasons = list(q.get("strengths") or []) + list(gv.get("strengths") or [])
        qp, mp = _pct(t, "quality", sym), _pct(t, "mom_12_1", sym)
        if qp is not None and qp >= 0.7:
            reasons.insert(0, f"Top {max(1, round((1 - qp) * 100))}% quality (profitable, low debt)")
        if mp is not None and mp >= 0.7:
            reasons.append(f"Top {max(1, round((1 - mp) * 100))}% momentum")
        if rv and rv.get("up_30d", 0) > rv.get("down_30d", 0):
            reasons.append(f"Analysts raising estimates ({rv['up_30d']:.0f}↑ {rv['down_30d']:.0f}↓)")
        if pd.notna(u.get("pe")) and u["pe"] > 0:
            reasons.append(f"P/E {u['pe']:.1f}")
        if pd.notna(u.get("dividend_yield_pct")) and u["dividend_yield_pct"] >= 2:
            reasons.append(f"Dividend yield {u['dividend_yield_pct']:.1f}%")
        picks.append({
            "symbol": sym, "name": u["name"], "sector": sector, "currency": u["currency"], "size_tier": u.get("size_tier"),
            "price": _price(sym, u.get("price")), "score": round(c["final"], 1),
            "components": {k: {"weight": round(w, 3), "value": None if v is None else round(float(v), 3)} for k, (w, v) in c["parts"].items()},
            "reasons": reasons[:5], "red_flags": c["flags"], "analyst_rating": u.get("analyst_rating"),
            "analyst_revisions": rv, "governance": {k: gv.get(k) for k in ("pledge", "insider")} if gv.get("available") else None,
            "ai": _ai_view(sym), "methods_1y": r252.get(sym, {}).get("by_method"),
            "volatility": t["companies"].get(sym, {}).get("vol_63"),
        })
        if len(picks) == 15:
            break
    return {
        "status": "ready", "market": market, "horizon": "long_term", "computed_at": now().isoformat(),
        "regime": rg,
        "weights": {**{k: round(w, 3) for k, w in weights.items()}, **late},
        "evidence": {
            "price_1y": {"verdict": h252.get("verdict_level"), "weight": round(weights["price_methods_1y"], 3)},
            "price_3m": {"verdict": h63.get("verdict_level"), "weight": round(weights["price_methods_3m"], 3)},
            "factor_ic_3m": {m: _method_ic(t, "63", m) for m in ("quality", "value_ey", "momentum_12_1", "quality_momentum", "earnings_surprise")},
        },
        "method": ("Research-based multi-factor model: point-in-time quality (profitability, safety, growth) and 12-1 momentum lead, plus value, "
                   "the fundamentals screen, the tested price-method ensemble, analyst revisions, governance (promoter pledging, insider buying) and "
                   "reputation/news. Every factor's weight is scaled by how it performed out-of-sample in this market; momentum is also scaled down in "
                   "risky market regimes. 2+ red flags excluded; max 3 per sector."),
        "picks": picks,
    }


# ---------------------------------------------------------------------------
# Short term
# ---------------------------------------------------------------------------

def _severe_governance(gv: dict) -> bool:
    """Exclude from buy lists: half or more of promoter shares pledged, or lenders invoking pledges."""
    pledge = gv.get("pledge") or {}
    insider = gv.get("insider") or {}
    return (pledge.get("pledged_pct_of_promoter") or 0) >= 50 or (insider.get("pledges_invoked_180d") or 0) > 0


def short_term(market: str, progress=lambda m: None) -> dict:
    t = tournament.get(market)
    if _pending(t):
        return t
    h21, h5 = t["horizons"]["21"], t["horizons"]["5"]
    r21, r5 = _rank_map(h21), _rank_map(h5)
    rg = regime.detect(market)
    mom_family = {"momentum_12_1", "momentum_6m", "risk_adj_momentum", "high_52w", "trend", "technical", "quality_momentum"}
    def mscale(h):  # momentum-crash protection: shrink momentum-led ensembles in risky regimes
        members = set(h.get("ensemble", {}).get("members") or [])
        return rg["momentum_scale"] if members and members <= mom_family | {"ml_gbm", "ml_linear"} and members & mom_family else 1.0
    w21 = 0.50 * VERDICT_WEIGHT[h21.get("verdict_level", "none")] * mscale(h21)
    w5 = 0.20 * VERDICT_WEIGHT[h5.get("verdict_level", "none")] * mscale(h5)
    comp = t["companies"]
    rows = []
    for sym, info in comp.items():
        ai = _ai_view(sym)
        parts = {
            "methods_1m": (w21, r21.get(sym, {}).get("percentile", 0.5)),
            "methods_1w": (w5, r5.get(sym, {}).get("percentile", 0.5)),
            "news": (0.15, (0.5 + 0.5 * ai["news"]) if ai and ai["news"] is not None else 0.5),
            "events": (0.08, (0.5 + 0.5 * ai["events"]) if ai and ai["events"] is not None else 0.5),
            "earnings_surprise": (0.05 * evidence_weight(t, "earnings_surprise", "21"), _pct(t, "surprise", sym) if _pct(t, "surprise", sym) is not None else 0.5),
        }
        wsum = sum(w for w, _ in parts.values())
        # Missing weight counts as neutral (0.5) so a stock needs real evidence to rank high.
        score = (sum(w * v for w, v in parts.values()) + (1 - wsum) * 0.5)
        price = _price(sym, info["price"])
        vol = info.get("vol_63") or 0.3
        sm = vol * math.sqrt(21 / 252)
        rows.append({
            "symbol": sym, "name": info["name"], "sector": info["sector"], "currency": info["currency"], "price": price,
            "score": round(100 * (score - 0.5) * 2, 1),
            "components": {k: {"weight": round(w, 3), "value": round(v, 3)} for k, (w, v) in parts.items()},
            "methods": r21.get(sym, {}).get("by_method"),
            "hist_prob_up": r21.get(sym, {}).get("hist_prob_up"), "hist_avg_return": r21.get(sym, {}).get("hist_avg_return"),
            "ai": ai,
            "levels": {"stop_loss": price * (1 - sm), "target": price * (1 + 2 * sm), "short_stop": price * (1 + sm), "short_target": price * (1 - 2 * sm)} if price else None,
            "volatility": vol,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    # Second pass on the shortlist: analyst revisions (current only) and governance red flags.
    short = rows[:30] + rows[-20:]
    progress("Checking analyst revisions and governance for the shortlist…")
    revs = earnings.revisions_many([r["symbol"] for r in short])
    for r in short:
        rv = revs.get(r["symbol"])
        r["analyst_revisions"] = rv
        if rv:
            r["components"]["analyst_revisions"] = {"weight": 0.07, "value": round(0.5 + 0.5 * rv["score"], 3)}
            r["score"] = round(r["score"] + 100 * 0.07 * rv["score"], 1)  # +-7 points at most
        gv = governance.assess(r["symbol"], r["name"], market) if market == "in" else {"flags": []}
        r["governance_flags"] = gv.get("flags", [])
        r["_severe"] = _severe_governance(gv)
    bullish = sorted([r for r in rows[:30] if not r.pop("_severe", False)], key=lambda r: r["score"], reverse=True)[:15]
    for r in rows[-20:]:
        r.pop("_severe", None)
    bearish = sorted(rows[-20:], key=lambda r: r["score"])[:10]
    edge = h21.get("verdict_level") in ("meaningful", "modest") or h5.get("verdict_level") in ("meaningful", "modest")
    return {
        "status": "ready", "market": market, "horizon": "short_term", "computed_at": now().isoformat(),
        "evidence": {"methods_1m": {"verdict": h21.get("verdict_level"), "members": h21.get("ensemble", {}).get("members"), "weight": w21},
                     "methods_1w": {"verdict": h5.get("verdict_level"), "members": h5.get("ensemble", {}).get("members"), "weight": w5}},
        "has_edge": edge,
        "regime": rg,
        "method": "Adaptive method ensemble (now including quality, value and earnings-surprise methods) at 1 month (up to 50%) and 1 week (up to 20%), each weighted by its out-of-sample verdict and cut back in risky market regimes, plus LLM-read news 15%, corporate events 8%, earnings surprise (post-earnings drift) and analyst revisions. Heavily pledged or pledge-invoked companies are excluded. Stop-loss = 1 monthly standard deviation; target = 2x that.",
        "bullish": bullish,
        "bearish": bearish,
    }


# ---------------------------------------------------------------------------
# F&O
# ---------------------------------------------------------------------------

def fo(market: str, progress=lambda m: None) -> dict:
    st = short_term(market)
    if _pending(st):
        return st
    edge = st["has_edge"]
    rg = st.get("regime") or regime.detect(market)
    targets: list[tuple[str, str, str, float]] = []  # (option symbol, yahoo symbol, direction, drift to expiry)
    t = tournament.get(market)
    q21 = t["horizons"]["21"].get("quintiles", []) if t.get("status") == "ready" else []
    avg = np.mean([q["avg_return"] for q in q21 if q["avg_return"] is not None]) if q21 else 0.0
    # Expected excess return of the top / bottom fifth over a month, from the out-of-sample test.
    tilt_up = max(0.0, (q21[-1]["avg_return"] or 0) - avg) if q21 and edge else 0.0
    tilt_down = min(0.0, (q21[0]["avg_return"] or 0) - avg) if q21 and edge else 0.0
    if market == "in":
        for idx, ysym in (("NIFTY", "^NSEI"), ("BANKNIFTY", "^NSEBANK")):
            tech = backend.technical(ysym) or {}
            sc = tech.get("score") or 0
            targets.append((idx, ysym, "bullish" if sc >= 25 else "bearish" if sc <= -25 else "neutral", 0.0))
        fo_list = set(options.nse_fo_stocks())
        name = lambda s: s.split(".")[0]  # noqa: E731
        bulls = [r for r in st["bullish"] if name(r["symbol"]) in fo_list][:5]
        bears = [r for r in st["bearish"] if name(r["symbol"]) in fo_list][:4]
        targets += [(name(r["symbol"]), r["symbol"], "bullish" if edge else "neutral", tilt_up) for r in bulls]
        targets += [(name(r["symbol"]), r["symbol"], "bearish" if edge else "neutral", tilt_down) for r in bears]
    else:
        for etf in ("SPY", "QQQ"):
            sc = (backend.technical(etf) or {}).get("score") or 0
            targets.append((etf, etf, "bullish" if sc >= 25 else "bearish" if sc <= -25 else "neutral", 0.0))
        targets += [(r["symbol"], r["symbol"], "bullish" if edge else "neutral", tilt_up) for r in st["bullish"][:5]]
        targets += [(r["symbol"], r["symbol"], "bearish" if edge else "neutral", tilt_down) for r in st["bearish"][:4]]

    progress(f"Analysing option chains for {len(targets)} underlyings…")

    def one(t):
        sym, ysym, direction, drift = t
        try:
            a = options.analytics(market, sym, ysym)
            strat = options.choose_strategy(a, direction, drift * min(1.0, a["days_to_expiry"] / 30), allow_credit=rg["regime"] != "risk-off")
            return {"underlying": sym, "yahoo_symbol": ysym, "direction": direction, "analytics": options.public(a), "strategy": strat}
        except Exception as exc:
            return {"underlying": sym, "yahoo_symbol": ysym, "direction": direction, "error": str(exc)[:200]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, targets))
    return {
        "status": "ready", "market": market, "computed_at": now().isoformat(),
        "direction_has_edge": edge,
        "regime": rg,
        "playbook": "Direction from the short-term ranking (or index trend). For each underlying, several defined-risk strategies are built from the live option chain and scored on expected profit per unit of risk, using Outhit's volatility forecast (implied blended with recent realised volatility) plus the ranking's tested monthly edge as drift. If nothing has positive expected value, the answer is no trade.",
        "unit_note": "Premiums and profits are per share/unit. Multiply by the lot size (NSE) or by 100 (US) for one contract.",
        "ideas": results,
        "warning": "Options can lose 100% of the premium paid, and spreads can lose their full maximum loss. SEBI found ~9 in 10 individual F&O traders in India lost money. Trade small, and only with money you can afford to lose.",
    }


# ---------------------------------------------------------------------------
# Public entry points + ledger
# ---------------------------------------------------------------------------

BUILDERS = {"long_term": long_term, "short_term": short_term, "intraday": None, "fo": fo}


def get(market: str, horizon: str) -> dict:
    if horizon == "intraday":
        return intraday_lab.get(market)
    t = tournament.get(market)  # everything else builds on the tournament: surface its progress while it runs
    if t.get("status") != "ready":
        return {**t, "message": f"Running the method tournament first: {t.get('message', '')}"}
    res = jobs.run(f"rec:{market}:{horizon}", lambda p: BUILDERS[horizon](market, p), TTL)
    if res.get("status") == "ready" and horizon in ("long_term", "short_term"):
        try:
            record(market, horizon, res)
        except Exception:
            log.exception("could not log recommendations")
        syms = [p["symbol"] for k in ("picks", "bullish", "bearish") for p in res.get(k, [])]
        if syms and not _tracked.issuperset(syms):
            _tracked.update(syms)
            threading.Thread(target=intel.track, args=(syms,), daemon=True, name="track-picks").start()
    return res


def record(market: str, horizon: str, res: dict) -> None:
    day = now().date().isoformat()
    lists = [(1, res["picks"])] if horizon == "long_term" else [(1, res["bullish"]), (-1, res["bearish"])]
    with SessionLocal() as db:
        if db.scalar(select(RecommendationLog.id).where(RecommendationLog.day == day, RecommendationLog.market == market, RecommendationLog.horizon == horizon)):
            return
        for side, items in lists:
            for i, p in enumerate(items[:10]):
                if p.get("price"):
                    db.add(RecommendationLog(day=day, market=market, horizon=horizon, symbol=p["symbol"], side=side, rank=i + 1, price=float(p["price"]), score=float(p["score"])))
        db.commit()


CHECKPOINTS = {"short_term": {"5d": 5, "21d": 21}, "long_term": {"21d": 21, "63d": 63, "126d": 126}}


def evaluate() -> int:
    with SessionLocal() as db:
        rows = [r for r in db.scalars(select(RecommendationLog).where(RecommendationLog.made_at <= now() - timedelta(days=2))).all()
                if not r.outcome or len(r.outcome) < len(CHECKPOINTS[r.horizon])]
        if not rows:
            return 0
        syms = sorted({r.symbol for r in rows} | {BENCHMARK[r.market] for r in rows if r.market in BENCHMARK})
        start = min(aware(r.made_at) for r in rows).date()
        raw = yf.download(syms, start=str(start), interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        n = 0
        for r in rows:
            try:
                c = (raw[r.symbol] if isinstance(raw.columns, pd.MultiIndex) else raw)["Close"].dropna()
                b = raw[BENCHMARK[r.market]]["Close"].dropna() if r.market in BENCHMARK else None
            except KeyError:
                continue
            after = c[c.index.date > aware(r.made_at).date()]
            base_b = b[b.index.date <= aware(r.made_at).date()] if b is not None else None
            out = dict(r.outcome or {})
            for k, days in CHECKPOINTS[r.horizon].items():
                if k in out or len(after) < days:
                    continue
                ret = float(after.iloc[days - 1] / r.price - 1)
                bench = None
                if b is not None and len(base_b):
                    b_after = b[b.index.date > aware(r.made_at).date()]
                    if len(b_after) >= days:
                        bench = float(b_after.iloc[days - 1] / base_b.iloc[-1] - 1)
                out[k] = {"return": ret, "benchmark": bench}
            if out != (r.outcome or {}):
                r.outcome = out
                n += 1
        db.commit()
        return n


def track_record() -> dict:
    with SessionLocal() as db:
        rows = list(db.scalars(select(RecommendationLog)).all())
    out = {"picks_logged": len(rows), "first_day": min((r.day for r in rows), default=None), "horizons": {}}
    for horizon, cps in CHECKPOINTS.items():
        out["horizons"][horizon] = {}
        for k in cps:
            done = [(r.side, r.outcome[k]) for r in rows if r.horizon == horizon and r.outcome and k in r.outcome]
            if not done:
                out["horizons"][horizon][k] = {"evaluated": 0}
                continue
            rets = np.array([s * o["return"] for s, o in done])
            excess = np.array([s * (o["return"] - o["benchmark"]) for s, o in done if o["benchmark"] is not None])
            out["horizons"][horizon][k] = {
                "evaluated": len(done),
                "avg_return": float(rets.mean()),
                "hit_rate": float((rets > 0).mean()),
                "avg_excess_vs_index": float(excess.mean()) if len(excess) else None,
                "beat_index_rate": float((excess > 0).mean()) if len(excess) else None,
            }
    return out
