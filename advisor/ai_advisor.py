"""Conversational advisor backed by Claude.

Claude never answers market questions from memory: it is given tools that call
the same signal, backtest and planning engine as the rest of the API, so every
number it quotes comes from today's data.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import anthropic
import groq
import httpx
from anthropic import beta_tool

from advisor import clients, planner
from advisor.config import get_settings
from advisor.schemas import AdvisorInput

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Outhit's financial advisor: a careful, plain-spoken guide for everyday investors, with data on stock markets in 55 countries.

How you work:
- For "which stocks will go up/down" questions, use predict_outperformers and lead with its validation verdict; if the verdict is "none" or "negative", say plainly that the model has not worked in that market and fall back to diversified, long-term advice.
- For recommendations (long-term, short-term, intraday, F&O), use outhit_ai_picks first: it combines every tested method and reports its evidence. stock_picks (long_term, swing or intraday) is the simpler screen for the relevant country, and sector_strength for sector questions. For the big picture (indices, currencies, oil, gold, yields), use global_markets. For one company, use company_snapshot. For news, results, policy changes or anything recent, use news_search (or web search when available) and cite the source. For a company's news-based outlook, reputation and controversies, use ai_outlook (Outhit's own AI).
- Never claim certainty or "guaranteed"/"maximum" profit. If asked for that, explain briefly that nobody can predict markets reliably, then give the best evidence-based answer. Every pick list includes how the method has performed or what could go wrong: pass that on.
- Intraday trading: always mention that most retail intraday traders lose money, quote the backtest verdict the intraday tool returns, and insist on a stop-loss and small position size.
- For any question about a specific stock, fund, crypto or market, call the tools first and base your answer on what they return (signal score, factor breakdown, backtest hit rate, forecast ranges, risk). Quote the actual numbers. If a tool fails, say you couldn't get data rather than guessing.
- Markets cannot be predicted with certainty. Present forecasts as probabilities and ranges ("about 60% chance of being higher in a year; 90% range -25% to +45%"), never as promises or single price targets. When the backtest shows the signal has had little edge on an asset, say so plainly.
- Put foundations before investing: emergency fund (6 months of expenses), paying off high-interest debt, health and term-life insurance. If the user's profile shows gaps, mention them when relevant.
- Prefer diversified, low-cost index funds as the core of a portfolio. Treat individual stocks and crypto as smaller satellite positions sized to the person's risk profile. Crypto above ~10% of a portfolio is speculation for almost everyone.
- Tailor advice to the user's profile below when one is available (country, age, horizon, risk profile). Use their currency.
- Mention taxes or product-specific rules only at a high level, and suggest confirming with a registered adviser (SEBI-registered in India, a fiduciary adviser in the US) for large or complex decisions.
- Be concise: lead with the answer, then the key numbers and the reasoning. Use short paragraphs or bullet points. No hype.

You are an educational tool, not a licensed adviser, and your output is not a recommendation to buy or sell any security."""


def _j(data) -> str:
    return json.dumps(data if data is not None else {"error": "That service is unavailable right now."}, default=str)


def _wait(fetch, limit: float = 120):
    """Poll an endpoint that computes in the background until it is ready (or time runs out)."""
    deadline = time.time() + limit
    while True:
        d = fetch()
        if not isinstance(d, dict) or d.get("status") not in ("running", "training") or time.time() > deadline:
            return d
        time.sleep(3)


@beta_tool
def analyze_asset(symbol: str) -> str:
    """Get today's full analysis of one instrument: price, returns, Outhit signal score (-100 to 100) with its factor breakdown, confidence, risk metrics, a backtest of how the signal has performed on this asset, and probability-based forecasts for 1m/3m/6m/1y.

    Args:
        symbol: Yahoo Finance symbol, e.g. AAPL, RELIANCE.NS (NSE), TCS.BO (BSE), BTC-USD, NIFTYBEES.NS, ^NSEI. Use search_symbol first if unsure.
    """
    d = clients.market.get(f"/api/market/asset/{symbol}", ttl=900)
    if isinstance(d, dict):
        for k in ("history", "score_history", "sparkline"):
            d.pop(k, None)
        (d.get("forecast") or {}).pop("cone", None)
    return _j(d)


@beta_tool
def market_overview(asset_class: str = "all") -> str:
    """Rank the funds, ETFs, gold, bonds, REITs and crypto Outhit tracks by today's signal score.

    Args:
        asset_class: One of all, index_fund, equity_us, equity_in, bonds, gold, reit, crypto.
    """
    d = clients.market.get("/api/market/overview", ttl=900) or {}
    groups = [g for g in d.get("groups", []) if asset_class in ("all", g["key"])]
    return _j([{**{k: a.get(k) for k in ("symbol", "name", "price", "change_1m", "change_1y", "risk_level", "signal", "score")}, "class": g["label"]}
               for g in groups for a in g["assets"]])


@beta_tool
def search_symbol(query: str) -> str:
    """Find the Yahoo Finance symbol for a company, fund or coin by name.

    Args:
        query: Company, fund or coin name, e.g. "Tata Motors" or "Vanguard S&P 500".
    """
    return _j(clients.market.get("/api/market/search", params={"q": query}))


@beta_tool
def project_savings(initial: float, monthly: float, years: int, annual_return: float, annual_volatility: float, target: float = 0.0) -> str:
    """Monte Carlo projection of a lump sum plus monthly investments. Returns pessimistic (10th pct), median and optimistic (90th pct) outcomes, probability of hitting a target, and the monthly amount needed for a 75% chance.

    Args:
        initial: Amount invested today.
        monthly: Amount invested each month.
        years: Number of years.
        annual_return: Expected annual return as a decimal, e.g. 0.10. Typical long-run: Indian equity 0.11, US equity 0.08, bonds 0.05-0.065, balanced portfolio 0.08-0.09.
        annual_volatility: Annual volatility as a decimal, e.g. 0.15. Typical: equity 0.16-0.18, bonds 0.05, balanced 0.10.
        target: Optional goal amount; 0 for none.
    """
    result = planner.project_goal(initial, monthly, max(1, min(years, 50)), annual_return, annual_volatility, target or None)
    result.pop("timeline", None)
    return _j(result)


def _trim_picks(d: dict) -> dict:
    for p in d.get("picks", []):
        p.pop("about", None)
        p.pop("sparkline", None)
    for k in ("setups", "watchlist", "movers"):
        for p in d.get(k, []):
            p.pop("sparkline", None)
    d["movers"] = d.get("movers", [])[:5]
    return d


@beta_tool
def stock_picks(market: str, horizon: str = "long_term", sector: str = "", size: str = "large_mid") -> str:
    """Simple stock screen for one country: long_term (fundamentals + quality), swing (technical, 1-4 weeks) or intraday (opening-range setups with a backtest).

    Args:
        market: Two-letter market code, e.g. in (India), us, gb, jp, de, cn, hk, au, ca, br.
        horizon: long_term, swing or intraday.
        sector: Optional sector filter for long_term, e.g. Technology, Financial Services, Healthcare, Energy.
        size: For long_term: large, mid, small, large_mid (default) or all.
    """
    params = {"horizon": horizon if horizon in ("long_term", "swing", "intraday") else "long_term", "size": size}
    if sector:
        params["sector"] = sector
    d = clients.market.get(f"/api/stocks/{market.lower()}/picks", ttl=900, timeout=180, params=params)
    return _j(_trim_picks(d) if isinstance(d, dict) else d)


@beta_tool
def sector_strength(market: str) -> str:
    """Every sector in one country ranked by strength (1-year momentum and share of stocks in uptrends), with median P/E, dividend yield and the top-scoring companies in each.

    Args:
        market: Two-letter market code, e.g. in, us, gb, jp.
    """
    return _j(clients.market.get(f"/api/stocks/{market.lower()}/sectors", ttl=900, timeout=120))


@beta_tool
def global_markets() -> str:
    """Snapshot of world markets: stock indices of ~35 countries, volatility (VIX), major currencies, gold, silver, oil, gas, copper, US bond yields and crypto, each with returns and the Outhit trend signal."""
    d = clients.market.get("/api/global/overview", ttl=900) or {}
    return _j([{"group": g["name"], **{k: a.get(k) for k in ("name", "symbol", "price", "change_1d", "change_1m", "change_1y", "signal", "score")}}
               for g in d.get("groups", []) for a in g["assets"]])


@beta_tool
def company_snapshot(symbol: str) -> str:
    """Fundamentals and quality for one listed company (ROE, debt, margins, growth, red flags) plus its technical analysis.

    Args:
        symbol: Yahoo Finance symbol, e.g. TCS.NS, AAPL, 7203.T, SAP.DE. Use search_symbol if unsure.
    """
    q = clients.market.get(f"/api/stocks/quality/{symbol}", ttl=24 * 3600) or {}
    t = clients.market.get(f"/api/market/asset/{symbol}", ttl=900) or {}
    for k in ("history", "score_history", "sparkline", "forecast", "backtest"):
        t.pop(k, None)
    return _j({"quality": q, "technical": t})


@beta_tool
def predict_outperformers(market: str, horizon_days: int = 63) -> str:
    """Machine-learning forecast of which stocks in a country are most likely to beat (bullish) or lag (bearish) the market over the next 1 or 3 months, with walk-forward validation results. Always report the validation verdict: in some markets the model has no edge or is worse than chance.

    Args:
        market: Two-letter market code, e.g. in, us, gb, jp.
        horizon_days: 21 (about 1 month) or 63 (about 3 months).
    """
    h = 21 if horizon_days < 42 else 63
    r = _wait(lambda: clients.market.get(f"/api/predict/{market.lower()}", params={"horizon": h}, timeout=60))
    if not isinstance(r, dict) or r.get("status") != "ready":
        return _j(r)
    v = r["validation"]
    return _j({
        "horizon": r["horizon_label"], "as_of": r["as_of"], "universe": r["universe"],
        "validation": {k: v[k] for k in ("period_start", "period_end", "auc", "information_coefficient", "baseline_prob_up", "top_decile", "bottom_decile", "verdict_level", "verdict")},
        "top_features": [f["label"] for f in r["feature_importance"][:5]],
        "bullish": r["bullish"][:10], "bearish": r["bearish"][:10], "caveats": r["caveats"],
    })


@beta_tool
def news_search(query: str, country: str = "us") -> str:
    """Search current news headlines (Google News) about a company, sector, economy or event. Returns up to 10 recent headlines with source and date.

    Args:
        query: What to search for, e.g. "Infosys results", "RBI interest rate", "oil prices OPEC".
        country: Two-letter edition code for local coverage: in, us, gb, au, ca, sg.
    """
    hl, gl = {"in": ("en-IN", "IN"), "gb": ("en-GB", "GB"), "au": ("en-AU", "AU"), "ca": ("en-CA", "CA"), "sg": ("en-SG", "SG")}.get(country.lower(), ("en-US", "US"))
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+when:14d&hl={hl}&gl={gl}&ceid={gl}:en"
    try:
        r = httpx.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
        root = ET.fromstring(r.text)
    except Exception as exc:
        return json.dumps({"error": f"News search failed: {exc}"})
    items = []
    for it in list(root.iter("item"))[:10]:
        src = it.find("source")
        items.append({"title": re.sub(r"\s+-\s+[^-]{2,60}$", "", it.findtext("title") or ""), "source": src.text if src is not None else None,
                      "published": it.findtext("pubDate")})
    return json.dumps({"query": query, "results": items})


@beta_tool
def ai_outlook(symbol: str) -> str:
    """Outhit AI's live view of one company: fused bullish/bearish outlook (price model + news sentiment + corporate events + reputation + today's trading), reputation score with controversies and governance risk, and the latest headlines scored by Outhit's own sentiment model.

    Args:
        symbol: Yahoo Finance symbol, e.g. RELIANCE.NS, AAPL.
    """
    d = clients.intelligence.get(f"/api/ai/outlook/{symbol}", timeout=60)
    if not isinstance(d, dict):
        return _j(d)
    o, rep = d.get("outlook") or {}, d.get("reputation") or {}
    return _j({
        "outlook": {k: o.get(k) for k in ("score", "label", "confidence", "agreement", "coverage")},
        "why": [{"signal": c["label"], "contribution": c["contribution"], "detail": c["detail"]} for c in o.get("components", [])],
        "reputation": {k: rep.get(k) for k in ("score", "grade", "breakdown", "governance", "controversies", "coverage_note")},
        "latest_news": [{k: n[k] for k in ("title", "source", "published_at", "label", "sentiment")} | {"events": [e["label"] for e in n["events"]]} for n in d.get("news", [])[:10]],
        "quote": {k: (d.get("quote") or {}).get(k) for k in ("price", "change_pct", "delayed_minutes", "trading")},
    })


@beta_tool
def outhit_ai_picks(market: str, horizon: str = "long_term") -> str:
    """Outhit AI's evidence-based recommendations for a country: long_term, short_term (1-4 weeks, with stop-loss/target), intraday (5-strategy lab with a verdict), fo (defined-risk option strategies; India and US only), rockets (likely top-10% winners over 6 months) or growers (steady top-20% compounders over 12 months). Always report the evidence/verdict it returns.

    Args:
        market: Two-letter market code, e.g. in, us, gb, jp.
        horizon: long_term, short_term, intraday, fo, rockets or growers.
    """
    m = market.lower()
    if horizon in ("rockets", "growers"):
        d = clients.research.get(f"/api/ai/rockets/{m}", params={"kind": horizon[:-1]}, timeout=60)
        if not isinstance(d, dict) or d.get("status") != "ready":
            return _j({"status": "still computing", "message": (d or {}).get("message")})
        return _j({"validation": {k: d["validation"].get(k) for k in ("verdict_level", "verdict", "notes", "base_rate", "top5_hit_rate", "lift_top5", "top5_avg_return", "all_avg_return", "top5_crash_rate", "all_crash_rate", "skill_beyond_volatility")},
                   "top_signals": [f["label"] for f in d["feature_importance"][:6]],
                   "picks": [{k: p.get(k) for k in ("symbol", "name", "sector", "price", "probability", "hist_hit_rate", "drivers", "red_flags")} for p in d["picks"][:10]],
                   "warning": d.get("warning")})
    d = clients.research.get(f"/api/ai/picks/{m}", params={"horizon": horizon}, timeout=60)
    if not isinstance(d, dict):
        return _j(d)
    if d.get("status") == "running":
        return _j({"status": "still computing", "message": d.get("message"), "advice": "Tell the user the analysis is being prepared and to check the Outhit AI page in a minute."})
    keep = {"long_term": ("evidence", "method", "picks"), "short_term": ("evidence", "has_edge", "bullish", "bearish"),
            "intraday": ("verdict_level", "verdict", "chosen", "strategies", "today", "warning"), "fo": ("direction_has_edge", "ideas", "warning")}
    slim = {k: d.get(k) for k in keep.get(horizon, ())}
    for k in ("picks", "bullish", "bearish"):
        if slim.get(k):
            slim[k] = [{x: p.get(x) for x in ("symbol", "name", "sector", "price", "score", "reasons", "red_flags", "levels", "hist_prob_up")} for p in slim[k][:8]]
    if slim.get("ideas"):
        slim["ideas"] = [{"underlying": i["underlying"], "direction": i["direction"], "strategy": {x: (i.get("strategy") or {}).get(x) for x in ("name", "legs", "max_profit", "max_loss", "prob_profit", "ev_per_risk", "reason")}} for i in slim["ideas"]]
    return _j(slim)


WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}

FUNCTION_TOOLS = [analyze_asset, market_overview, search_symbol, project_savings, stock_picks, sector_strength, global_markets,
                  company_snapshot, predict_outperformers, news_search, ai_outlook, outhit_ai_picks]
TOOLS = [*FUNCTION_TOOLS, WEB_SEARCH]  # Claude also gets Anthropic's server-side web search
TOOL_BY_NAME = {t.name: t for t in FUNCTION_TOOLS}
# Smaller tool set for Groq (each tool schema costs tokens on every round).
GROQ_TOOLS = [outhit_ai_picks, ai_outlook, company_snapshot, news_search, global_markets, predict_outperformers, sector_strength, search_symbol, project_savings]
GROQ_MAX_ROUNDS = 4
# Groq's free tier allows 8,000 tokens per minute, and every round re-sends the whole
# conversation, so tool output is compacted hard before the model sees it.
GROQ_TOOL_RESULT_CHARS = 1800
DROP_KEYS = {"url", "urls", "sparkline", "about", "history", "score_history", "intraday", "cone", "deciles", "all", "caveats",
             "method", "longBusinessSummary", "website", "factor_scores", "feature_importance", "trained_at", "as_of", "currency"}
LIST_LIMIT = 8

GROQ_SYSTEM_PROMPT = """You are Outhit's financial advisor for everyday investors (data on 49 countries).
- Use the tools for any market, stock or news question and quote their numbers exactly as given (e.g. "change_52w_pct" is the 1-year change, not year-to-date). Never invent data, probabilities or price targets the tools did not return. If a tool fails, say so.
- Recommendations (long-term, short-term, intraday, F&O): outhit_ai_picks first (it combines every tested method); "which will go up/down": predict_outperformers (state its validation verdict first; if "none"/"negative", say the model hasn't worked there); one company: ai_outlook (news, reputation, outlook) and company_snapshot (fundamentals); news: news_search; world markets: global_markets.
- Never promise profit or certainty; give probabilities and ranges. For intraday, warn that most retail intraday traders lose money and insist on stop-losses.
- Foundations first: emergency fund, high-interest debt, insurance. Prefer diversified index funds as the core; stocks and crypto as small satellites.
- Be concise: answer first, then key numbers and reasons, in short bullets. You are an educational tool, not a licensed adviser."""
MAX_PAUSE_RESTARTS = 3


class AdvisorUnavailable(Exception):
    pass


def _profile_context(profile: dict | None) -> str:
    if not profile:
        return "The user has not filled in the Outhit planner yet. If their question depends on their situation, ask briefly for age, country, horizon and risk appetite, or suggest the Planner page."
    try:
        inp = AdvisorInput(**profile)
    except Exception:
        return "The user's saved profile could not be read."
    health = planner.financial_health(inp)
    risk = planner.risk_profile(inp)
    currency = "INR" if inp.country == "IN" else "USD"
    return (
        "The user's saved profile (amounts in " + currency + "):\n"
        + json.dumps(inp.model_dump())
        + "\nDerived: risk profile = " + risk["profile"]
        + f", financial health score = {health['score']}/100 ({health['grade']})"
        + ", open action items = " + "; ".join(a["title"] for a in health["actions"] if a["priority"] != "good")
    )


def chat(messages: list[dict], profile: dict | None) -> str:
    """Answer with Claude when an Anthropic key is configured, otherwise with Groq."""
    settings = get_settings()
    if settings.anthropic_api_key:
        return _chat_claude(messages, profile)
    if settings.groq_api_key:
        return _chat_groq(messages, profile)
    raise AdvisorUnavailable("The AI advisor is not configured. Set ADVISOR_GROQ_API_KEY (or ADVISOR_ANTHROPIC_API_KEY) in outhit-services/.env.")


def _compact(v, depth: int = 0):
    """Shrink tool output for small-context models: drop bulky fields, round numbers, cap lists."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in DROP_KEYS or x is None:
                continue
            if k in ("debtToEquity", "debt_to_equity") and isinstance(x, (int, float)):
                # Yahoo reports debt/equity in percent (76 = 0.76x); models misread it as 76x.
                out["debt_to_equity_ratio"] = round(x / 100, 2)
                continue
            out[k] = _compact(x, depth + 1)
        return out
    if isinstance(v, list):
        return [_compact(x, depth + 1) for x in v[:LIST_LIMIT]]
    if isinstance(v, float):
        return round(v, 4) if abs(v) < 10 else round(v, 2)
    if isinstance(v, str) and len(v) > 220:
        return v[:220] + "…"
    return v


def _run_tool(name: str, raw_args: str) -> str:
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        args = json.loads(raw_args or "{}")
        out = tool.call(args)
    except Exception as exc:  # bad arguments or a failing data source: tell the model, don't crash the chat
        log.info("tool %s failed: %s", name, exc)
        return json.dumps({"error": f"{name} failed: {exc}"})
    text = out if isinstance(out, str) else json.dumps(out, default=str)
    try:
        text = json.dumps(_compact(json.loads(text)), separators=(",", ":"), default=str)
    except ValueError:
        pass
    if len(text) > GROQ_TOOL_RESULT_CHARS:
        text = text[:GROQ_TOOL_RESULT_CHARS] + '..." [truncated]'
    return text


def _chat_groq(messages: list[dict], profile: dict | None) -> str:
    settings = get_settings()
    # Retries honour Groq's retry-after header, so hitting the per-minute token limit means waiting, not failing.
    client = groq.Groq(api_key=settings.groq_api_key, max_retries=5, timeout=120)
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description.split("\n")[0].split(". ")[0], "parameters": t.input_schema}} for t in GROQ_TOOLS]
    history: list[dict] = [{"role": "system", "content": GROQ_SYSTEM_PROMPT + "\n\n" + _profile_context(profile)}, *messages[-8:]]
    try:
        for _ in range(GROQ_MAX_ROUNDS):
            resp = client.chat.completions.create(
                model=settings.groq_model,
                messages=history,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=4096,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip() or "Sorry, I couldn't produce an answer. Please try rephrasing."
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in msg.tool_calls],
            })
            for c in msg.tool_calls:
                history.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": _run_tool(c.function.name, c.function.arguments)})
        return _groq_final_answer(client, settings.groq_model, history)
    except groq.AuthenticationError as exc:
        raise AdvisorUnavailable("The Groq API key was rejected. Check ADVISOR_GROQ_API_KEY.") from exc
    except groq.RateLimitError as exc:
        raise AdvisorUnavailable("The AI advisor hit Groq's rate limit. Please wait a minute and try again.") from exc
    except groq.APIConnectionError as exc:
        raise AdvisorUnavailable("Couldn't reach Groq. Please try again.") from exc
    except groq.BadRequestError as exc:
        log.warning("Groq rejected the request: %s", exc)
        raise AdvisorUnavailable("The AI advisor couldn't process that request. Please try rephrasing.") from exc
    except groq.APIStatusError as exc:
        if exc.status_code == 413:  # single request bigger than the per-minute token allowance
            raise AdvisorUnavailable("That question needed more data than the free Groq tier allows per minute. Try a narrower question (one company or one market).") from exc
        raise AdvisorUnavailable("The AI service returned an error. Please try again.") from exc


def _groq_final_answer(client: "groq.Groq", model: str, history: list[dict]) -> str:
    """Out of tool rounds: restate everything gathered as plain text in a fresh, tool-free
    conversation, so the model can only write the answer (sending the tool transcript
    without tools makes some models try to call a tool anyway, which Groq rejects)."""
    system = history[0]["content"]
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    gathered = "\n\n".join(
        f"[{m.get('name') or 'tool result'}] {m['content']}" for m in history if m["role"] == "tool"
    )[-12000:]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system + "\n\nYou cannot call tools now. Answer using only the data below."},
            {"role": "user", "content": f"Data gathered from Outhit's tools:\n{gathered}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_completion_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip() or "Sorry, I couldn't produce an answer. Please try again."


def _chat_claude(messages: list[dict], profile: dict | None) -> str:
    settings = get_settings()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    history = list(messages)
    final = None
    try:
        # Web search is a server tool: a long search turn can pause (stop_reason "pause_turn"),
        # which ends the runner. Mirror the history and restart so the answer isn't cut short.
        for _ in range(MAX_PAUSE_RESTARTS + 1):
            runner = client.beta.messages.tool_runner(
                model=settings.anthropic_model,
                max_tokens=16000,
                output_config={"effort": "medium"},
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": _profile_context(profile)},
                ],
                tools=TOOLS,
                messages=history,
                max_iterations=10,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            for message in runner:
                final = message
                history.append({"role": "assistant", "content": message.content})
                if (tool_response := runner.generate_tool_call_response()) is not None:
                    history.append(tool_response)
            if final is None or final.stop_reason != "pause_turn":
                break
    except anthropic.AuthenticationError as exc:
        raise AdvisorUnavailable("The Anthropic API key was rejected. Check ADVISOR_ANTHROPIC_API_KEY.") from exc
    except anthropic.RateLimitError as exc:
        raise AdvisorUnavailable("The AI advisor is busy right now. Please try again in a minute.") from exc
    except anthropic.APIConnectionError as exc:
        raise AdvisorUnavailable("Couldn't reach the AI service. Please try again.") from exc

    if final is None:
        return "Sorry, I couldn't produce an answer. Please try again."
    if final.stop_reason == "refusal":
        return "I can't help with that request. I'm happy to help with budgeting, saving, investing or analysing a specific asset."
    text = "".join(b.text for b in final.content if b.type == "text").strip()
    return text or "Sorry, I couldn't produce an answer. Please try rephrasing."
