import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from advisor import ai_advisor, clients
from advisor.main import app
from advisor.tests.conftest import INTERNAL, FakeServiceClient, user_headers

client = TestClient(app)

PLAN_INPUT = {
    "age": 28, "country": "US", "monthly_income": 6000, "monthly_expenses": 3000, "current_savings": 20000,
    "investment_horizon_years": 20, "lump_sum": 10000, "monthly_investment": 1000, "include_crypto": True,
}


def test_health_reports_no_ai_provider_without_keys():
    assert client.get("/health").json() == {"service": "advisor", "status": "ok", "ai_provider": None}


def test_plan_is_saved_for_signed_in_users(fake_market):
    headers = user_headers(101)
    r = client.post("/api/advisor/plan", json=PLAN_INPUT, headers=headers)
    assert r.status_code == 200
    assert r.json()["currency"] == "USD"
    assert client.get("/api/advisor/profile", headers=headers).json()["age"] == 28
    # Another user sees nothing.
    assert client.get("/api/advisor/profile", headers=user_headers(102)).json() is None


def test_plan_works_anonymously_and_validates(fake_market):
    assert client.post("/api/advisor/plan", json=PLAN_INPUT).status_code == 200
    assert client.post("/api/advisor/plan", json=PLAN_INPUT, headers=INTERNAL).status_code == 200
    assert client.post("/api/advisor/plan", json={**PLAN_INPUT, "age": 5}).status_code == 422


def test_user_id_without_internal_token_is_rejected(fake_market):
    """A client can't claim to be a user by sending X-User-Id directly to the service."""
    assert client.post("/api/advisor/plan", json=PLAN_INPUT, headers={"X-User-Id": "101"}).status_code == 403
    assert client.get("/api/advisor/profile", headers={"X-User-Id": "101"}).status_code == 403
    assert client.get("/api/advisor/profile", headers={"X-Internal-Token": "guess", "X-User-Id": "101"}).status_code == 403


def test_profile_requires_login():
    assert client.get("/api/advisor/profile").status_code == 403
    assert client.get("/api/advisor/profile", headers=INTERNAL).status_code == 401


def test_chat_requires_login_and_reports_missing_key():
    body = {"messages": [{"role": "user", "content": "Should I buy BTC?"}]}
    assert client.post("/api/advisor/chat", json=body).status_code == 403
    assert client.post("/api/advisor/chat", json=body, headers=INTERNAL).status_code == 401
    r = client.post("/api/advisor/chat", json=body, headers=user_headers(7))
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_chat_last_message_must_be_from_user():
    body = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    assert client.post("/api/advisor/chat", json=body, headers=user_headers(7)).status_code == 422


def test_stateless_endpoints():
    assert client.post("/api/advisor/health", json=PLAN_INPUT).json()["score"] >= 0
    assert 0 <= client.post("/api/advisor/risk-profile", json=PLAN_INPUT).json()["level"] <= 4
    out = client.post("/api/advisor/project", json={"initial": 1000, "monthly": 100, "years": 5}).json()
    assert out["pessimistic_final"] < out["median_final"] < out["optimistic_final"]


def test_groq_loop_runs_tools_then_answers(monkeypatch):
    """Simulate Groq: one round of tool calls, then a plain answer."""
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                tc = NS(id="t1", function=NS(name="news_search", arguments='{"query": "Infosys"}'))
                return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])
            return NS(choices=[NS(message=NS(content="Here is the answer.", tool_calls=None))])

    monkeypatch.setattr(ai_advisor.groq, "Groq", lambda **kw: NS(chat=NS(completions=FakeCompletions())))
    monkeypatch.setattr(ai_advisor, "_run_tool", lambda name, args: '{"results": []}')
    monkeypatch.setattr(ai_advisor, "get_settings", lambda: NS(anthropic_api_key=None, groq_api_key="x", groq_model="m"))
    assert ai_advisor.chat([{"role": "user", "content": "News on Infosys?"}], None) == "Here is the answer."
    assert calls[1]["messages"][-1]["role"] == "tool" and calls[1]["messages"][-1]["name"] == "news_search"


def test_compact_drops_bulky_fields_and_caps_lists():
    out = ai_advisor._compact({"url": "x", "picks": list(range(20)), "price": 1.234567, "about": "long", "name": "A", "debtToEquity": 76.0})
    assert out == {"picks": list(range(8)), "price": 1.2346, "name": "A", "debt_to_equity_ratio": 0.76}


# ---------------- tools are HTTP calls to the other services ----------------

def test_analyze_asset_calls_market_data_and_trims_history(monkeypatch):
    fake = FakeServiceClient({"/api/market/asset/AAPL": {"symbol": "AAPL", "score": 42, "history": [1, 2], "forecast": {"cone": [1], "horizons": {}}}})
    monkeypatch.setattr(clients, "market", fake)
    out = json.loads(ai_advisor._run_tool("analyze_asset", '{"symbol": "AAPL"}'))
    assert out["symbol"] == "AAPL" and out["score"] == 42
    assert "history" not in out and "cone" not in out.get("forecast", {})
    assert fake.calls[0][0] == "/api/market/asset/AAPL"


def test_tool_reports_unavailable_service(monkeypatch):
    monkeypatch.setattr(clients, "market", FakeServiceClient())
    out = json.loads(ai_advisor._run_tool("search_symbol", '{"query": "Tata"}'))
    assert "unavailable" in out["error"]


def test_rockets_and_picks_tools_call_research(monkeypatch):
    fake = FakeServiceClient({
        "/api/ai/rockets/in": {"status": "running", "message": "training"},
        "/api/ai/picks/in": {"evidence": {"ic": 0.05}, "method": "ensemble", "picks": [{"symbol": "A.NS", "score": 80, "about": "x"}]},
    })
    monkeypatch.setattr(clients, "research", fake)
    rockets = json.loads(ai_advisor.outhit_ai_picks.call({"market": "IN", "horizon": "rockets"}))
    assert rockets == {"status": "still computing", "message": "training"}
    picks = json.loads(ai_advisor.outhit_ai_picks.call({"market": "in", "horizon": "long_term"}))
    assert picks["picks"][0]["symbol"] == "A.NS"
    assert fake.calls == [("/api/ai/rockets/in", {"kind": "rocket"}), ("/api/ai/picks/in", {"horizon": "long_term"})]


def test_ai_outlook_tool_calls_intelligence(monkeypatch):
    fake = FakeServiceClient({"/api/ai/outlook/INFY.NS": {
        "outlook": {"score": 12, "label": "Mildly bullish", "components": [{"label": "News", "contribution": 5, "detail": "d"}]},
        "reputation": {"score": 70}, "news": [], "quote": {"price": 1500},
    }})
    monkeypatch.setattr(clients, "intelligence", fake)
    out = json.loads(ai_advisor.ai_outlook.call({"symbol": "INFY.NS"}))
    assert out["outlook"]["score"] == 12 and out["why"][0]["signal"] == "News"
    assert out["quote"]["price"] == 1500
