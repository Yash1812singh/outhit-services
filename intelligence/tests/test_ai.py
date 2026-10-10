from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from intelligence import events, ledger, live, news, outlook, reputation
from intelligence.db import Company, NewsItem, Prediction, SessionLocal, now
from intelligence.main import app
from intelligence.sentiment import model as sentiment
from intelligence.tests.conftest import INTERNAL

client = TestClient(app)


# ---------------- sentiment & events ----------------

def test_sentiment_direction_on_clear_headlines():
    bull, bear, flat = sentiment.score([
        "Company shares surge 12% after record quarterly profit beats estimates",
        "Stock plunges to 52-week low as company cuts full-year guidance",
        "Company to announce quarterly results on Thursday",
    ])
    assert bull["sentiment"] > 0.3 and bull["label"] == "bullish"
    assert bear["sentiment"] < -0.3 and bear["label"] == "bearish"
    assert abs(flat["sentiment"]) < 0.5


def test_explanations_are_whole_words():
    terms = sentiment.explain("Shares plunge after company cuts guidance")
    assert terms and all(len(t["term"]) >= 3 for t in terms)
    assert any(t["push"] == "bearish" for t in terms)


@pytest.mark.parametrize("title,key", [
    ("SEBI issues show-cause notice to XYZ over disclosure lapses", "regulator"),
    ("Auditor resigns citing concerns over accounts", "governance"),
    ("XYZ beats estimates as profit jumps 30%", "beat"),
    ("Brokerage downgrades XYZ, cuts target price", "downgrade"),
    ("XYZ bags Rs 2,000 crore order from railways", "order_win"),
    ("XYZ defaults on bond payment", "default"),
])
def test_event_tags(title, key):
    assert key in {e["key"] for e in events.tag(title)}


def test_no_false_controversy_on_plain_news():
    assert not [e for e in events.tag("XYZ shares rise ahead of quarterly results") if e["controversy"]]


# ---------------- news parsing ----------------

RSS = """<rss><channel>
<item><title>Infosys shares rise 3% - Economic Times</title><link>https://x/1</link><pubDate>Thu, 08 Oct 2026 10:00:00 GMT</pubDate><source url="https://et">Economic Times</source></item>
<item><title>Unrelated market wrap - Mint</title><link>https://x/2</link><pubDate>Thu, 08 Oct 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_rss_parse_and_relevance():
    items = news._parse_rss(RSS)
    assert len(items) == 2 and items[0]["source"] == "Economic Times"
    c = Company(symbol="INFY.NS", name="Infosys Limited", market="in")
    assert news._relevant(items[0]["title"], c) and not news._relevant(items[1]["title"], c)


def test_short_name():
    assert news.short_name("Reliance Industries Limited") == "Reliance"
    assert news.short_name("Apple Inc.") == "Apple"


# ---------------- live quotes ----------------

def test_quote_from_minute_bars():
    idx = pd.date_range("2026-10-08 09:15", periods=5, freq="1min", tz="Asia/Kolkata").append(
        pd.date_range("2026-10-09 09:15", periods=5, freq="1min", tz="Asia/Kolkata"))
    close = np.array([100, 100, 100, 100, 100, 101, 102, 103, 104, 105], dtype=float)
    f = pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1000.0}, index=idx)
    q = live._quote("X", f)
    assert q["prev_close"] == 100 and q["price"] == 105
    assert q["change_pct"] == pytest.approx(5.0)
    assert 101 < q["vwap"] < 105
    assert len(q["intraday"]) == 5


# ---------------- outlook, reputation, ledger ----------------

def _company_with_news(titles: list[str], days_ago: float = 1):
    with SessionLocal() as db:
        db.add(Company(symbol="TEST", name="Test Corp", market="us", pinned=True, governance={"overallRisk": 2, "fetched_at": now().isoformat()}))
        for i, (t, s) in enumerate(zip(titles, sentiment.score(titles))):
            db.add(NewsItem(symbol="TEST", key=f"k{i}", title=t, url=f"u{i}", published_at=now() - timedelta(days=days_ago),
                            sentiment=s["sentiment"], label=s["label"], confidence=s["confidence"], events=events.tag(t)))
        db.commit()


@pytest.fixture
def no_backend(monkeypatch):
    from intelligence import backend
    monkeypatch.setattr(backend, "quant", lambda market, horizon=21: None)
    monkeypatch.setattr(backend, "technical", lambda symbol: None)


def test_good_news_makes_outlook_bullish(no_backend):
    _company_with_news([f"Test Corp shares surge after record profit beats estimates {i}" for i in range(8)])
    o = outlook.compute("TEST")
    assert o["score"] > 10 and "bullish" in o["label"].lower()
    assert {c["key"] for c in o["components"]} >= {"news", "events", "reputation"}


def test_scandal_hurts_reputation_and_outlook(no_backend):
    _company_with_news(["SEC charges Test Corp with accounting fraud", "Test Corp CFO resigns amid probe", "Test Corp shares plunge 30%"])
    rep = reputation.score("TEST")
    assert rep["controversies"] and rep["score"] < 60
    assert outlook.compute("TEST")["score"] < -10


def test_missing_signals_pull_toward_neutral(no_backend):
    _company_with_news([])
    o = outlook.compute("TEST")
    assert abs(o["score"]) < 10 and o["confidence"] == "Low"


def test_ledger_records_once_per_day_and_scores_outcomes():
    o = {"symbol": "TEST", "market": "us", "price": 100.0, "score": 40.0, "label": "Bullish", "components": [{"key": "news", "value": 0.5}]}
    assert ledger.record(o) and not ledger.record(o)
    with SessionLocal() as db:
        p = db.query(Prediction).one()
        p.outcome = {"1d": 0.02, "5d": 0.05, "21d": -0.01}
        db.add(Prediction(symbol="B", market="us", day="2026-01-01", price=10, score=-30, label="Bearish", components={}, outcome={"1d": -0.01}))
        db.commit()
    tr = ledger.track_record()
    assert tr["predictions_logged"] == 2
    assert tr["horizons"]["1d"]["bullish_hit_rate"] == 1.0 and tr["horizons"]["1d"]["bearish_hit_rate"] == 1.0
    assert tr["horizons"]["1d"]["directional_accuracy"] == 1.0


# ---------------- API ----------------

def test_sentiment_endpoint_and_model_card():
    r = client.post("/api/ai/sentiment", json={"text": "Regulator raids company offices in fraud probe; shares tank"})
    body = r.json()
    assert r.status_code == 200 and body["label"] == "bearish"
    assert any(e["controversy"] for e in body["events"])
    card = client.get("/api/ai/model").json()
    assert card["sentiment"]["accuracy"] > 0.7
    assert card["outlook_weights"]["news"] == 0.25


def test_session_split_handles_mixed_timezone_batches():
    # A US session expressed in IST crosses midnight: 19:00 -> 01:30 next day.
    y = pd.date_range("2026-10-08 19:00", "2026-10-09 01:29", freq="1min", tz="Asia/Kolkata")
    t = pd.date_range("2026-10-09 19:00", "2026-10-09 19:30", freq="1min", tz="Asia/Kolkata")
    close = np.r_[np.full(len(y), 100.0), np.full(len(t), 98.0)]
    idx = y.append(t)
    f = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 10.0}, index=idx)
    q = live._quote("AAPL", f)
    assert q["prev_close"] == 100 and q["price"] == 98
    assert q["intraday"][0][0].startswith("2026-10-09T19:00")
