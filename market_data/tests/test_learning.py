from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_data.db import Base, SessionLocal, engine
from market_data.models import PredictionRecord
from market_data.services import learning, predictor
from market_data.tests.test_predictor import _market

Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def _empty_ledger():
    with SessionLocal() as db:
        db.query(PredictionRecord).delete()
        db.commit()


def _closes(days=40, n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, n)), axis=0)), index=idx, columns=[f"S{i}" for i in range(n)])


def _publish(closes, day, horizon=5, good=True):
    """Publish a prediction whose probabilities do (good) or don't track the real future return."""
    i = closes.index.get_loc(day)
    fwd = closes.iloc[i + horizon] / closes.iloc[i] - 1 if i + horizon < len(closes) else pd.Series(0.0, index=closes.columns)
    score = fwd.rank(pct=True) if good else (-fwd).rank(pct=True)
    rows = [{"symbol": s, "prob": float(score[s]), "decile": int(np.ceil(score[s] * 10)), "variant": "blend",
             "variants": {"blend": float(score[s]), "trees": float(1 - score[s])}, "price": float(closes.at[day, s])} for s in closes.columns]
    return learning.record("us", horizon, day.date(), rows)


def test_predictions_resolve_only_after_the_horizon_passes():
    closes = _closes()
    _publish(closes, closes.index[10])
    _publish(closes, closes.index[-2])  # 5 trading days haven't passed yet
    assert learning.resolve("us", 5, closes) == 30
    rec = learning.track_record("us", 5)
    assert rec["logged"] == 60 and rec["pending"] == 30 and rec["resolved"] == 30
    with SessionLocal() as db:
        r = db.query(PredictionRecord).filter_by(symbol="S3", as_of=closes.index[10].date()).one()
    assert r.fwd_ret == pytest.approx(closes.iat[15, 3] / closes.iat[10, 3] - 1)
    assert learning.resolve("us", 5, closes) == 0  # idempotent


def test_republishing_the_same_day_keeps_the_first_call():
    closes = _closes()
    assert _publish(closes, closes.index[10]) == 30
    assert _publish(closes, closes.index[10], good=False) == 0


def test_track_record_and_misses_reflect_outcomes():
    closes = _closes(days=60)
    for i in range(5, 50, 2):
        _publish(closes, closes.index[i])
    learning.resolve("us", 5, closes)
    rec = learning.track_record("us", 5, names={"S1": "Stock one"})
    assert rec["ic_mean"] > 0.9  # these probabilities were built from the real outcome
    assert rec["top_avg_return"] > rec["bottom_avg_return"]
    assert rec["live_verdict"] == "working"
    assert all(m["call"] in ("bullish", "bearish") for m in rec["biggest_misses"])


def test_live_outcomes_steer_the_variant_choice():
    closes = _closes(days=80)
    for i in range(5, 70):
        _publish(closes, closes.index[i])  # "blend" tracked reality; "trees" was its mirror image
    learning.resolve("us", 5, closes)
    live = learning.live_variant_ic("us", 5)
    assert live["blend"].mean() > 0.9 and live["trees"].mean() < -0.9
    # The backtest slightly prefers trees, but scored live predictions overrule it.
    chosen, board = learning.choose({"blend": [0.02] * 8, "trees": [0.03] * 8}, live, default="blend")
    assert chosen == "blend" and board["blend"]["live_dates"] > 40
    chosen, _ = learning.choose({"blend": [0.02] * 8, "trees": [0.03] * 8}, None, default="blend")
    assert chosen == "trees"


def test_choice_needs_a_clear_margin_and_weights_recent_folds():
    assert learning.choose({"blend": [0.020] * 8, "trees": [0.022] * 8})[0] == "blend"  # within noise: keep default
    # Trees were great long ago but lately the linear model is better.
    chosen, board = learning.choose({"blend": [0.02] * 8, "trees": [0.08] * 4 + [0.0] * 4, "linear": [0.0] * 4 + [0.06] * 4})
    assert chosen == "linear" and board["linear"]["recent_ic"] > board["trees"]["recent_ic"]


def test_walk_forward_adapts_without_peeking():
    closes, volumes, index = _market(signal=1.0)
    feats = predictor.features(closes, volumes, index)
    table = predictor.build_table(closes, feats, 21)
    cols = [c for c in predictor.FEATURE_LABELS if c in table.columns]
    wf = predictor.walk_forward(table, closes.index, cols, 21)
    assert wf.choices[0]["variant"] == predictor.DEFAULT_VARIANT  # nothing learned yet in the first window
    assert set(wf.fold_ic) == set(predictor.VARIANTS)
    assert all(len(v) == len(wf.choices) for v in wf.fold_ic.values())
    for c in wf.choices:  # each window's prob is exactly the chosen variant's
        part = wf.oos[(wf.oos["date"] >= pd.Timestamp(c["from"])) & (wf.oos["date"] <= pd.Timestamp(c["to"]))]
        assert (part["prob"] == part[f"v_{c['variant']}"]).all()


def test_recency_weights_halve_each_half_life():
    d = pd.Series(pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]))
    w = predictor.recency_weights(d, 252)
    assert w[-1] == 1 and w[1] == pytest.approx(0.5, abs=0.01) and w[0] == pytest.approx(0.25, abs=0.01)
    assert predictor.recency_weights(d, None) is None


def test_asset_universe_spans_every_class():
    meta = predictor._asset_meta()
    assert len(meta) >= predictor.MIN_STOCKS
    assert {"Bonds", "Gold & commodities", "Crypto", "Real estate", "US stocks"} <= set(meta["sector"])
    assert predictor.market_name("assets") == "All asset classes"
