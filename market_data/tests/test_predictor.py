import numpy as np
import pandas as pd

from market_data.services import predictor


def _market(n_stocks=60, days=1100, signal=0.0, seed=1):
    """Synthetic market. With signal > 0, a hidden per-stock trait drives both the
    past 3-month return and the future return, so there is something learnable."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-08", periods=days)
    trait = rng.normal(size=n_stocks)
    regime = np.sign(np.sin(np.arange(days) / 90))  # traits switch on/off for months at a time
    drift = signal * 0.002 * np.outer((regime > 0).astype(float), trait)
    rets = rng.normal(0, 0.015, (days, n_stocks)) + drift
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"S{i}" for i in range(n_stocks)])
    volumes = pd.DataFrame(rng.integers(1e5, 1e6, (days, n_stocks)), index=idx, columns=closes.columns).astype(float)
    index = closes.mean(axis=1)
    return closes, volumes, index


def _run(closes, volumes, index, horizon=21):
    feats = predictor.features(closes, volumes, index)
    table = predictor.build_table(closes, feats, horizon)
    cols = [c for c in predictor.FEATURE_LABELS if c in table.columns]
    wf = predictor.walk_forward(table, closes.index, cols, horizon)
    return predictor.evaluate(wf.oos), table


def test_features_have_no_lookahead():
    closes, volumes, index = _market()
    full = predictor.features(closes, volumes, index)
    cut = predictor.features(closes.iloc[:-50], volumes.iloc[:-50], index.iloc[:-50])
    d = closes.index[-51]
    for name in ("ret_63", "rsi_14", "beta_126", "dist_high_252", "breadth"):
        pd.testing.assert_series_equal(full[name].loc[d], cut[name].loc[d], check_names=False)


def test_label_is_balanced_and_last_date_is_unlabeled():
    closes, volumes, index = _market()
    table = predictor.build_table(closes, predictor.features(closes, volumes, index), 21)
    labeled = table[table["outperform"].notna()]
    assert 0.45 < labeled["outperform"].mean() < 0.55
    assert table[table["date"] == closes.index[-1]]["outperform"].isna().all()


def test_finds_a_planted_signal():
    v, _ = _run(*_market(signal=1.0))
    assert v["auc"] > 0.53
    assert v["top_minus_bottom_return"] > 0


def test_reports_no_edge_on_pure_noise():
    v, _ = _run(*_market(signal=0.0, seed=7))
    assert 0.45 < v["auc"] < 0.55
    assert v["verdict_level"] in ("none", "negative", "modest")
    assert len(v["deciles"]) == 10


def _bars(closes, seed=3):
    rng = np.random.default_rng(seed)
    opens = closes.shift(1) * np.exp(rng.normal(0, 0.004, closes.shape))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, closes.shape)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, closes.shape)))
    return {"open": opens, "high": highs, "low": lows}


def test_short_term_features_have_no_lookahead():
    closes, volumes, index = _market()
    bars = _bars(closes)
    full = predictor.features(closes, volumes, index, bars, short_term=True)
    cut = predictor.features(closes.iloc[:-50], volumes.iloc[:-50], index.iloc[:-50], {k: v.iloc[:-50] for k, v in bars.items()}, short_term=True)
    d = closes.index[-51]
    for name in ("ret_1", "gap_1", "intraday_1", "clv_1", "range_ratio", "volume_shock", "mkt_ret_1"):
        pd.testing.assert_series_equal(full[name].loc[d], cut[name].loc[d], check_names=False)
    assert "gap_1" not in predictor.features(closes, volumes, index)  # long-horizon models are unchanged


def test_next_day_table_is_daily_and_tracks_tradable_return():
    closes, volumes, index = _market(days=700)
    bars = _bars(closes)
    table = predictor.build_table(closes, predictor.features(closes, volumes, index, bars, short_term=True), 1, bars["open"])
    dates = table["date"].drop_duplicates()
    assert (dates.diff().dropna() <= pd.Timedelta(days=4)).all()  # one row per trading day, not per week
    d0, d1 = closes.index[300], closes.index[301]
    row = table[(table["date"] == d0) & (table["symbol"] == "S0")].iloc[0]
    assert row["fwd_ret_open"] == closes.at[d1, "S0"] / bars["open"].at[d1, "S0"] - 1
    assert table[table["date"] == closes.index[-1]]["outperform"].isna().all()


def test_next_day_validation_reports_from_open_returns():
    closes, volumes, index = _market(days=900, signal=1.0)
    bars = _bars(closes)
    feats = predictor.features(closes, volumes, index, bars, short_term=True)
    table = predictor.build_table(closes, feats, 1, bars["open"])
    cols = [c for c in predictor.FEATURE_LABELS if c in table.columns]
    v = predictor.evaluate(predictor.walk_forward(table, closes.index, cols, 1).oos)
    assert v["avg_return_from_open"] is not None and v["top_minus_bottom_return_from_open"] is not None
    assert all("avg_return_from_open" in d for d in v["deciles"])
