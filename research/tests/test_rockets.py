import numpy as np
import pandas as pd

from research import rockets


def _panel(n=60, days=800, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-08", periods=days)
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (days, n)), axis=0)), index=idx, columns=[f"S{i}" for i in range(n)])
    return closes


def test_rocket_target_is_top_decile():
    closes = _panel()
    dates = closes.index[252:500:5]
    label, fwd = rockets._targets(closes, dates, "rocket")
    share = label.mean(axis=1).dropna()
    assert share.between(0.08, 0.13).all()  # ~10% of stocks each date


def test_grower_requires_shallow_drawdown():
    closes = _panel()
    idx = closes.index
    # Two stocks with the same 12-month gain: one smooth, one that crashes 50% first.
    smooth = 100 * np.exp(np.linspace(0, np.log(30), len(idx)))  # ~3x a year: clearly a top-20% stock
    bumpy = smooth.copy()
    mid = 400
    bumpy[300:mid] = np.linspace(bumpy[300], bumpy[300] * 0.5, mid - 300)
    bumpy[mid:552] = np.linspace(bumpy[mid - 1], smooth[551], 552 - mid)
    closes["SMOOTH"], closes["BUMPY"] = smooth, bumpy
    label, _ = rockets._targets(closes, [idx[300]], "grower")
    assert label.loc[idx[300], "SMOOTH"] == 1.0
    assert label.loc[idx[300], "BUMPY"] == 0.0


def test_features_cover_research_signals(monkeypatch):
    closes = _panel()
    vols = pd.DataFrame(1e6, index=closes.index, columns=closes.columns)
    monkeypatch.setattr(rockets.fundamentals_history, "load", lambda syms, progress=None: {})
    monkeypatch.setattr(rockets.earnings, "surprise_panel", lambda syms, c, progress=None: None)
    monkeypatch.setattr(rockets, "_surprise_streak", lambda syms, c: None)
    sectors = pd.Series({c: ("Tech" if i % 2 else "Bank") for i, c in enumerate(closes.columns)})
    f = rockets.build_features(closes, vols, sectors)
    for k in ("new_highs_63", "vol_contraction", "updown_volume_63", "sector_mom_126", "sector_rel_126", "illiquidity", "skew_63"):
        assert k in f and f[k].iloc[-1].notna().any()
    assert set(f) <= set(rockets.FEATURE_LABELS)
