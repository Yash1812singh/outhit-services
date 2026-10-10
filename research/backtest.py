"""Backtest engine: replays a strategy through history, trading the way you would have.

- Signals use only data up to the signal day's close; trades execute at the NEXT day's close.
- Every trade pays costs (per side, on the value traded).
- Results are compared with the market index and, for stock strategies, with an equal-weighted basket
  of the same 300 companies. That second comparison removes most of the survivorship bias of testing
  on today's largest companies: what's left is the strategy's own edge.
- A t-test on monthly excess returns says whether the result could plausibly be luck.
"""

import hashlib
import json
import logging
import math

import numpy as np
import pandas as pd

from research import gurus, jobs, strategies
from research.strategies import BENCHMARK, CURRENCY, DEFAULT_COST_BPS, RISK_FREE, Context, Strategy

log = logging.getLogger(__name__)

TTL = 6 * 3600
PERIOD_CODE = {"W": "W-FRI", "M": "M", "Q": "Q", "A": "Y"}
WARMUP_DAYS = 260


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def signal_dates(index: pd.DatetimeIndex, start: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    """The start date, then the last trading day of each week / month / quarter / year (excluding the final day)."""
    idx = index[index >= start]
    if len(idx) < 2:
        return []
    last = idx.to_series().groupby(idx.to_period(PERIOD_CODE[freq])).max()
    return [idx[0]] + [d for d in last if idx[0] < d < idx[-1]]


def simulate(closes: pd.DataFrame, targets: dict[pd.Timestamp, pd.Series], start: pd.Timestamp, cost_bps: float):
    """Daily equity curve (starting at 1) and the executed rebalances."""
    idx = closes.index[closes.index >= start]
    cols = {s: i for i, s in enumerate(closes.columns)}
    px = closes.loc[idx].to_numpy(dtype=float)
    rets = np.zeros_like(px)
    with np.errstate(all="ignore"):
        rets[1:] = px[1:] / px[:-1] - 1
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    pos = {d: i for i, d in enumerate(idx)}
    execute = {}
    for d, w in targets.items():
        i = pos.get(d)
        if i is not None and i + 1 < len(idx):
            vec = np.zeros(len(cols))
            for s, v in w.items():
                if s in cols and v > 0:
                    vec[cols[s]] = v
            execute[i + 1] = (d, vec)
    w = np.zeros(len(cols))
    cash, equity = 1.0, 1.0
    curve = np.empty(len(idx))
    trades = []
    for i in range(len(idx)):
        if i:
            assets = w * (1 + rets[i])
            total = assets.sum() + cash
            equity *= total
            w, cash = assets / total, cash / total
        if i in execute:
            signal_day, tgt = execute[i]
            turnover = float(np.abs(tgt - w).sum())
            equity *= 1 - turnover * cost_bps / 1e4
            trades.append({"signal": signal_day, "date": idx[i], "before": w.copy(), "after": tgt, "turnover": turnover})
            w, cash = tgt.copy(), max(0.0, 1 - tgt.sum())
        curve[i] = equity
    return pd.Series(curve, index=idx), trades


def buy_and_hold(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    s = series[series.index >= start].ffill().dropna()
    return s / s.iloc[0]


def equal_weight(closes: pd.DataFrame, start: pd.Timestamp, cost_bps: float) -> pd.Series:
    targets = {}
    for d in signal_dates(closes.index, start, "M"):
        live = closes.loc[d].dropna().index
        targets[d] = pd.Series(1 / len(live), index=live)
    return simulate(closes, targets, start, cost_bps)[0]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(eq: pd.Series, rf: float) -> dict:
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(252)
    down = r[r < 0].std() * math.sqrt(252)
    dd = eq / eq.cummax() - 1
    monthly = eq.resample("ME").last().pct_change().dropna()
    return {
        "total_return": eq.iloc[-1] - 1,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": (cagr - rf) / vol if vol > 0 else None,
        "sortino": (cagr - rf) / down if down > 0 else None,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else None,
        "positive_months": (monthly > 0).mean() if len(monthly) else None,
        "best_month": monthly.max() if len(monthly) else None,
        "worst_month": monthly.min() if len(monthly) else None,
    }


def yearly(curves: dict[str, pd.Series]) -> list[dict]:
    out: dict[int, dict] = {}
    for name, eq in curves.items():
        ends = eq.resample("YE").last()
        prev = eq.iloc[0]
        for d, v in ends.items():
            out.setdefault(d.year, {"year": d.year})[name] = v / prev - 1
            prev = v
    return [out[y] for y in sorted(out)]


def verdict(strat: pd.Series, ref: pd.Series, s: dict, r: dict, b: dict, stock_picking: bool, momentum: bool = False) -> dict:
    """Judge the strategy against its fair reference: the index for portfolios, and for stock strategies an
    equal-weighted basket of the same companies (which carries the same survivorship bias)."""
    m = pd.concat([strat.resample("ME").last().pct_change(), ref.resample("ME").last().pct_change()], axis=1).dropna()
    excess = m.iloc[:, 0] - m.iloc[:, 1]
    t = excess.mean() / excess.std() * math.sqrt(len(excess)) if len(excess) > 2 and excess.std() > 0 else 0.0
    gap = s["cagr"] - r["cagr"]
    vs = "the same companies bought equally" if stock_picking else "the index"
    if gap > 0.005 and t >= 2:
        level, title = "meaningful", "Works: beat " + ("its fair benchmark" if stock_picking else "the market") + ", unlikely to be luck"
        text = f"{gap:+.1%} a year vs {vs}, consistently enough (t = {t:.1f}) that luck is an unlikely explanation."
    elif gap > 0.005:
        level, title = "modest", "Promising, but it could be luck"
        text = f"{gap:+.1%} a year vs {vs}, but monthly results were too uneven (t = {t:.1f}; 2 or more is convincing) to rule out luck."
    elif gap > -0.01:
        level, title = "marginal", "No real edge"
        text = f"{gap:+.1%} a year vs {vs}: not worth the extra effort and taxes."
    else:
        level, title = "negative", "Did not work"
        text = f"{gap:+.1%} a year vs {vs} over this period."
    notes = []
    if momentum and level == "meaningful":
        level, title = "modest", "Promising in the backtest; paper trade to confirm"
        notes.append("Rules that buy recent winners look best of all on today's large companies, because the stocks that soared "
                     "into that list are exactly the ones they would have bought. Expect much less in real trading.")
    if stock_picking:
        notes.append(f"Vs the index: {s['cagr']:+.1%} a year against {b['cagr']:+.1%}. That gap is inflated: the test uses today's "
                     f"largest companies, and simply owning all of them equally made {r['cagr']:+.1%} a year. Paper trade it to see real, unbiased results.")
    if s["max_drawdown"] < b["max_drawdown"] - 0.10:
        notes.append(f"Riskier ride: its worst fall was {s['max_drawdown']:.0%} vs {b['max_drawdown']:.0%} for the index.")
    elif s["max_drawdown"] > b["max_drawdown"] + 0.05:
        notes.append(f"Smoother ride: its worst fall was {s['max_drawdown']:.0%} vs {b['max_drawdown']:.0%} for the index.")
    years = (strat.index[-1] - strat.index[0]).days / 365.25
    if years < 3:
        notes.append(f"Only {years:.1f} years could be tested: treat this as a first look, not proof.")
    return {"level": level, "title": title, "text": text, "t_stat": t, "excess_cagr": gap,
            "reference": "equal_weight" if stock_picking else "benchmark", "notes": notes}


# ---------------------------------------------------------------------------
# Running a strategy
# ---------------------------------------------------------------------------

def _fmt(kind: str, v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if kind == "pe":
        return round(1 / v, 1) if v > 0 else None
    return float(v)


def _holdings(st: Strategy, ctx: Context, t: pd.Timestamp, w: pd.Series) -> list[dict]:
    out = []
    for sym, wt in w.sort_values(ascending=False).items():
        metrics = []
        for feat, label, kind in st.show:
            row = ctx.row(feat, t)
            metrics.append({"label": label, "kind": "num" if kind == "pe" else kind, "value": _fmt(kind, row.get(sym))})
        out.append({"symbol": sym, "name": ctx.names.get(sym, sym), "weight": float(wt), "price": float(ctx.closes.at[t, sym]),
                    "sector": ctx.sectors.get(sym), "metrics": metrics})
    return out


def _load(key: str, market: str, years: int, params: dict, progress):
    """(strategy, context, start date, filings) for a run."""
    if key in gurus.GURUS:
        if market != "us":
            raise ValueError("Investor 13F portfolios are US stocks only.")
        ctx, fl = gurus.context(key, years, progress)
        st = gurus.strategy(key, fl)
        start = max(ctx.closes.index[-1] - pd.DateOffset(years=years), pd.Timestamp(min(f["filed"] for f in fl)))
        return st, ctx, start, fl
    st = strategies.get(key, params)
    if market not in st.markets:
        raise ValueError(f"{st.name} is not available for this market.")
    ctx = strategies.allocation_context(market) if st.kind == "allocation" else strategies.stock_context(market, progress)
    start = max(ctx.closes.index[-1] - pd.DateOffset(years=years), ctx.closes.index[min(WARMUP_DAYS, len(ctx.closes) - 1)])
    if st.needs:
        start = max(start, ctx.first_date(st.needs))
    if st.kind == "allocation":
        first = [ctx.closes[s].first_valid_index() for s in st.assets.get(market, {}) if s in ctx.closes]
        if first:
            start = max(start, max(first) + pd.Timedelta(days=300 if key in ("antonacci", "faber") else 0))
    return st, ctx, start, None


def _signals(st: Strategy, ctx: Context, start, fl) -> list[pd.Timestamp]:
    if st.rebalance == "F":
        filed = sorted({pd.Timestamp(f["filed"]) for f in fl})
        days = []
        for f in filed:
            later = ctx.closes.index[ctx.closes.index >= f]
            if len(later) and later[0] > start:
                days.append(later[0])
        return [ctx.closes.index[ctx.closes.index >= start][0]] + days
    return signal_dates(ctx.closes.index, start, st.rebalance)


def run(key: str, market: str, years: int, cost_bps: float, params: dict, progress=lambda m: None) -> dict:
    st, ctx, start, fl = _load(key, market, years, params, progress)
    if start >= ctx.closes.index[-30]:
        raise ValueError("Not enough history to test this strategy yet.")
    progress(f"Replaying {st.name} day by day since {start.date()}…")
    targets = {d: st.select(ctx, d, params) for d in _signals(st, ctx, start, fl)}
    eq, trades = simulate(ctx.closes, targets, start, cost_bps)
    bench = buy_and_hold(ctx.bench, start).reindex(eq.index).ffill()
    rf = RISK_FREE[market]
    curves = {"strategy": eq, "benchmark": bench}
    ew = None
    if st.kind in ("stocks", "custom"):
        progress("Comparing with an equal-weighted basket of the same companies…")
        ew = equal_weight(ctx.closes, start, cost_bps).reindex(eq.index).ffill()
        curves["equal_weight"] = ew
    s, b = stats(eq, rf), stats(bench, rf)
    ews = stats(ew, rf) if ew is not None else None

    cols = list(ctx.closes.columns)
    rebalances = []
    for tr in trades[-12:][::-1]:
        before, after = tr["before"], tr["after"]
        name = lambda i: {"symbol": cols[i], "name": ctx.names.get(cols[i], cols[i])}
        rebalances.append({
            "date": str(tr["date"].date()),
            "buys": [name(i) for i in np.where((after > 1e-6) & (before <= 1e-6))[0]],
            "sells": [name(i) for i in np.where((after <= 1e-6) & (before > 1e-6))[0]],
            "holdings": int((after > 1e-6).sum()),
            "invested": float(after.sum()),
            "turnover": tr["turnover"],
        })
    n_trades = len(trades)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    last = ctx.closes.index[-1]
    now = st.select(ctx, last, params)
    weekly = eq.index[::5].append(eq.index[-1:]).unique()
    dd = lambda c: c / c.cummax() - 1
    out = {
        "status": "ready",
        "strategy": st.meta(),
        "market": market, "currency": "USD" if st.kind == "guru" else CURRENCY[market],
        "benchmark": "S&P 500 (SPY ETF)" if st.kind == "guru" else BENCHMARK[market][1],
        "cost_bps": cost_bps, "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "years": yrs,
        "curve": {"dates": [str(d.date()) for d in weekly], **{k: c.reindex(weekly).round(5).tolist() for k, c in curves.items()}},
        "drawdown": {"dates": [str(d.date()) for d in weekly], **{k: dd(c).reindex(weekly).round(4).tolist() for k, c in curves.items()}},
        "stats": {"strategy": s, "benchmark": b, **({"equal_weight": ews} if ews else {})},
        "yearly": yearly(curves),
        "verdict": verdict(eq, ew if ew is not None else bench, s, ews or b, b, ew is not None, strategies.momentum_biased(st, params)),
        "activity": {"rebalances": n_trades, "turnover_per_year": sum(t["turnover"] for t in trades) / yrs / 2 if yrs else None,
                     "avg_holdings": float(np.mean([(t["after"] > 1e-6).sum() for t in trades])) if trades else 0,
                     "avg_invested": float(np.mean([t["after"].sum() for t in trades])) if trades else 0},
        "rebalances": rebalances,
        "holdings_now": _holdings(st, ctx, last, now),
        "as_of": str(last.date()),
        "caveats": _caveats(st, ctx, start, params),
    }
    if fl:
        out["filing"] = {"filed": fl[0]["filed"], "period": fl[0]["period"], "mapped_weight": fl[0]["mapped_weight"],
                         "changes": gurus.changes(fl[0], fl[1] if len(fl) > 1 else None)[:25]}
    return out


def _caveats(st: Strategy, ctx: Context, start, params) -> list[str]:
    c = ["Past performance does not guarantee future results; strategies go through long losing spells."]
    if st.kind in ("stocks", "custom"):
        c.append("Tested on today's 300 largest companies, which favours past winners (survivorship bias); "
                 "the equal-weight comparison shows how much of the return is real stock picking.")
    if st.needs:
        c.append("Fundamentals are annual and used only 90 days after each fiscal year ends, so the test starts "
                 f"when enough of them exist ({start.date()}).")
    if st.kind == "guru":
        c.append("A 13F shows holdings up to 45 days after the quarter ends; real trades in between are invisible, "
                 "and short positions, options, bonds and non-US stocks are not included.")
    c.append("Dividends are included via adjusted prices; taxes are not.")
    return c


def _key(*parts) -> str:
    return "trade:bt:" + hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def get(key: str, market: str, years: int, cost_bps: float | None, params: dict) -> dict:
    cost = DEFAULT_COST_BPS.get(market, 10) if cost_bps is None else cost_bps
    norm = strategies.parse_custom(params) if key == "custom" else {}
    return jobs.run(_key(key, market, years, cost, norm),
                    lambda progress: run(key, market, years, cost, norm, progress), TTL)


# ---------------------------------------------------------------------------
# Today's target holdings (used by copy-trading in the paper service)
# ---------------------------------------------------------------------------

def _targets(key: str, market: str, params: dict, progress) -> dict:
    if key in gurus.GURUS:
        ctx, fl = gurus.context(key, 1, progress)
        st = gurus.strategy(key, fl)
        signature = fl[0]["accession"]
    else:
        st = strategies.get(key, params)
        ctx = strategies.allocation_context(market) if st.kind == "allocation" else strategies.stock_context(market, progress)
        signature = None
    last = ctx.closes.index[-1]
    w = st.select(ctx, last, params)
    return {"status": "ready", "strategy": st.meta(), "as_of": str(last.date()), "signature": signature,
            "weights": [{"symbol": s, "name": ctx.names.get(s, s), "weight": float(v), "price": float(ctx.closes.at[last, s])}
                        for s, v in w.sort_values(ascending=False).items()]}


def targets(key: str, market: str, params: dict) -> dict:
    norm = strategies.parse_custom(params) if key == "custom" else {}
    return jobs.run(_key("targets", key, market, norm), lambda progress: _targets(key, market, norm, progress), 3 * 3600)
