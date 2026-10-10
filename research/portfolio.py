"""Portfolio builder: turns the recommendations into a diversified, sized portfolio.

Research is clear that most of the result comes from how a portfolio is built,
not only what is picked:
- Core: a low-cost index fund (most active funds lose to the index; SPIVA).
- Satellite: the long-term picks, sized by inverse volatility (calmer stocks get
  more), capped per stock and per sector so no single idea can sink the plan.
- Defensive: short-term bonds, plus gold only while gold is in an uptrend (trend rule).
- Regime: in risky markets part of the satellite moves to the defensive bucket.
- Rebalance quarterly, or when a holding drifts more than 25% from its target.
"""

import math

from research import data, recommend, regime, tournament

SPLITS = {  # core / satellite / defensive
    "conservative": (0.60, 0.15, 0.25),
    "balanced": (0.55, 0.30, 0.15),
    "aggressive": (0.45, 0.45, 0.10),
}
N_PICKS = {"conservative": 8, "balanced": 10, "aggressive": 12}
MAX_STOCK = 0.15  # of the satellite
MAX_SECTOR = 0.30  # of the satellite
CORE = {"in": [("NIFTYBEES.NS", "Nippon India Nifty 50 ETF", 0.75), ("JUNIORBEES.NS", "Nippon India Nifty Next 50 ETF", 0.25)],
        "us": [("VTI", "Vanguard Total Stock Market ETF", 1.0)]}
DEFENSIVE = {"in": {"bonds": ("EBBETF0431.NS", "Bharat Bond ETF 2031"), "gold": ("GOLDBEES.NS", "Nippon India Gold ETF")},
             "us": {"bonds": ("BND", "Vanguard Total Bond Market ETF"), "gold": ("GLD", "SPDR Gold Shares")}}
TAX = {
    "in": "India (rules as of Budget 2024; check current rates): equity held over 12 months is taxed at 12.5% on gains above ₹1.25 lakh a year; under 12 months at 20%. Rebalancing triggers tax, so prefer adding new money to underweight holdings over selling.",
    "us": "US: gains on holdings kept over a year are taxed at lower long-term rates; prefer rebalancing with new money or inside tax-advantaged accounts.",
}


def _cap_weights(w: dict[str, float], sectors: dict[str, str]) -> dict[str, float]:
    """Iteratively cap each stock at MAX_STOCK and each sector at MAX_SECTOR, redistributing the excess."""
    for _ in range(50):
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        changed = False
        for k, v in w.items():
            if v > MAX_STOCK + 1e-9:
                w[k] = MAX_STOCK
                changed = True
        sec_tot: dict[str, float] = {}
        for k, v in w.items():
            sec_tot[sectors[k]] = sec_tot.get(sectors[k], 0) + v
        for sec, tot in sec_tot.items():
            if tot > MAX_SECTOR + 1e-9:
                for k in w:
                    if sectors[k] == sec:
                        w[k] *= MAX_SECTOR / tot
                changed = True
        if not changed:
            break
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def build(market: str, amount: float, risk: str) -> dict:
    if market not in CORE:
        raise ValueError("The portfolio builder supports India (in) and the US (us).")
    lt = recommend.get(market, "long_term")
    if lt.get("status") != "ready":
        return lt
    t = tournament.get(market)
    rg = regime.detect(market)
    core_w, sat_w, def_w = SPLITS[risk]
    shift = sat_w * (1 - rg["equity_exposure"])  # risky regime: move part of the satellite to defensive
    sat_w, def_w = sat_w - shift, def_w + shift

    picks = lt["picks"][: N_PICKS[risk]]
    vols = {p["symbol"]: (p.get("volatility") or t["companies"].get(p["symbol"], {}).get("vol_63") or 0.3) for p in picks}
    inv = {s: 1 / max(v, 0.08) for s, v in vols.items()}
    sat = _cap_weights(inv, {p["symbol"]: p["sector"] for p in picks}) if picks else {}

    gold_sym = DEFENSIVE[market]["gold"][0]
    gc, _ = data.daily([gold_sym], period="2y")
    gold_up = bool(gold_sym in gc and gc[gold_sym].dropna().iloc[-1] > gc[gold_sym].rolling(200).mean().iloc[-1])
    gold_share = 0.33 if gold_up else 0.0  # trend rule: hold gold only while it is above its 200-day average

    fund_syms = [x[0] for x in CORE[market]] + [DEFENSIVE[market]["bonds"][0], gold_sym]
    fc, _ = data.daily(fund_syms, period="2y")
    fund_price = {k: float(fc[k].dropna().iloc[-1]) for k in fund_syms if k in fc and fc[k].notna().any()}
    ccy = "INR" if market == "in" else "USD"
    holdings = []

    def add(bucket, sym, name, weight, price=None, currency=None, note=None, sector=None):
        value = amount * weight
        shares = math.floor(value / price) if price else None
        if shares == 0:
            note = (note + ". " if note else "") + "Amount is below one share: use fractional shares if your broker offers them, or skip this one"
        holdings.append({"bucket": bucket, "symbol": sym, "name": name, "weight": weight, "amount": value,
                         "approx_shares": shares, "price": price, "currency": currency, "sector": sector, "note": note})

    for sym, name, share in CORE[market]:
        add("core", sym, name, core_w * share, fund_price.get(sym), ccy, note="Low-cost index fund: the base of the portfolio")
    for p in picks:
        if p["symbol"] in sat:
            add("satellite", p["symbol"], p["name"], sat_w * sat[p["symbol"]], p.get("price"), p.get("currency"),
                note=f"Outhit score {p['score']:.0f}; {('; '.join(p['reasons'][:2]))}", sector=p["sector"])
    bsym, bname = DEFENSIVE[market]["bonds"]
    gsym, gname = DEFENSIVE[market]["gold"]
    add("defensive", bsym, bname, def_w * (1 - gold_share), fund_price.get(bsym), ccy, note="Short/medium-term bonds for stability")
    if gold_share:
        add("defensive", gsym, gname, def_w * gold_share, fund_price.get(gsym), ccy, note="Gold, held while it is in an uptrend (above its 200-day average)")
    else:
        holdings[-1]["note"] += ". Gold is left out for now: it is below its 200-day average (trend rule)"

    return {
        "status": "ready", "market": market, "risk": risk, "amount": amount,
        "split": {"core": core_w, "satellite": sat_w, "defensive": def_w},
        "regime": rg,
        "holdings": holdings,
        "rules": [
            f"Core {core_w:.0%} in index funds, satellite {sat_w:.0%} in Outhit's long-term picks, defensive {def_w:.0%} in bonds{' and gold' if gold_share else ''}.",
            f"Stocks are sized by inverse volatility, max {MAX_STOCK:.0%} of the satellite each and {MAX_SECTOR:.0%} per sector.",
            "Rebalance every quarter, or sooner if any holding drifts more than 25% from its target weight.",
            "Replace a pick if it falls out of Outhit's top 30 long-term ranking or gets 2+ red flags; don't react to day-to-day moves.",
            "If you are investing a large lump sum in a 'caution' or 'risk-off' regime, consider staggering it over 3-6 months.",
        ],
        "tax_note": TAX[market],
    }
