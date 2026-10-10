"""The curated set of instruments the advisor ranks and recommends from.

Users can still analyse any Yahoo Finance symbol via /api/market/asset/{symbol};
this list is only what the overview and the allocation planner choose among.
"""

from dataclasses import dataclass

# Asset classes, in the order the UI shows them.
EQUITY_IN = "equity_in"
EQUITY_US = "equity_us"
INDEX_FUND = "index_fund"
BONDS = "bonds"
GOLD = "gold"
REIT = "reit"
CRYPTO = "crypto"

ASSET_CLASS_LABELS = {
    INDEX_FUND: "Index funds & ETFs",
    EQUITY_US: "US stocks",
    EQUITY_IN: "Indian stocks",
    BONDS: "Bonds",
    GOLD: "Gold & commodities",
    REIT: "Real estate (REITs)",
    CRYPTO: "Crypto",
}


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    asset_class: str
    region: str  # "IN", "US" or "GLOBAL"
    currency: str


UNIVERSE: list[Asset] = [
    # Broad index funds - the core of most sensible portfolios.
    Asset("NIFTYBEES.NS", "Nippon India Nifty 50 BeES", INDEX_FUND, "IN", "INR"),
    Asset("JUNIORBEES.NS", "Nippon India Nifty Next 50 BeES", INDEX_FUND, "IN", "INR"),
    Asset("SPY", "SPDR S&P 500 ETF", INDEX_FUND, "US", "USD"),
    Asset("VTI", "Vanguard Total Stock Market ETF", INDEX_FUND, "US", "USD"),
    Asset("QQQ", "Invesco QQQ (Nasdaq 100)", INDEX_FUND, "US", "USD"),
    Asset("VXUS", "Vanguard Total International Stock ETF", INDEX_FUND, "GLOBAL", "USD"),
    # Large-cap US stocks.
    Asset("AAPL", "Apple", EQUITY_US, "US", "USD"),
    Asset("MSFT", "Microsoft", EQUITY_US, "US", "USD"),
    Asset("NVDA", "NVIDIA", EQUITY_US, "US", "USD"),
    Asset("GOOGL", "Alphabet", EQUITY_US, "US", "USD"),
    Asset("AMZN", "Amazon", EQUITY_US, "US", "USD"),
    Asset("META", "Meta Platforms", EQUITY_US, "US", "USD"),
    Asset("JPM", "JPMorgan Chase", EQUITY_US, "US", "USD"),
    Asset("V", "Visa", EQUITY_US, "US", "USD"),
    Asset("BRK-B", "Berkshire Hathaway", EQUITY_US, "US", "USD"),
    Asset("TSLA", "Tesla", EQUITY_US, "US", "USD"),
    # Large-cap Indian stocks.
    Asset("RELIANCE.NS", "Reliance Industries", EQUITY_IN, "IN", "INR"),
    Asset("TCS.NS", "Tata Consultancy Services", EQUITY_IN, "IN", "INR"),
    Asset("HDFCBANK.NS", "HDFC Bank", EQUITY_IN, "IN", "INR"),
    Asset("ICICIBANK.NS", "ICICI Bank", EQUITY_IN, "IN", "INR"),
    Asset("INFY.NS", "Infosys", EQUITY_IN, "IN", "INR"),
    Asset("ITC.NS", "ITC", EQUITY_IN, "IN", "INR"),
    Asset("BHARTIARTL.NS", "Bharti Airtel", EQUITY_IN, "IN", "INR"),
    Asset("LT.NS", "Larsen & Toubro", EQUITY_IN, "IN", "INR"),
    # Bonds.
    Asset("BND", "Vanguard Total Bond Market ETF", BONDS, "US", "USD"),
    Asset("IEF", "iShares 7-10Y Treasury ETF", BONDS, "US", "USD"),
    Asset("TLT", "iShares 20+Y Treasury ETF", BONDS, "US", "USD"),
    Asset("EBBETF0431.NS", "Bharat Bond ETF April 2031", BONDS, "IN", "INR"),
    Asset("GILT5YBEES.NS", "Nippon India 5Y Gilt BeES", BONDS, "IN", "INR"),
    # Gold & commodities.
    Asset("GOLDBEES.NS", "Nippon India Gold BeES", GOLD, "IN", "INR"),
    Asset("GLD", "SPDR Gold Shares", GOLD, "US", "USD"),
    Asset("SLV", "iShares Silver Trust", GOLD, "US", "USD"),
    # Real estate.
    Asset("VNQ", "Vanguard Real Estate ETF", REIT, "US", "USD"),
    Asset("EMBASSY.NS", "Embassy Office Parks REIT", REIT, "IN", "INR"),
    # Crypto.
    Asset("BTC-USD", "Bitcoin", CRYPTO, "GLOBAL", "USD"),
    Asset("ETH-USD", "Ethereum", CRYPTO, "GLOBAL", "USD"),
    Asset("SOL-USD", "Solana", CRYPTO, "GLOBAL", "USD"),
    Asset("BNB-USD", "BNB", CRYPTO, "GLOBAL", "USD"),
    Asset("XRP-USD", "XRP", CRYPTO, "GLOBAL", "USD"),
]

BENCHMARKS: list[Asset] = [
    Asset("^NSEI", "Nifty 50", INDEX_FUND, "IN", "INR"),
    Asset("^BSESN", "Sensex", INDEX_FUND, "IN", "INR"),
    Asset("^GSPC", "S&P 500", INDEX_FUND, "US", "USD"),
    Asset("^IXIC", "Nasdaq Composite", INDEX_FUND, "US", "USD"),
    Asset("BTC-USD", "Bitcoin", CRYPTO, "GLOBAL", "USD"),
    Asset("GC=F", "Gold futures", GOLD, "GLOBAL", "USD"),
]

_BY_SYMBOL = {a.symbol: a for a in UNIVERSE + BENCHMARKS}


def lookup(symbol: str) -> Asset:
    """Metadata for a symbol, inferring sensible defaults for ones outside the universe."""
    symbol = symbol.upper()
    if symbol in _BY_SYMBOL:
        return _BY_SYMBOL[symbol]
    if symbol.endswith((".NS", ".BO")):
        return Asset(symbol, symbol, EQUITY_IN, "IN", "INR")
    if symbol.endswith(("-USD", "-USDT")):
        return Asset(symbol, symbol, CRYPTO, "GLOBAL", "USD")
    return Asset(symbol, symbol, EQUITY_US, "US", "USD")
