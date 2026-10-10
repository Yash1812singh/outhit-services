"""Countries the stock screener covers, and the global macro dashboard.

`region` and `exchanges` are Yahoo screener codes. Where a country has several
venues listing the same companies (e.g. NSE and BSE), we keep only the primary
one so each company appears once.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    code: str
    name: str
    currency: str
    exchanges: tuple[str, ...] = ()  # empty = every exchange in the region
    index: str | None = None  # benchmark index symbol
    per_sector: int = 60  # companies kept per sector (largest by market cap)
    featured: bool = False  # kept warm by the background refresher


MARKETS: list[Market] = [
    Market("in", "India", "INR", ("NSI",), "^NSEI", per_sector=200, featured=True),
    Market("us", "United States", "USD", ("NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"), "^GSPC", per_sector=200, featured=True),
    Market("gb", "United Kingdom", "GBP", ("LSE",), "^FTSE", per_sector=80),
    Market("de", "Germany", "EUR", ("GER",), "^GDAXI", per_sector=80),
    Market("fr", "France", "EUR", ("PAR",), "^FCHI", per_sector=80),
    Market("jp", "Japan", "JPY", ("JPX",), "^N225", per_sector=100),
    Market("cn", "China", "CNY", ("SHH", "SHZ"), "000001.SS", per_sector=100),
    Market("hk", "Hong Kong", "HKD", ("HKG",), "^HSI", per_sector=80),
    Market("kr", "South Korea", "KRW", ("KSC", "KOE"), "^KS11", per_sector=80),
    Market("tw", "Taiwan", "TWD", ("TAI", "TWO"), "^TWII", per_sector=80),
    Market("au", "Australia", "AUD", ("ASX",), "^AXJO", per_sector=80),
    Market("ca", "Canada", "CAD", ("TOR",), "^GSPTSE", per_sector=80),
    Market("br", "Brazil", "BRL", ("SAO",), "^BVSP"),
    Market("mx", "Mexico", "MXN", ("MEX",), "^MXX"),
    Market("ch", "Switzerland", "CHF", ("EBS",), "^SSMI"),
    Market("nl", "Netherlands", "EUR", ("AMS",), "^AEX"),
    Market("es", "Spain", "EUR", ("MCE",), "^IBEX"),
    Market("it", "Italy", "EUR", ("MIL",), "FTSEMIB.MI"),
    Market("se", "Sweden", "SEK", ("STO",), "^OMX"),
    Market("dk", "Denmark", "DKK", ("CPH",), "^OMXC25"),
    Market("fi", "Finland", "EUR", ("HEL",), "^OMXH25"),
    Market("be", "Belgium", "EUR", ("BRU",), "^BFX"),
    Market("at", "Austria", "EUR", ("VIE",), "^ATX"),
    Market("ie", "Ireland", "EUR", ("ISE",), "^ISEQ"),
    Market("pt", "Portugal", "EUR", ("LIS",), "PSI20.LS"),
    Market("gr", "Greece", "EUR", ("ATH",), "GD.AT"),
    Market("sg", "Singapore", "SGD", ("SES",), "^STI"),
    Market("id", "Indonesia", "IDR", ("JKT",), "^JKSE"),
    Market("my", "Malaysia", "MYR", ("KLS",), "^KLSE"),
    Market("nz", "New Zealand", "NZD", ("NZE",), "^NZ50"),
    Market("il", "Israel", "ILS", ("TLV",), "^TA125.TA"),
    Market("za", "South Africa", "ZAR", ("JNB",), "^J203.JO"),
    Market("ar", "Argentina", "ARS", ("BUE",), "^MERV"),
    Market("tr", "Turkey", "TRY", ("IST",), "XU100.IS"),
    # Covered by the screener; Yahoo has no reliable index history for these.
    Market("th", "Thailand", "THB", ("SET",)),
    Market("sa", "Saudi Arabia", "SAR", ("SAU",)),
    Market("no", "Norway", "NOK", ("OSL",)),
    Market("pl", "Poland", "PLN", ("WSE",)),
    Market("cl", "Chile", "CLP", ("SGO",)),
    Market("eg", "Egypt", "EGP", ("CAI",)),
    Market("qa", "Qatar", "QAR", ("DOH",)),
    Market("kw", "Kuwait", "KWD", ("KUW",)),
    Market("cz", "Czechia", "CZK", ("PRA",)),
    Market("hu", "Hungary", "HUF", ("BUD",)),
    Market("ro", "Romania", "RON", ("BVB",)),
    Market("is", "Iceland", "ISK", ("ICE",)),
    Market("ee", "Estonia", "EUR", ("TAL",)),
    Market("lt", "Lithuania", "EUR", ("LIT",)),
    Market("lv", "Latvia", "EUR", ("RIS",)),
]

MARKETS_BY_CODE = {m.code: m for m in MARKETS}

SECTORS = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Communication Services",
    "Energy",
    "Basic Materials",
    "Utilities",
    "Real Estate",
]


@dataclass(frozen=True)
class MacroAsset:
    symbol: str
    name: str
    group: str
    unit: str = ""
    extra: dict = field(default_factory=dict)


MACRO: list[MacroAsset] = [
    *(MacroAsset(m.index, f"{m.name}", "Stock indices", extra={"market": m.code}) for m in MARKETS if m.index),
    MacroAsset("^IXIC", "Nasdaq Composite (US)", "Stock indices", extra={"market": "us"}),
    MacroAsset("^DJI", "Dow Jones (US)", "Stock indices", extra={"market": "us"}),
    MacroAsset("^BSESN", "Sensex (India)", "Stock indices", extra={"market": "in"}),
    MacroAsset("^STOXX50E", "Euro Stoxx 50", "Stock indices"),
    MacroAsset("^VIX", "VIX (US fear gauge)", "Volatility"),
    MacroAsset("^INDIAVIX", "India VIX", "Volatility"),
    MacroAsset("DX-Y.NYB", "US Dollar Index", "Currencies"),
    MacroAsset("USDINR=X", "USD / INR", "Currencies"),
    MacroAsset("EURUSD=X", "EUR / USD", "Currencies"),
    MacroAsset("GBPUSD=X", "GBP / USD", "Currencies"),
    MacroAsset("USDJPY=X", "USD / JPY", "Currencies"),
    MacroAsset("USDCNY=X", "USD / CNY", "Currencies"),
    MacroAsset("GC=F", "Gold", "Commodities", "USD/oz"),
    MacroAsset("SI=F", "Silver", "Commodities", "USD/oz"),
    MacroAsset("CL=F", "Crude oil (WTI)", "Commodities", "USD/bbl"),
    MacroAsset("BZ=F", "Crude oil (Brent)", "Commodities", "USD/bbl"),
    MacroAsset("NG=F", "Natural gas", "Commodities", "USD/MMBtu"),
    MacroAsset("HG=F", "Copper", "Commodities", "USD/lb"),
    MacroAsset("^IRX", "US 3-month T-bill yield", "Interest rates", "%"),
    MacroAsset("^TNX", "US 10-year yield", "Interest rates", "%"),
    MacroAsset("^TYX", "US 30-year yield", "Interest rates", "%"),
    MacroAsset("BTC-USD", "Bitcoin", "Crypto"),
    MacroAsset("ETH-USD", "Ethereum", "Crypto"),
    MacroAsset("SOL-USD", "Solana", "Crypto"),
]
