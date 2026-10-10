"""HTTP clients for the services the advisor depends on."""

from advisor.config import get_settings
from outhit_common.client import ServiceClient

_s = get_settings()
market = ServiceClient(_s.market_data_url, "market-data")
intelligence = ServiceClient(_s.intelligence_url, "intelligence")
research = ServiceClient(_s.research_url, "research")
