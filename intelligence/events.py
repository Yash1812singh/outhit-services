"""Corporate-event tagging for headlines.

The sentiment model judges tone; these rules catch specific events that matter
for a company's prospects and reputation (a fraud probe is not "a bit negative").
`impact` is in [-1, 1]; `controversy` events drive the reputation score.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    key: str
    label: str
    impact: float
    pattern: re.Pattern
    controversy: bool = False


def _p(rx: str) -> re.Pattern:
    return re.compile(rx, re.IGNORECASE)


EVENTS = [
    # Positive
    Event("upgrade", "Analyst upgrade", 0.6, _p(r"\bupgrade[sd]?\b|\braises? (?:price )?target|\bbuy rating|\boutperform rating|\brated buy")),
    Event("beat", "Results beat expectations", 0.7, _p(r"\bbeats?\b.*\b(estimates?|expectations?|forecasts?)|\bprofit (?:jumps?|soars?|surges?|rises?)|\brecord (?:profit|revenue|quarter|sales)")),
    Event("guidance_up", "Raised guidance", 0.7, _p(r"\braises? (?:its |full-year |annual )?(?:guidance|outlook|forecast)|\bhikes? (?:guidance|outlook)")),
    Event("order_win", "Big order / contract win", 0.5, _p(r"\b(?:wins?|bags?|secures?|lands?)\b.*\b(?:order|contract|deal|tender)")),
    Event("buyback", "Share buyback", 0.4, _p(r"\bbuy-?backs?\b|\bshare repurchase")),
    Event("dividend_up", "Dividend increase", 0.3, _p(r"\b(?:raises?|hikes?|increases?)\b.*\bdividend|\bspecial dividend")),
    Event("approval", "Regulatory approval", 0.5, _p(r"\b(?:fda|usfda|regulatory|government) (?:approval|nod|clearance)|\bgets? (?:nod|approval)")),
    Event("partnership", "Partnership / acquisition", 0.3, _p(r"\bpartnership\b|\bpartners with\b|\bacquires?\b|\bacquisition of\b|\bjoint venture")),
    # Negative
    Event("downgrade", "Analyst downgrade", -0.6, _p(r"\bdowngrade[sd]?\b|\bcuts? (?:price )?target|\bsell rating|\bunderperform rating|\brated sell")),
    Event("miss", "Results miss expectations", -0.7, _p(r"\bmiss(?:es|ed)?\b.*\b(estimates?|expectations?|forecasts?)|\bprofit (?:falls?|drops?|plunges?|slumps?|declines?)|\b(?:net )?loss widens")),
    Event("guidance_down", "Cut guidance", -0.7, _p(r"\bcuts? (?:its |full-year |annual )?(?:guidance|outlook|forecast)|\blowers? (?:guidance|outlook|forecast)|\bprofit warning")),
    Event("layoffs", "Layoffs", -0.3, _p(r"\blay-?offs?\b|\bjob cuts?\b|\bcuts? \d[\d,]* jobs")),
    # Reputation-damaging controversies
    Event("fraud", "Fraud / accounting allegations", -1.0, _p(r"\bfraud\b|\bscam\b|\bembezzl|\baccounting (?:irregularit|scandal|probe)|\bforensic audit|\bmisstat"), controversy=True),
    Event("regulator", "Regulator action / probe", -0.8, _p(r"\b(?:sebi|sec|ed|cbi|doj|ftc|fca|cci|income tax|i-t)\b.*\b(?:probe|order|notice|ban|raid|charges?|penalt|investigat|show-cause)|\bunder investigation|\braids?\b"), controversy=True),
    Event("lawsuit", "Lawsuit / legal trouble", -0.5, _p(r"\blawsuit\b|\bsued\b|\bclass action\b|\bcourt (?:order|ruling) against|\bverdict against"), controversy=True),
    Event("penalty", "Fine / penalty", -0.5, _p(r"\bfined?\b|\bpenalt(?:y|ies)\b"), controversy=True),
    Event("default", "Debt default / insolvency", -1.0, _p(r"\bdefault(?:s|ed)?\b.*\b(?:loan|debt|bond|payment)|\binsolvency\b|\bbankrupt|\bnclt\b|\bchapter 11"), controversy=True),
    Event("governance", "Governance / management turmoil", -0.6, _p(r"\b(?:ceo|cfo|chairman|md|auditor|director)s? (?:resigns?|quits?|steps? down|sacked|ousted)|\bwhistle-?blower"), controversy=True),
    Event("recall", "Product recall / safety issue", -0.5, _p(r"\brecalls?\b|\bsafety (?:issue|concern|probe)|\bdata breach|\bcyber-?attack|\boutage"), controversy=True),
    Event("pledge", "Promoter share pledge / stake sale", -0.4, _p(r"\bpledge[sd]?\b|\bpromoters? (?:sell|offload|cut) stake")),
]


def tag(title: str) -> list[dict]:
    return [{"key": e.key, "label": e.label, "impact": e.impact, "controversy": e.controversy} for e in EVENTS if e.pattern.search(title)]
