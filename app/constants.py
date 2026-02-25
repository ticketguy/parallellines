from enum import StrEnum


class LayerType(StrEnum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    SENTIMENT = "sentiment"
    GEOPOLITICAL = "geopolitical"
    SYNTHESIS = "synthesis"


class NarrativeType(StrEnum):
    EMERGING = "emerging"
    DOMINANT = "dominant"
    FADING = "fading"
    CONTRARIAN = "contrarian"


class ResolutionStatus(StrEnum):
    OPEN = "open"
    RESOLVED_YES = "resolved_yes"
    RESOLVED_NO = "resolved_no"
    CANCELLED = "cancelled"
