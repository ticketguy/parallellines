from enum import StrEnum


class LayerType(StrEnum):
    PROBABILITY = "probability"
    CONVICTION = "conviction"
    ECHO = "echo"
    MEMORY = "memory"
    SHADOW = "shadow"
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
