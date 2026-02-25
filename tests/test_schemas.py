"""Basic schema validation tests — no DB required."""
from datetime import datetime, timezone

import pytest

from app.schemas.layer import LayerScoreCreate
from app.schemas.report import IntuOneReportCreate
from app.schemas.signal import SignalCreate


def test_signal_create_valid():
    sig = SignalCreate(
        source="polymarket",
        layer="market",
        raw_data={"question": "Will X happen?"},
        signal_strength=0.75,
        confidence=0.9,
        signal_timestamp=datetime.now(timezone.utc),
    )
    assert sig.source == "polymarket"
    assert sig.signal_strength == 0.75


def test_signal_strength_bounds():
    with pytest.raises(Exception):
        SignalCreate(
            source="test",
            layer="market",
            raw_data={},
            signal_strength=1.5,  # out of range
            signal_timestamp=datetime.now(timezone.utc),
        )


def test_layer_score_score_bounds():
    with pytest.raises(Exception):
        LayerScoreCreate(
            layer="market",
            topic="BTC",
            score=2.0,  # out of range
            confidence=0.5,
        )


def test_report_create_valid():
    report = IntuOneReportCreate(
        topic="US election 2026",
        overall_score=0.42,
        confidence=0.68,
        layer_breakdown={
            "market": {"score": 0.6, "confidence": 0.9, "signal_count": 12},
            "social": {"score": 0.2, "confidence": 0.4, "signal_count": 3},
        },
    )
    assert report.overall_score == 0.42
