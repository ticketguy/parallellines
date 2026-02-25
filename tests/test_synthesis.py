"""Unit tests for SynthesisLayer — no DB or HTTP required."""
import pytest

from app.layers.synthesis import SynthesisLayer


@pytest.fixture
def synthesis():
    return SynthesisLayer()


def test_synthesize_single_layer(synthesis):
    result = synthesis.synthesize(
        {"market": {"score": 0.8, "confidence": 0.9, "signal_count": 10}}
    )
    assert result["score"] == pytest.approx(0.8, abs=0.01)
    assert 0.0 <= result["confidence"] <= 1.0


def test_synthesize_all_zero_confidence(synthesis):
    result = synthesis.synthesize(
        {
            "market": {"score": 0.5, "confidence": 0.0},
            "social": {"score": -0.3, "confidence": 0.0},
        }
    )
    assert result == {"score": 0.0, "confidence": 0.0}


def test_synthesize_mixed_layers(synthesis):
    result = synthesis.synthesize(
        {
            "market": {"score": 0.6, "confidence": 0.9, "signal_count": 5},
            "social": {"score": -0.2, "confidence": 0.5, "signal_count": 2},
            "news": {"score": 0.1, "confidence": 0.3, "signal_count": 1},
        }
    )
    # Market dominates (weight 0.35) with high confidence → overall should be positive
    assert result["score"] > 0
    assert 0.0 <= result["confidence"] <= 1.0


def test_synthesize_empty(synthesis):
    result = synthesis.synthesize({})
    assert result == {"score": 0.0, "confidence": 0.0}
