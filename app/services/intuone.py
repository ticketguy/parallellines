from app.layers.conviction import ConvictionLayer
from app.layers.echo import EchoLayer
from app.layers.memory import MemoryLayer
from app.layers.probability import ProbabilityLayer
from app.layers.shadow import ShadowLayer
from app.layers.synthesis import SynthesisLayer
from app.schemas.report import IntuOneReportCreate
from app.schemas.signal import SignalRead

# Primary layer processors — all five lenses run over every signal
PRIMARY_LAYERS = [
    ProbabilityLayer(),
    ConvictionLayer(),
    EchoLayer(),
    MemoryLayer(),
    ShadowLayer(),
]

_synthesis = SynthesisLayer()


async def generate_report(
    signals: list[SignalRead],
    topic: str,
    time_window: str = "24h",
) -> IntuOneReportCreate:
    """
    Run all five primary layers over the provided signals and synthesise
    the results into an IntuOneReport.

    Steps:
    1. Pass all signals through every layer independently.
    3. Synthesise: weighted confidence-adjusted aggregation.
    4. Return a report object ready to be persisted.
    """
    layer_scores: dict[str, dict] = {}

    for layer in PRIMARY_LAYERS:
        ls = await layer.score(signals, topic, time_window)
        layer_scores[layer.layer_name] = {
            "score": ls.score,
            "confidence": ls.confidence,
            "signal_count": ls.signal_count,
        }

    overall = await _synthesis.synthesize(layer_scores, topic=topic)

    return IntuOneReportCreate(
        topic=topic,
        overall_score=overall["score"],
        confidence=overall["confidence"],
        layer_breakdown=layer_scores,
        time_window=time_window,
    )
