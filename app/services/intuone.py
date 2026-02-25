from app.layers.geopolitical import GeopoliticalLayer
from app.layers.market import MarketLayer
from app.layers.news import NewsLayer
from app.layers.sentiment import SentimentLayer
from app.layers.social import SocialLayer
from app.layers.synthesis import SynthesisLayer
from app.schemas.report import IntuOneReportCreate
from app.schemas.signal import SignalRead

# Primary layer processors, in order of precedence
PRIMARY_LAYERS = [
    MarketLayer(),
    SocialLayer(),
    NewsLayer(),
    SentimentLayer(),
    GeopoliticalLayer(),
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
    1. Partition signals by layer.
    2. Score each layer independently.
    3. Synthesise: weighted confidence-adjusted aggregation.
    4. Return a report object ready to be persisted.
    """
    layer_scores: dict[str, dict] = {}

    for layer in PRIMARY_LAYERS:
        layer_signals = [s for s in signals if s.layer == layer.layer_name]
        ls = await layer.score(layer_signals, topic, time_window)
        layer_scores[layer.layer_name] = {
            "score": ls.score,
            "confidence": ls.confidence,
            "signal_count": ls.signal_count,
        }

    overall = _synthesis.synthesize(layer_scores)

    return IntuOneReportCreate(
        topic=topic,
        overall_score=overall["score"],
        confidence=overall["confidence"],
        layer_breakdown=layer_scores,
        time_window=time_window,
    )
