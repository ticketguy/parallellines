"""
IntuOne reasoning — chain-of-thought runs through the LOCAL model.

The model generates its own private reasoning before producing its final
response. No external API calls. No Claude. The model thinks for itself.

If the model is not yet loaded (pre-training), thinking is skipped — the
pipeline still works, just without the reasoning pre-pass.

Prompt design:
  The thinking prompt instructs the model to reason through signals step
  by step inside [THINK]...[/THINK] tags. The model is trained to produce
  these tags as part of its output. The reasoning is stripped before showing
  the user and stored separately as a "thinking" message.
"""
import logging
import re

logger = logging.getLogger(__name__)

_THINK_PREFIX = """\
[THINK]
You are IntuOne reasoning privately before responding. Work through this step by step:
1. Which layers have the strongest signals? What direction are they pointing?
2. Are the layers converging or diverging? What does that mean for confidence?
3. What is the single most important signal driving this read?
4. What would change your mind? What is the biggest uncertainty?
5. What is your overall directional read and why?

Topic: {topic}
User asked: {user_message}

Signal context:
{signal_context}
[/THINK]
"""

_EXPLORE_PREFIX = """\
[EXPLORE]
Look beyond the surface. What non-obvious patterns exist in this data?
- Any anomalies -- signals that do not fit the overall picture?
- Any notable absences -- layers that should have data but do not?
- Any cross-layer dynamics -- one layer leading another?
- Any second-order implications if the primary signal is correct?

Signal context:
{signal_context}

Your previous reasoning:
{thinking}
[/EXPLORE]
"""


def _strip_tags(text: str, tag: str) -> tuple[str, str]:
    pattern = rf"\[{tag}\](.*?)\[/{tag}\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return "", text
    inner = match.group(1).strip()
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return inner, cleaned


def think(
    topic: str,
    user_message: str,
    signal_context: str,
    memory_context: str = "",
) -> str:
    """
    Run a private chain-of-thought pass through the local model.
    Returns "" if the model is not yet loaded.
    """
    from app.inference.pipeline import generate_briefing_local, is_model_loaded

    if not is_model_loaded():
        return ""

    prompt = _THINK_PREFIX.format(
        topic=topic,
        user_message=user_message,
        signal_context=signal_context[:1200],
    )
    if memory_context:
        prompt += f"\nRelevant memory context:\n{memory_context[:400]}\n"

    try:
        raw = generate_briefing_local(
            prompt,
            max_new_tokens=350,
            temperature=0.3,
            repetition_penalty=1.1,
        )
        inner, _ = _strip_tags(raw, "THINK")
        return inner if inner else raw.strip()
    except Exception as exc:
        logger.warning("Thinking pass failed (non-fatal): %s", exc)
        return ""


def explore_insights(
    topic: str,
    signal_context: str,
    thinking: str,
) -> list[str]:
    """
    Run a second model pass to surface non-obvious cross-layer patterns.
    Returns [] if the model is not loaded.
    """
    from app.inference.pipeline import generate_briefing_local, is_model_loaded

    if not is_model_loaded() or not thinking:
        return []

    prompt = _EXPLORE_PREFIX.format(
        signal_context=signal_context[:800],
        thinking=thinking[:400],
    )

    try:
        raw = generate_briefing_local(
            prompt,
            max_new_tokens=200,
            temperature=0.4,
            repetition_penalty=1.1,
        )
        inner, _ = _strip_tags(raw, "EXPLORE")
        text = inner if inner else raw.strip()
        insights = [
            line.strip().lstrip("*-").strip()
            for line in text.splitlines()
            if len(line.strip()) > 30
        ]
        return insights[:4]
    except Exception as exc:
        logger.debug("Insight exploration failed (non-fatal): %s", exc)
        return []


def format_thinking_for_context(thinking: str) -> str:
    if not thinking:
        return ""
    return (
        "[INTUONE PRIOR REASONING -- use this to inform your response]\n"
        + thinking
        + "\n[END REASONING]\n\n"
    )
