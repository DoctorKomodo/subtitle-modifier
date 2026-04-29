"""LLM-based subtitle recasing via the native Anthropic SDK."""

import logging

from .converter import (
    reinsert_ass_tags,
    reinsert_newline_markers,
    strip_ass_tags,
    strip_newline_markers,
    to_sentence_case,
)
from .llm import (
    _SYSTEM_PROMPT,
    _build_prompt,
    _parse_response,
)

logger = logging.getLogger(__name__)

_MAX_TOKENS = 4096
_DEFAULT_MODEL = "claude-haiku-4-5"


def recase_batch_claude(texts, client, model):
    """Send a batch of texts to Claude for recasing.

    Mirrors recase_batch() in llm.py: parse-fail retry once, then
    sentence-case fallback for the whole batch.
    """
    if not texts:
        return []

    prompt = _build_prompt(texts)

    def _call():
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), None
        )
        return text, response.stop_reason

    response_text, stop_reason = _call()
    parsed = (
        _parse_response(response_text, len(texts))
        if response_text is not None
        else None
    )
    if parsed is not None:
        return parsed

    logger.warning(
        "Claude response parse failed (stop_reason=%s), retrying batch of %d",
        stop_reason, len(texts),
    )
    response_text, stop_reason = _call()
    parsed = (
        _parse_response(response_text, len(texts))
        if response_text is not None
        else None
    )
    if parsed is not None:
        return parsed

    logger.warning(
        "Claude response parse failed twice (stop_reason=%s), falling back to sentence case",
        stop_reason,
    )
    return [to_sentence_case(t) for t in texts]


def convert_texts_claude(texts, client, model, batch_size=50):
    """Full Claude conversion pipeline for a list of subtitle event texts.

    For each text: strip ASS tags -> strip \\N markers -> lowercase ->
    batch Claude recase -> rstrip trailing whitespace -> validate wording
    invariant -> reinsert \\N markers -> reinsert ASS tags.

    Args:
        texts: Raw subtitle event texts (may contain ASS tags).
        client: An anthropic.Anthropic client instance.
        model: Claude model identifier.
        batch_size: Number of events per API call.

    Returns:
        List of recased texts, same length as input.
    """
    if not texts:
        return []

    # Pre-process: strip tags, strip \N markers, then lowercase
    stripped = []
    tag_data = []
    newline_data = []
    for text in texts:
        plain, tags = strip_ass_tags(text)
        no_markers, positions = strip_newline_markers(plain)
        lowered = no_markers.lower()
        stripped.append(lowered)
        tag_data.append(tags)
        newline_data.append(positions)

    # Batch and recase via Claude
    recased = []
    for start in range(0, len(stripped), batch_size):
        batch = stripped[start : start + batch_size]
        batch_results = recase_batch_claude(batch, client, model)
        recased.extend(batch_results)

    # Strip trailing whitespace the LLM may have added
    recased = [line.rstrip() for line in recased]

    # Validate wording invariant, reinsert \N markers and ASS tags
    results = []
    for i, (original_lowered, result_text, positions, tags) in enumerate(
        zip(stripped, recased, newline_data, tag_data)
    ):
        if result_text.lower() != original_lowered:
            logger.warning(
                "Claude altered wording for event %d, falling back to sentence case. "
                "Input: %r | Output: %r",
                i,
                original_lowered[:100],
                result_text[:100],
            )
            result_text = to_sentence_case(original_lowered)

        result_text = reinsert_newline_markers(result_text, positions)
        results.append(reinsert_ass_tags(result_text, tags))

    return results
