"""LLM-based subtitle recasing via the native Anthropic SDK."""

import logging

from .converter import to_sentence_case
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
    """Full Claude conversion pipeline.

    Implementation in a subsequent task.
    """
    raise NotImplementedError
