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

    Implementation in subsequent tasks.
    """
    raise NotImplementedError


def convert_texts_claude(texts, client, model, batch_size=50):
    """Full Claude conversion pipeline.

    Implementation in a subsequent task.
    """
    raise NotImplementedError
