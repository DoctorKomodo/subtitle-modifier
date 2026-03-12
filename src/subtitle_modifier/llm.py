"""LLM-based subtitle recasing via OpenAI-compatible API."""

import logging
import re

from .converter import (
    reinsert_ass_tags,
    strip_ass_tags,
    to_sentence_case,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Re-capitalize these subtitles. ONLY change the case of letters (uppercase/lowercase). \
Do not add, remove, or change any characters including spaces, punctuation, and apostrophes. \
The output must be identical to the input except for letter casing.

Rules:
- Sentence case: capitalize first letter of each sentence
- Capitalize proper nouns, names, places, and abbreviations (e.g. D.E.A., U.S.)
- Preserve spacing exactly — do not collapse, add, or remove spaces \
(e.g. "he went to the store" must NOT become "he went to  the store" or "he went to thestore")
- Return ONLY the numbered lines in the same format"""

_LINE_RE = re.compile(r"^(\d+):\s?(.*)", re.MULTILINE)

# Markdown code fences that LLMs sometimes wrap responses in
_CODE_FENCE_RE = re.compile(r"^```\w*\n?(.*?)```\s*$", re.DOTALL)


def _strip_newline_markers(text: str) -> tuple[str, list[int]]:
    """Replace \\N markers with a space, recording their character positions."""
    positions = []
    parts = []
    pos = 0
    i = 0
    while i < len(text):
        if text[i:i+2] == "\\N":
            positions.append(pos)
            parts.append(" ")
            pos += 1
            i += 2
        else:
            parts.append(text[i])
            pos += 1
            i += 1
    return "".join(parts), positions


def _reinsert_newline_markers(text: str, positions: list[int]) -> str:
    """Reinsert \\N markers, replacing the space at each recorded position."""
    if not positions:
        return text
    result = []
    prev = 0
    for pos in positions:
        result.append(text[prev:pos])
        result.append("\\N")
        prev = pos + 1  # skip the space that \N replaced
    result.append(text[prev:])
    return "".join(result)


def _build_prompt(texts: list[str]) -> str:
    """Build a numbered prompt from a list of texts."""
    return "\n".join(f"{i + 1}: {t}" for i, t in enumerate(texts))


def _strip_code_fences(response: str) -> str:
    """Strip markdown code fences if the LLM wrapped its response."""
    m = _CODE_FENCE_RE.match(response.strip())
    if m:
        return m.group(1)
    return response


def _parse_response(response: str, expected_count: int) -> list[str] | None:
    """Parse numbered lines from the LLM response.

    Returns None if parsing fails or count doesn't match.
    """
    response = _strip_code_fences(response)

    matches = _LINE_RE.findall(response)
    if len(matches) != expected_count:
        logger.debug(
            "Parse: expected %d lines, got %d. Raw response:\n%s",
            expected_count,
            len(matches),
            response[:2000],
        )
        return None

    result = []
    for i, (num_str, text) in enumerate(matches):
        if int(num_str) != i + 1:
            logger.debug(
                "Parse: expected line %d, got %s. Raw response:\n%s",
                i + 1,
                num_str,
                response[:2000],
            )
            return None
        result.append(text)
    return result


def recase_batch(texts: list[str], client, model: str) -> list[str]:
    """Send a batch of texts to the LLM for recasing.

    Args:
        texts: Pre-processed (tag-stripped, lowercased) subtitle lines.
        client: An OpenAI-compatible client instance.
        model: Model name to use.

    Returns:
        List of recased texts, same length as input.
        Falls back to to_sentence_case() on failure.
    """
    if not texts:
        return []

    prompt = _build_prompt(texts)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    response_text = response.choices[0].message.content

    parsed = _parse_response(response_text, len(texts))
    if parsed is not None:
        return parsed

    # Retry once on parse failure
    logger.warning("LLM response parse failed, retrying batch of %d", len(texts))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content

    parsed = _parse_response(response_text, len(texts))
    if parsed is not None:
        return parsed

    # Fall back to sentence case for entire batch
    logger.warning("LLM response parse failed twice, falling back to sentence case")
    return [to_sentence_case(t) for t in texts]


def convert_texts_llm(
    texts: list[str],
    client,
    model: str,
    batch_size: int = 50,
) -> list[str]:
    """Full LLM conversion pipeline for a list of subtitle event texts.

    For each text: strip ASS tags -> strip \\N markers -> lowercase ->
    batch LLM recase -> validate wording invariant ->
    reinsert \\N markers -> reinsert ASS tags.

    Args:
        texts: Raw subtitle event texts (may contain ASS tags).
        client: An OpenAI-compatible client instance.
        model: Model name to use.
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
        no_markers, positions = _strip_newline_markers(plain)
        lowered = no_markers.lower()
        stripped.append(lowered)
        tag_data.append(tags)
        newline_data.append(positions)

    # Batch and recase
    recased = []
    for start in range(0, len(stripped), batch_size):
        batch = stripped[start : start + batch_size]
        batch_results = recase_batch(batch, client, model)
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
                "LLM altered wording for event %d, falling back to sentence case. "
                "Input: %r | Output: %r",
                i,
                original_lowered[:100],
                result_text[:100],
            )
            result_text = to_sentence_case(original_lowered)

        result_text = _reinsert_newline_markers(result_text, positions)
        results.append(reinsert_ass_tags(result_text, tags))

    return results
