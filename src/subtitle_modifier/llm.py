"""LLM-based subtitle recasing via OpenAI-compatible API."""

import logging
import re

from .converter import reinsert_ass_tags, strip_ass_tags, to_sentence_case

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Re-capitalize these subtitles. ONLY change the case of letters (uppercase/lowercase). \
Do not add, remove, or change any characters including spaces, punctuation, and apostrophes. \
The output must be identical to the input except for letter casing.

Rules:
- Sentence case: capitalize first letter of each sentence
- Capitalize proper nouns, names, places, and abbreviations (e.g. D.E.A., U.S.)
- \\N is a visual line break, NOT a sentence boundary. Do not capitalize after \\N unless it starts a new sentence or is a proper noun
- Return ONLY the numbered lines in the same format"""

_LINE_RE = re.compile(r"^(\d+):\s?(.*)", re.MULTILINE)

# Markdown code fences that LLMs sometimes wrap responses in
_CODE_FENCE_RE = re.compile(r"^```\w*\n?(.*?)```\s*$", re.DOTALL)


def _lowercase_preserving_markers(text: str) -> str:
    """Lowercase text while preserving \\N markers (ASS line breaks).

    Plain .lower() would turn \\N into \\n, which LLMs may misinterpret
    as a newline escape.
    """
    return "\\N".join(part.lower() for part in text.split("\\N"))


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
        texts: Pre-processed (tag-stripped, lowercased) subtitle texts.
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

    For each text: strip ASS tags -> lowercase -> batch LLM recase ->
    validate wording invariant -> reinsert ASS tags.

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

    # Pre-process: strip tags and lowercase (preserving \N markers)
    stripped = []
    tag_data = []
    for text in texts:
        plain, tags = strip_ass_tags(text)
        lowered = _lowercase_preserving_markers(plain)
        stripped.append(lowered)
        tag_data.append(tags)

    # Batch and recase
    recased = []
    for start in range(0, len(stripped), batch_size):
        batch = stripped[start : start + batch_size]
        batch_results = recase_batch(batch, client, model)
        recased.extend(batch_results)

    # Validate wording invariant and reinsert tags
    results = []
    for i, (original_lowered, result_text, tags) in enumerate(
        zip(stripped, recased, tag_data)
    ):
        # Normalize \N casing for comparison (LLM might return \n or \N)
        orig_normalized = original_lowered.replace("\\N", "\\n")
        result_normalized = result_text.replace("\\N", "\\n")

        if result_normalized.lower() != orig_normalized.lower():
            logger.warning(
                "LLM altered wording for event %d, falling back to sentence case. "
                "Input: %r | Output: %r",
                i,
                original_lowered[:100],
                result_text[:100],
            )
            result_text = to_sentence_case(original_lowered)

        results.append(reinsert_ass_tags(result_text, tags))

    return results
