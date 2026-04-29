"""LLM-based subtitle recasing via OpenAI-compatible API."""

import logging
import re

from .converter import (
    reinsert_ass_tags,
    reinsert_newline_markers,
    strip_ass_tags,
    strip_newline_markers,
    to_sentence_case,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a subtitle capitalization tool. You receive numbered subtitle lines and \
return them with corrected letter casing ONLY.

## CORE CONSTRAINT
The ONLY changes you may make are converting lowercase letters to uppercase or \
uppercase letters to lowercase (a-z ↔ A-Z). This means:
- Do NOT add any characters, including apostrophes, hyphens, or accents
- Do NOT remove any characters, including spaces inside parentheses or between words
- Do NOT move, reorder, or substitute any characters
- If the input spells a word "wrong" (e.g. "Dewars" without an apostrophe), \
keep it exactly as "Dewars" — you are not a spellchecker
- This applies especially to **proper nouns and names**. If a name is spelled \
with extra, missing, or unusual letters compared to the conventional spelling \
(e.g. "Donnatella" instead of "Donatella", "Stefani" instead of "Stefanie", \
"Phillip" instead of "Philip"), keep the input spelling exactly as given. \
Only change which letters are upper vs. lower case.
- If the input has unusual spacing (e.g. "( siren )" with spaces inside parentheses), \
preserve those exact spaces

The character count of each output line MUST be identical to its input line. \
The only difference is which letters are upper vs. lower case.

## CAPITALIZATION RULES
1. Capitalize the first letter of each subtitle line.
2. Capitalize the first letter after sentence-ending punctuation (. ! ?) \
when followed by a space and a new word on the same line.
3. A colon (:) does NOT start a new sentence. Do not capitalize the word after \
a colon unless another rule requires it.
4. Capitalize proper nouns: personal names, place names, nationalities, languages, \
brand names.
5. Capitalize abbreviations/acronyms that are conventionally uppercase \
(e.g. FBI, U.S., DNA, TV).
6. Capitalize the pronoun "I" as a standalone word.
7. All other words should be lowercase.

## OUTPUT FORMAT
- Return ONLY the numbered lines, same order, same numbering.
- No commentary, no explanations, no extra lines.
- Output line count MUST equal input line count.

## EXAMPLES
Input:  1: man: two absolut martinis up, another dewars rocks.
Output: 1: Man: two Absolut martinis up, another Dewars rocks.

Input:  2: ( siren continues )
Output: 2: ( Siren continues )

Input:  3: he went to new york with the F.B.I. and met john.
Output: 3: He went to New York with the F.B.I. and met John.

Input:  4: donnatella moss, when did you start working for me?
Output: 4: Donnatella Moss, when did you start working for me?"""

_LINE_RE = re.compile(r"^(\d+):\s?(.*)", re.MULTILINE)

# Markdown code fences that LLMs sometimes wrap responses in
_CODE_FENCE_RE = re.compile(r"^```\w*\n?(.*?)```\s*$", re.DOTALL)



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
        no_markers, positions = strip_newline_markers(plain)
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

        result_text = reinsert_newline_markers(result_text, positions)
        results.append(reinsert_ass_tags(result_text, tags))

    return results
