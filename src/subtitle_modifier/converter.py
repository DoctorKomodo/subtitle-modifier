"""Core sentence-case conversion logic with NER-based proper noun preservation."""

import re

_HONORIFICS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "vs", "etc",
    "gov", "sgt", "gen", "col", "rev",
})


def strip_ass_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Remove ASS override tags ({...}) from text.

    Returns the plain text and a list of (position_in_plain, tag_string) tuples
    so tags can be reinserted after casing changes.
    """
    tags = []
    plain_parts = []
    plain_len = 0
    last_end = 0

    for match in re.finditer(r"\{[^}]*\}", text):
        # Add the text before this tag
        before = text[last_end:match.start()]
        plain_parts.append(before)
        plain_len += len(before)
        # Record the tag and its position in the plain text
        tags.append((plain_len, match.group()))
        last_end = match.end()

    # Add remaining text after last tag
    plain_parts.append(text[last_end:])
    plain_text = "".join(plain_parts)
    return plain_text, tags


def reinsert_ass_tags(plain: str, tags: list[tuple[int, str]]) -> str:
    """Reinsert ASS override tags at their original positions in plain text."""
    if not tags:
        return plain

    result = []
    prev_pos = 0
    for pos, tag in tags:
        result.append(plain[prev_pos:pos])
        result.append(tag)
        prev_pos = pos
    result.append(plain[prev_pos:])
    return "".join(result)


def strip_newline_markers(text: str) -> tuple[str, list[int]]:
    """Replace \\N markers with a space, recording their character positions."""
    positions: list[int] = []
    parts: list[str] = []
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


def reinsert_newline_markers(text: str, positions: list[int]) -> str:
    """Reinsert \\N markers, replacing the space at each recorded position."""
    if not positions:
        return text
    result: list[str] = []
    prev = 0
    for pos in positions:
        result.append(text[prev:pos])
        result.append("\\N")
        prev = pos + 1  # skip the space that \N replaced
    result.append(text[prev:])
    return "".join(result)


def to_sentence_case(text: str) -> str:
    """Convert text to sentence case.

    Lowercases everything, then capitalizes:
    - First character of the text
    - First letter after sentence-ending punctuation (. ! ?)
    - Standalone "I" (the pronoun)
    """
    result = _capitalize_sentences(text.lower())

    # Capitalize standalone "I"
    result = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "I", result)

    return result


def _is_abbreviation_period(text: str, dot_pos: int) -> bool:
    """Check whether the period at dot_pos is part of an abbreviation.

    Two rules:
    1. Single-letter-dot: the character before the period is a single letter
       (preceded by a non-alpha character or the start of the string).
    2. Known honorific: the word before the period is in _HONORIFICS.
    """
    if dot_pos == 0:
        return False

    # Rule 1: single letter before the dot
    before = text[dot_pos - 1]
    if before.isalpha():
        if dot_pos - 1 == 0 or not text[dot_pos - 2].isalpha():
            return True

        # Rule 2: known honorific
        # Walk back to find the start of the word
        word_start = dot_pos - 1
        while word_start > 0 and text[word_start - 1].isalpha():
            word_start -= 1
        word = text[word_start:dot_pos]
        if word.lower() in _HONORIFICS:
            return True

    return False


def _capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of each sentence."""
    if not text:
        return text

    result = list(text)
    capitalize_next = True

    for i, ch in enumerate(result):
        if capitalize_next and ch.isalpha():
            result[i] = ch.upper()
            capitalize_next = False
        elif capitalize_next and ch.isdigit():
            capitalize_next = False
        elif ch in "!?":
            capitalize_next = True
        elif ch == ".":
            if not _is_abbreviation_period(text, i):
                capitalize_next = True

    return "".join(result)



def convert_text(text: str, nlp) -> str:
    """Full conversion pipeline for a single subtitle event's text.

    1. Strip ASS tags and \\N newline markers
    2. Run NER on lowercased text
    3. Apply sentence case
    4. Re-capitalize NER-detected proper nouns
    5. Reinsert \\N markers and ASS tags
    """
    if not text or not text.strip():
        return text

    plain, tags = strip_ass_tags(text)

    if not plain.strip():
        return text

    # Replace \N (ASS visual line break) with a space, recording positions
    # so markers can be reinserted after casing changes.
    plain, newline_positions = strip_newline_markers(plain)

    # Run NER on lowercased text. Lowercase avoids false positives
    # that title-casing causes (e.g. "John Went" as single PERSON).
    spacy_text = plain.lower()
    doc = nlp(spacy_text)

    # Collect character spans from NER entities with proper-noun labels only.
    _NER_LABELS = {
        "PERSON", "GPE", "ORG", "NORP", "FAC",
        "LOC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }
    proper_spans = set()
    for ent in doc.ents:
        if ent.label_ in _NER_LABELS:
            proper_spans.add((ent.start_char, ent.end_char))

    # Apply sentence case
    sentence_cased = to_sentence_case(plain)

    # Re-capitalize the proper noun spans detected by NER
    if proper_spans:
        chars = list(sentence_cased)
        for start, end in proper_spans:
            i = start
            cap_next = True
            while i < end and i < len(chars):
                if chars[i].isalpha():
                    if cap_next:
                        chars[i] = chars[i].upper()
                        cap_next = False
                elif chars[i].isspace() or chars[i] == "-":
                    cap_next = True
                i += 1
        sentence_cased = "".join(chars)

    # Restore \N markers and reinsert ASS tags
    sentence_cased = reinsert_newline_markers(sentence_cased, newline_positions)
    return reinsert_ass_tags(sentence_cased, tags)
