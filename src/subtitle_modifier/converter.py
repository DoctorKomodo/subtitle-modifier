"""Core sentence-case conversion logic with NER-based proper noun preservation."""

import re


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


def to_sentence_case(text: str) -> str:
    """Convert text to sentence case.

    Lowercases everything, then capitalizes:
    - First character of the text
    - First letter after sentence-ending punctuation (. ! ?)
    - Standalone "I" (the pronoun)

    Handles \\N (ASS newline) as a line break.
    """
    # Split on \N first to preserve the literal string, then lowercase each line
    lines = text.split("\\N")
    lines = [line.lower() for line in lines]
    processed_lines = []

    for line in lines:
        line = _capitalize_sentences(line)
        processed_lines.append(line)

    result = "\\N".join(processed_lines)

    # Capitalize standalone "I"
    result = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "I", result)

    return result


def _capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of each sentence in a line."""
    if not text:
        return text

    result = list(text)
    capitalize_next = True

    for i, ch in enumerate(result):
        if capitalize_next and ch.isalpha():
            result[i] = ch.upper()
            capitalize_next = False
        elif ch in ".!?":
            capitalize_next = True

    return "".join(result)



def convert_text(text: str, nlp) -> str:
    """Full conversion pipeline for a single subtitle event's text.

    1. Strip ASS tags
    2. Run NER on title-cased text (spaCy needs proper casing for accuracy)
    3. Apply sentence case
    4. Re-capitalize NER-detected proper nouns
    5. Reinsert ASS tags
    """
    if not text or not text.strip():
        return text

    plain, tags = strip_ass_tags(text)

    if not plain.strip():
        return text

    # Run NER on lowercased text. While spaCy is trained on mixed-case,
    # lowercase still gives good entity detection and avoids false positives
    # that title-casing causes (e.g. "John Went" detected as single PERSON).
    lower_version = plain.lower()
    spacy_text = lower_version.replace("\\N", "\n")
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
        chars = list(sentence_cased.replace("\\N", "\n"))
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
        sentence_cased = "".join(chars).replace("\n", "\\N")

    # Reinsert any ASS tags
    return reinsert_ass_tags(sentence_cased, tags)
