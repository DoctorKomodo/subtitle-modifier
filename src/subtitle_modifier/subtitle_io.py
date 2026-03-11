"""Subtitle file I/O — load and save SRT/ASS/SSA files via pysubs2."""

from pathlib import Path

import pysubs2

from .converter import convert_text


def load_subtitles(path: str) -> pysubs2.SSAFile:
    """Load a subtitle file (SRT, ASS, or SSA)."""
    return pysubs2.load(path)


def save_subtitles(subs: pysubs2.SSAFile, path: str, format_: str | None = None) -> None:
    """Save subtitles to a file. Format is auto-detected from extension if not given."""
    subs.save(path, format_=format_)


def process_file(
    input_path: str,
    output_path: str | None,
    nlp=None,
    *,
    convert_fn=None,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Load a subtitle file, convert all events to sentence case, and save.

    Args:
        input_path: Path to the input subtitle file.
        output_path: Path to write the output. If None, adds '_converted' suffix.
        nlp: A loaded spaCy language model (used when convert_fn is None).
        convert_fn: A callable that takes a list of texts and returns a list of
            converted texts. If None, uses spaCy NER via convert_text().
        dry_run: If True, return changes but don't write the file.

    Returns:
        A list of (original_text, converted_text) tuples showing what changed.
    """
    input_p = Path(input_path)

    if output_path is None:
        output_path = str(input_p.with_stem(input_p.stem + "_converted"))

    subs = load_subtitles(input_path)

    # Build default convert_fn from nlp if not provided
    if convert_fn is None:
        def convert_fn(texts):
            return [convert_text(t, nlp) for t in texts]

    # Collect non-drawing event indices and texts
    event_indices = []
    event_texts = []
    for i, event in enumerate(subs.events):
        if not event.is_drawing:
            event_indices.append(i)
            event_texts.append(event.text)

    # Convert all at once (enables batching for LLM mode)
    converted_texts = convert_fn(event_texts)

    # Map results back and track changes
    changes = []
    for idx, original, converted in zip(event_indices, event_texts, converted_texts):
        if original != converted:
            changes.append((original, converted))
            subs.events[idx].text = converted

    if not dry_run:
        save_subtitles(subs, output_path)

    return changes
