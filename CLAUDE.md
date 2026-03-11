# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable mode)
pip install -e .
python -m spacy download en_core_web_sm

# Install with LLM support
pip install -e ".[llm]"

# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or class
pytest tests/test_converter.py
pytest tests/test_llm.py

# Run the CLI (spaCy mode)
subtitle-modifier <file.srt>

# Run the CLI (LLM mode)
subtitle-modifier <file.ass> --llm --llm-model <model-name>
```

## Architecture

Python CLI tool that converts ALL-UPPERCASE subtitle text to sentence case while preserving proper noun capitalization. Two backends: spaCy NER (local) or OpenAI-compatible LLM API. Supports SRT and ASS/SSA formats.

**Package layout:** `src/subtitle_modifier/` (src-layout, installed via setuptools)

### Module responsibilities

- **cli.py** — Entry point (`subtitle-modifier` command). Parses args, expands globs, routes to benchmark/spaCy/LLM mode. Heavy imports (spaCy, openai) are lazy-loaded.
- **converter.py** — spaCy conversion logic. The `convert_text()` pipeline: strip ASS tags → lowercase → run spaCy NER → apply sentence case → re-capitalize detected proper nouns → reinsert ASS tags. Also exports shared utilities (`strip_ass_tags`, `reinsert_ass_tags`, `to_sentence_case`) used by both backends.
- **llm.py** — LLM conversion logic. Batches subtitle texts into numbered prompts, sends to OpenAI-compatible API, parses numbered responses. Validates wording invariant per event, falls back to `to_sentence_case()` on violation.
- **subtitle_io.py** — File I/O via pysubs2. `process_file()` accepts an optional `convert_fn` for batch conversion (LLM mode) or falls back to per-event spaCy conversion.
- **benchmark.py** — Benchmarks spaCy model speed against subtitle files.

### Key design decisions

- spaCy and openai are lazy-imported in `cli.py` to keep `--help` fast
- NER runs on **lowercased** text (not title-cased) to avoid false positives like "John Went" being detected as a single PERSON entity
- ASS override tags (`{...}`) are stripped before processing and reinserted at original character positions after casing changes
- `\\N` (ASS newline literal) is handled by splitting before sentence-case logic and rejoining after
- The invariant "only casing changes, never wording" is enforced by tests comparing `.lower()` of input and output
- **LLM mode**: `\\N` markers are preserved during lowercasing (plain `.lower()` would turn `\\N` into `\\n`). The LLM prompt explicitly instructs that `\\N` is a visual line break, not a sentence boundary. Wording invariant normalizes `\\N`/`\\n` casing before comparison.
- **LLM batching**: default 50 events/batch for token efficiency. Parse failures retry once, then fall back to sentence case.

### Dependencies

- **pysubs2** — subtitle parsing/serialization (SRT, ASS/SSA)
- **spaCy** — NLP/NER for proper noun detection (requires a model like `en_core_web_sm`)
- **openai** (optional) — client for OpenAI-compatible APIs (LLM mode)
