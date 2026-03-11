# subtitle-modifier

A CLI tool that converts ALL-UPPERCASE subtitle text to sentence case while preserving proper noun capitalization using spaCy NER. Supports SRT and ASS/SSA formats.

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Convert a single file (saves as movie_converted.srt alongside the original)
subtitle-modifier movie.srt

# Convert multiple files
subtitle-modifier *.srt

# Write output to a specific directory
subtitle-modifier movie.srt -o converted/

# Preview changes without writing files
subtitle-modifier movie.srt --dry-run

# Use a different spaCy model for better accuracy
subtitle-modifier movie.srt --model en_core_web_trf
```

## Options

| Flag | Description |
|---|---|
| `-o`, `--output` | Output directory (default: save alongside input with `_converted` suffix) |
| `--model` | spaCy model for NER (default: `en_core_web_sm`) |
| `--dry-run` | Preview changes without writing files |
| `--benchmark MODEL [MODEL ...]` | Benchmark spaCy models and print a speed comparison |

## How it works

1. Parses the subtitle file (SRT or ASS/SSA via pysubs2)
2. Strips any ASS override tags to get plain text
3. Lowercases the text and runs spaCy NER to detect proper nouns (people, places, organizations, etc.)
4. Applies sentence-case capitalization (first letter of each sentence, standalone "I")
5. Re-capitalizes detected proper nouns
6. Reinserts ASS tags and writes the output file

Only casing is changed — wording is never modified.

## Benchmarking

Compare processing speed across different spaCy models:

```bash
# Benchmark a single model
subtitle-modifier movie.srt --benchmark en_core_web_sm

# Compare multiple models
subtitle-modifier movie.srt --benchmark en_core_web_sm en_core_web_md en_core_web_trf
```

This prints a table showing model load time, processing time, and throughput (subtitles/sec) for each model. No output files are written in benchmark mode.

## Limitations

- English-only (uses English spaCy models)
- NER may miss some proper nouns or acronyms depending on context
- For better accuracy, use a transformer-based model (`en_core_web_trf`) at the cost of speed
