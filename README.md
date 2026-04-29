# subtitle-modifier

A CLI tool that converts ALL-UPPERCASE subtitle text to sentence case while preserving proper noun capitalization. Supports SRT and ASS/SSA formats. Two backends available: spaCy NER (local, fast) or any OpenAI-compatible LLM (more accurate proper nouns and abbreviations).

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

For LLM mode (optional):

```bash
pip install -e ".[llm]"
```

For Claude mode (optional):

```bash
pip install -e ".[claude]"
```

On Windows, if `pip` is not on your PATH, use `python -m pip` instead:

```bash
python -m pip install -e .
```

## Usage

### spaCy mode (default)

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

### LLM mode

Uses an OpenAI-compatible API (Ollama, OpenAI, Venice, etc.) instead of spaCy for recasing. Better at handling proper nouns, abbreviations (D.E.A., POTUS), and context-dependent capitalization.

```bash
# Use with Ollama (default URL: http://localhost:11434/v1)
subtitle-modifier movie.ass --llm --llm-model llama3.2

# Use with a remote API
subtitle-modifier movie.ass --llm --llm-model gpt-4o \
  --llm-url https://api.openai.com/v1 \
  --llm-api-key sk-...

# Adjust batch size (default: 50 subtitles per API call)
subtitle-modifier movie.ass --llm --llm-model llama3.2 --llm-batch-size 25

# Debug LLM responses
subtitle-modifier movie.ass --llm --llm-model llama3.2 --verbose --dry-run
```

### Claude mode

Uses the native Anthropic SDK for recasing. Supports Haiku (default and recommended), Sonnet, and Opus models. `--claude` is faster and more direct than the generic OpenAI-compatible `--llm` path for Anthropic API access.

```bash
# Use Claude with the default Haiku model
ANTHROPIC_API_KEY=sk-... subtitle-modifier subs.ass --claude

# Use a different Claude model
subtitle-modifier subs.ass --claude --claude-model claude-sonnet-4-20250514
```

> `--claude` uses the native Anthropic SDK and surfaces typed Anthropic errors. `--llm` is the generic OpenAI-compatible path, which can also reach Anthropic via Anthropic's OpenAI-compatible endpoint but loses Anthropic-native error typing.

## Options

| Flag | Description |
|---|---|
| `-o`, `--output` | Output directory (default: save alongside input with `_converted` suffix) |
| `--model` | spaCy model for NER (default: `en_core_web_sm`) |
| `--dry-run` | Preview changes without writing files |
| `-v`, `--verbose` | Show debug logging (e.g. raw LLM responses on parse failures) |
| `--benchmark MODEL [MODEL ...]` | Benchmark spaCy models and print a speed comparison |
| `--llm` | Enable LLM mode (skips spaCy entirely) |
| `--llm-model` | LLM model name (required with `--llm`) |
| `--llm-url` | API base URL (default: `http://localhost:11434/v1` for Ollama) |
| `--llm-api-key` | API key (default: `OPENAI_API_KEY` env var, falls back to `ollama`) |
| `--llm-batch-size` | Subtitles per API call (default: 50) |
| `--claude` | Enable Claude mode via native Anthropic SDK (skips spaCy entirely) |
| `--claude-model` | Claude model name (default: `claude-haiku-4-5`) |
| `--claude-api-key` | Anthropic API key (default: `ANTHROPIC_API_KEY` env var) |
| `--claude-batch-size` | Subtitles per Claude API call (default: 50) |

## How it works

### spaCy mode

1. Parses the subtitle file (SRT or ASS/SSA via pysubs2)
2. Strips any ASS override tags to get plain text
3. Lowercases the text and runs spaCy NER to detect proper nouns (people, places, organizations, etc.)
4. Applies sentence-case capitalization (first letter of each sentence, standalone "I")
5. Re-capitalizes detected proper nouns
6. Reinserts ASS tags and writes the output file

### LLM mode

1. Parses the subtitle file and strips ASS override tags
2. Lowercases text (preserving `\N` markers) and batches subtitles
3. Sends each batch to the LLM as numbered lines for recasing
4. Validates the wording invariant (only casing may change) — falls back to sentence case on violation
5. Reinserts ASS tags and writes the output file

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

- English-only (uses English spaCy models / English prompts)
- **spaCy mode**: NER may miss some proper nouns or acronyms depending on context. Use `en_core_web_trf` for better accuracy at the cost of speed.
- **LLM mode**: Quality depends on the model. Some models may mangle `\N` markers or make unwanted punctuation changes — the wording invariant check catches these and falls back to sentence case.
