# Claude API Support — Design

**Date:** 2026-04-29
**Branch:** `claude-api-support`
**Status:** Draft for review

## Summary

Add native Anthropic SDK support to `subtitle-modifier` as a third first-class
recasing backend, alongside the existing spaCy NER mode and OpenAI-compatible
LLM mode. New `--claude` CLI flag and `claude.py` module. Default model is
`claude-haiku-4-5`. Prompt caching, extended thinking, and streaming are
intentionally out of scope for this version.

## Motivation

The project today supports two recasing backends:

1. **spaCy NER** (default) — local, fast, but proper-noun detection is bounded
   by the model's training data.
2. **OpenAI-compatible LLM** (`--llm`) — generic OpenAI client, works against
   Ollama, vLLM, OpenAI itself, or any compatible endpoint.

Anthropic's Claude API is reachable through option 2 via Anthropic's
OpenAI-compatible endpoint, but that path forfeits Anthropic-native features
(prompt caching, structured Messages API, typed errors, vision, extended
thinking) and uses a generic SDK that doesn't surface Anthropic-specific
errors well. A native integration positions the project for those features
while staying opt-in: spaCy and `--llm` are unchanged.

This first version covers the core integration; prompt caching and other
Anthropic-native features are deferred (see "Future Work").

## Non-Goals

- Prompt caching. The system prompt is ~400 tokens — below the 1024-token
  minimum for Sonnet/Opus and 2048-token minimum for Haiku. Caching is a
  follow-up if we later expand the prompt with a few-shot block.
- Streaming responses. Output for a 50-event batch is well under
  `max_tokens=4096`; non-streaming is simpler and avoids HTTP-timeout
  edge cases at small response sizes.
- Extended thinking. The casing task is constrained and pattern-matching;
  thinking would add latency and cost for no quality gain.
- Refactoring the existing `--llm` code path. Out of scope to avoid risk
  to the working OpenAI integration.
- Replacing or deprecating the `--llm` flag.

## Architecture

```
src/subtitle_modifier/
├── llm.py              ← unchanged. OpenAI-compatible path.
│                         Holds shared constants & helpers:
│                         _SYSTEM_PROMPT, _LINE_RE, _CODE_FENCE_RE,
│                         _build_prompt, _strip_code_fences, _parse_response
└── claude.py           ← NEW. Native Anthropic SDK path.
                          Imports the 6 shared symbols above from .llm.
```

The Claude module imports the system prompt, line-parsing regex, and
response-parsing helpers from `llm.py` directly. The helpers are
underscore-prefixed (private by convention) but importing them across
sibling modules in the same package is normal Python; no rename is needed.

No changes to `converter.py`, `subtitle_io.py`, or `benchmark.py`. The Claude
path enters via the same `convert_fn` hook in `cli.py` that `--llm` uses
today — `process_file()` doesn't care which backend produced the function.

### Why not extract a shared pipeline?

The most code-clean design would extract a backend-agnostic
`convert_texts_with_backend(texts, recase_fn, batch_size)` function that
both `--llm` and `--claude` call. We're explicitly not doing that here:

- The existing OpenAI path works and is well-tested. Refactoring it bundles
  risk into a feature delivery.
- The duplication is small (one function body, ~25 lines).
- Once both backends exist and are stable, the refactor becomes a
  straightforward, isolated cleanup.

Flagged in "Future Work" below.

## CLI Surface

New flag group in `cli.py`, mirroring the existing `--llm-*` group:

```
Claude mode (native Anthropic SDK):
  --claude                   Enable Claude mode.
  --claude-model MODEL       Claude model ID (default: claude-haiku-4-5)
  --claude-api-key KEY       Anthropic API key
                             (default: ANTHROPIC_API_KEY env var)
  --claude-batch-size N      Subtitle events per API call (default: 50)
```

### Validation rules added to `cli.py`

1. `--claude` and `--llm` are mutually exclusive — error before processing
   if both are provided.
2. If `--claude` is set with no `--claude-api-key` and no `ANTHROPIC_API_KEY`
   in the environment: error out before any API call.
3. Lazy-import guard: if `--claude` and `import anthropic` fails, print
   `Install with: pip install 'subtitle-modifier[claude]'` and exit. Same
   pattern as the existing `openai` import guard.

### Excluded flags

- **No `--claude-max-tokens`.** Locked at 4096 internally. Output for a
  50-event batch is ~50 short lines, well under that. Add the flag later
  if any user actually hits the ceiling.
- **No `--claude-url`.** The Anthropic SDK reads `ANTHROPIC_BASE_URL` from
  the environment for users who need a proxy; not worth a CLI flag.

## The `claude.py` module

```python
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


def recase_batch_claude(texts: list[str], client, model: str) -> list[str]:
    """Send a batch of texts to Claude for recasing.

    Mirrors recase_batch() in llm.py: parse-fail retry once,
    then sentence-case fallback for the whole batch.
    """
    if not texts:
        return []

    prompt = _build_prompt(texts)

    def _call() -> str:
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b.text for b in response.content if b.type == "text")

    response_text = _call()
    parsed = _parse_response(response_text, len(texts))
    if parsed is not None:
        return parsed

    logger.warning("Claude response parse failed, retrying batch of %d", len(texts))
    response_text = _call()
    parsed = _parse_response(response_text, len(texts))
    if parsed is not None:
        return parsed

    logger.warning("Claude response parse failed twice, falling back to sentence case")
    return [to_sentence_case(t) for t in texts]


def convert_texts_claude(
    texts: list[str],
    client,
    model: str,
    batch_size: int = 50,
) -> list[str]:
    """Full Claude conversion pipeline.

    Mirrors convert_texts_llm(): for each text, strip ASS tags ->
    strip \\N markers -> lowercase -> batch -> Claude recase ->
    validate wording invariant -> reinsert \\N markers -> reinsert ASS tags.
    """
    # Body is a copy of convert_texts_llm's body, with one line different:
    # the inner call switches from recase_batch(...) to recase_batch_claude(...).
```

The body of `convert_texts_claude` is a copy of `convert_texts_llm`'s body
with one substitution: `recase_batch(...)` → `recase_batch_claude(...)`.
This is the duplication that "Why not extract a shared pipeline?" above
addresses.

### Anthropic-API specifics

- **`system=`** is a top-level parameter on `client.messages.create()`,
  not a `{"role": "system"}` message. This is the most visible API
  difference vs. OpenAI.
- **Response shape:** `response.content` is a list of typed content blocks.
  We iterate and pick the first `text` block. For a `max_tokens=4096`
  non-streaming text-only request, there is exactly one text block, but
  the iteration is the SDK-idiomatic shape and is robust if Anthropic
  later interleaves other block types into a similar response.
- **`temperature=0`** for deterministic recasing on Haiku 4.5 (and Sonnet/Opus
  4.6). **This will 400 on Opus 4.7**, which removed sampling parameters
  entirely. Opus is overkill for this task; users who deliberately pass
  `--claude-model claude-opus-4-7` will get the API error and can switch
  models. Documented in `CLAUDE.md`.
- **No `thinking`, no `effort`.** Adaptive thinking is not supported on
  Haiku 4.5; the casing task doesn't benefit from it. Not setting
  `thinking` means the request runs in the model's default mode.
- **Built-in SDK retries.** The `anthropic` SDK auto-retries connection
  errors, 408, 409, 429, and ≥500 with exponential backoff (default
  `max_retries=2`). We rely on this — no custom retry logic.

### Error handling

Match the existing `--llm` path exactly. The Anthropic SDK raises typed
exceptions (`anthropic.APIError`, `anthropic.RateLimitError`, etc.); we
let them propagate to `cli.py`, where the existing per-file
`except Exception as e:` around `process_file()` catches them, prints the
error, and continues to the next input file. This is the behavior the
user explicitly wants (parity with `--llm`).

Parse failures (LLM returned text that doesn't match the numbered-line
format) retry once, then fall back to `to_sentence_case()` for the entire
batch — identical to `recase_batch()` in `llm.py`.

Wording-invariant violations (LLM altered characters beyond casing) are
caught in `convert_texts_claude` per-event and fall back to
`to_sentence_case()` for that single event. Identical to `convert_texts_llm`.

## CLI Routing in `cli.py`

The Claude branch is added next to the existing LLM branch:

```python
if args.llm and args.claude:
    print("Error: --llm and --claude are mutually exclusive.", file=sys.stderr)
    sys.exit(1)

if args.claude:
    try:
        import anthropic
    except ImportError:
        print(
            "Error: anthropic package not installed. "
            "Install it with: pip install 'subtitle-modifier[claude]'",
            file=sys.stderr,
        )
        sys.exit(1)

    import os
    from .claude import convert_texts_claude

    api_key = args.claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: no Anthropic API key. Pass --claude-api-key or set "
            "ANTHROPIC_API_KEY in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    def convert_fn(texts, _client=client, _model=args.claude_model,
                   _bs=args.claude_batch_size):
        return convert_texts_claude(texts, _client, _model, batch_size=_bs)

elif args.llm:
    # ... existing OpenAI branch unchanged ...
```

## Dependencies

`pyproject.toml`:

```toml
[project.optional-dependencies]
llm = ["openai>=1.0"]
claude = ["anthropic>=0.40"]
dev = ["pytest>=7.0"]
```

Install: `pip install -e ".[claude]"`. Mirrors the existing `[llm]` extra.

## Testing

Light tests per the brainstorming decision (Q6, option B): focus on the
Anthropic-specific code, trust the shared-pipeline coverage already in
`tests/test_llm.py`. New file: `tests/test_claude.py`.

### Test list

1. **`test_request_shape`** — assert `client.messages.create` is called with
   `model=<arg>`, `max_tokens=4096`, `temperature=0`,
   `system=_SYSTEM_PROMPT`, `messages=[{"role": "user", "content": <numbered prompt>}]`.
   This is the contract test for the Anthropic-specific call.
2. **`test_response_text_extraction`** — given a mocked response with
   `content=[TextBlock(type="text", text="1: Hello\n2: World")]`, verify
   the parser receives the text and `recase_batch_claude` returns
   `["Hello", "World"]`.
3. **`test_parse_failure_retries_once`** — first mocked call returns garbage,
   second returns valid; assert two `messages.create` calls and successful
   parse on the retry.
4. **`test_parse_failure_falls_back_to_sentence_case`** — both calls return
   garbage; assert fallback to `to_sentence_case()` and exactly two API
   calls (no infinite retry).
5. **`test_wording_invariant_violation_falls_back`** — Claude returns valid
   numbered format but altered wording (e.g. added an apostrophe); assert
   fallback to sentence case for that event.
6. **`test_ass_tags_and_newline_markers_preserved`** — input with an
   `{\an8}` ASS tag and a `\N` marker round-trips correctly through the
   full `convert_texts_claude` pipeline.

### Mock helper

A small helper in the test file:

```python
def make_mock_client(response_texts: list[str]):
    """Mock that returns response_texts in order, one per messages.create() call."""
    # builds a Mock with .messages.create returning objects shaped like
    # response.content = [SimpleNamespace(type="text", text=...)]
```

### Tests not written (covered by `test_llm.py`)

- Code-fence stripping
- Line-number parsing edge cases
- Multi-batch boundary correctness
- `to_sentence_case()` formatting itself
- `_SYSTEM_PROMPT` content correctness

These all live in `llm.py` and are exercised through the OpenAI mock client
in `test_llm.py`. Re-testing them through the Anthropic mock would
duplicate coverage without adding signal.

## Documentation

- **`CLAUDE.md`** — add a section under "Architecture / Module
  responsibilities" describing `claude.py` and its relationship to
  `llm.py` (shared symbols, mirror-not-merge approach). Add a "Key design
  decisions" bullet documenting the `temperature=0` / Opus-4.7-incompatibility
  note. Add a command-line example for `--claude` to the Commands section.
- **`README.md`** — add a usage example for `--claude` mode and a sentence
  about installing the `[claude]` extra.

## Future Work

Tracked here so we don't lose them — out of scope for this delivery.

1. **Extract a shared backend-agnostic pipeline.** Refactor
   `convert_texts_llm` and `convert_texts_claude` into a single
   `convert_texts_with_backend(texts, recase_fn, batch_size)` once both
   backends are stable. Mechanical change; no behavior shift.
2. **Prompt caching.** If the system prompt grows past 1024 tokens
   (e.g. by adding a few-shot block to improve quality on edge cases),
   add `cache_control={"type": "ephemeral"}` on `messages.create()`. With
   batch-mode use, every call after the first within 5 minutes pays
   ~10% of the system-prompt cost.
3. **Streaming with `.stream()` / `.get_final_message()`.** Useful only if
   `max_tokens` later climbs above ~16K; below that, non-streaming
   non-issue.

## Open Questions

None at design time. All Q&A decisions are encoded above.
