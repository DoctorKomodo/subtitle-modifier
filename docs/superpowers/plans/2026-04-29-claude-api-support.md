# Claude API Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--claude` mode to `subtitle-modifier` — a third recasing backend using the native Anthropic SDK, alongside the existing spaCy and OpenAI-compatible (`--llm`) backends.

**Architecture:** Add a thin `claude.py` module that mirrors `llm.py`'s structure but calls Anthropic's `messages.create()` API. Reuse `_SYSTEM_PROMPT`, `_build_prompt`, and `_parse_response` from `llm.py` directly — no refactor of the existing OpenAI path. Wire a new `--claude` flag group into `cli.py`.

**Tech Stack:** Python 3.9+, `anthropic` SDK (>=0.40, optional via `[claude]` extra), pytest with `unittest.mock` for unit tests.

**Spec:** See `docs/superpowers/specs/2026-04-29-claude-api-support-design.md` for the full design rationale.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `claude = ["anthropic>=0.40"]` extra |
| `src/subtitle_modifier/claude.py` | **Create** | Anthropic-SDK-specific recasing pipeline. Imports `_SYSTEM_PROMPT`, `_build_prompt`, `_parse_response` from `.llm`. Exports `recase_batch_claude` and `convert_texts_claude`. |
| `src/subtitle_modifier/cli.py` | Modify | Add `--claude*` flag group; add routing branch with mutual-exclusion check vs. `--llm`; add lazy import + API-key validation. |
| `tests/test_claude.py` | **Create** | Light tests for the Anthropic-specific code (~6 tests). Mocks the Anthropic client; no real API calls. |
| `CLAUDE.md` | Modify | Add install line, runtime example, module section, dependencies bullet, design-decision bullet. |
| `README.md` | Modify | Add `--claude` usage example and `[claude]` extra install instruction. |

---

## Task 1: Add `[claude]` optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `claude` extra**

Open `pyproject.toml`. Find the `[project.optional-dependencies]` block:

```toml
[project.optional-dependencies]
llm = ["openai>=1.0"]
dev = ["pytest>=7.0"]
```

Edit it to:

```toml
[project.optional-dependencies]
llm = ["openai>=1.0"]
claude = ["anthropic>=0.40"]
dev = ["pytest>=7.0"]
```

- [ ] **Step 2: Install the extra to verify resolution**

Run: `pip install -e ".[claude]"`
Expected: pip resolves `anthropic` and any transitive deps (httpx, pydantic, etc.) without errors. If the `>=0.40` floor produces an SDK that does not accept `claude-haiku-4-5` as a model identifier, bump the floor — confirm by running step 3.

- [ ] **Step 3: Smoke-test the SDK import**

Run: `python -c "import anthropic; c = anthropic.Anthropic(api_key='dummy'); print(type(c).__name__)"`
Expected output: `Anthropic`

(This only verifies the SDK loads — it does not call the API.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Add [claude] optional dependency for native Anthropic SDK"
```

---

## Task 2: Create `claude.py` skeleton

**Files:**
- Create: `src/subtitle_modifier/claude.py`

This task creates the module shell so test-file imports succeed. The function bodies will be filled in by subsequent tasks.

- [ ] **Step 1: Create the file**

Create `src/subtitle_modifier/claude.py` with this content:

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


def recase_batch_claude(texts, client, model):
    """Send a batch of texts to Claude for recasing.

    Implementation in subsequent tasks.
    """
    raise NotImplementedError


def convert_texts_claude(texts, client, model, batch_size=50):
    """Full Claude conversion pipeline.

    Implementation in a subsequent task.
    """
    raise NotImplementedError
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from subtitle_modifier.claude import recase_batch_claude, convert_texts_claude, _SYSTEM_PROMPT, _MAX_TOKENS, _DEFAULT_MODEL; print('ok')"`
Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/subtitle_modifier/claude.py
git commit -m "Add claude.py skeleton with shared-symbol imports from llm.py"
```

---

## Task 3: Mock helper + `test_request_shape`

**Files:**
- Create: `tests/test_claude.py`
- Modify: `src/subtitle_modifier/claude.py`

This task introduces the Anthropic-client mock helper and the contract test for the request shape, then implements the minimal `recase_batch_claude` body needed to pass it. Other tests (parse failure, retry, fallback) are added in later tasks.

- [ ] **Step 1: Write the test file scaffold and `test_request_shape`**

Create `tests/test_claude.py`:

```python
"""Unit tests for the Claude (Anthropic SDK) conversion pipeline.

No real API calls — all tests use a mocked Anthropic client.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from subtitle_modifier.claude import (
    _MAX_TOKENS,
    convert_texts_claude,
    recase_batch_claude,
)
from subtitle_modifier.llm import _SYSTEM_PROMPT


def _make_mock_anthropic_client(response_specs):
    """Create a mock Anthropic client.

    response_specs: list of dicts, one per messages.create() call, each with:
        - "content": list of (type, text) tuples for content blocks, OR
        - "text": shorthand for content=[("text", text)]
        - "stop_reason": optional, defaults to "end_turn"

    The mock returns specs in order; raises if the test calls more times
    than specs were provided.
    """
    client = MagicMock()

    responses = []
    for spec in response_specs:
        if "text" in spec:
            blocks = [SimpleNamespace(type="text", text=spec["text"])]
        else:
            blocks = [SimpleNamespace(type=t, text=x) for (t, x) in spec["content"]]
        responses.append(SimpleNamespace(
            content=blocks,
            stop_reason=spec.get("stop_reason", "end_turn"),
        ))

    client.messages.create.side_effect = responses
    return client


class TestRequestShape:
    def test_request_shape(self):
        """Verify the call is made with the Anthropic-specific shape:
        system as a top-level kwarg (not a message), max_tokens=4096,
        temperature=0, messages contains only the user turn.
        """
        client = _make_mock_anthropic_client([
            {"text": "1: Hello\n2: World"},
        ])

        recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")

        client.messages.create.assert_called_once()
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == _MAX_TOKENS
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0
        assert kwargs["system"] == _SYSTEM_PROMPT
        assert kwargs["messages"] == [
            {"role": "user", "content": "1: hello\n2: world"}
        ]
        # Critical: system goes top-level, NOT as a role="system" message
        # (this is the most likely Anthropic-vs-OpenAI regression)
        assert not any(
            m.get("role") == "system" for m in kwargs["messages"]
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_claude.py::TestRequestShape::test_request_shape -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `recase_batch_claude` minimally to pass the test**

In `src/subtitle_modifier/claude.py`, replace the `recase_batch_claude` body:

```python
def recase_batch_claude(texts, client, model):
    """Send a batch of texts to Claude for recasing.

    Mirrors recase_batch() in llm.py: parse-fail retry once, then
    sentence-case fallback. (Retry and fallback added in later tasks.)
    """
    if not texts:
        return []

    prompt = _build_prompt(texts)

    def _call():
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), None
        )
        return text, response.stop_reason

    response_text, _stop_reason = _call()
    parsed = (
        _parse_response(response_text, len(texts))
        if response_text is not None
        else None
    )
    if parsed is not None:
        return parsed

    # Retry/fallback added in later tasks
    return [to_sentence_case(t) for t in texts]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_claude.py::TestRequestShape::test_request_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_claude.py src/subtitle_modifier/claude.py
git commit -m "Add request-shape contract test for Claude backend"
```

---

## Task 4: Response text extraction (3 variants)

**Files:**
- Modify: `tests/test_claude.py`

Per the spec, `next(..., None)` already handles the multi-block case correctly. The implementation from Task 3 is sufficient — we just need to assert it via tests.

- [ ] **Step 1: Write the three text-extraction variants**

Append to `tests/test_claude.py`:

```python
class TestResponseTextExtraction:
    def test_text_only_response(self):
        """Single text block — the canonical happy path."""
        client = _make_mock_anthropic_client([
            {"text": "1: Hello\n2: World"},
        ])
        result = recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")
        assert result == ["Hello", "World"]

    def test_multi_block_response_picks_first_text(self):
        """If Anthropic interleaves a non-text block (e.g. thinking), pick the text."""
        client = _make_mock_anthropic_client([
            {"content": [
                ("thinking", "let me think..."),
                ("text", "1: Hello\n2: World"),
            ]},
        ])
        result = recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")
        assert result == ["Hello", "World"]

    def test_zero_text_blocks_does_not_raise(self):
        """No text block at all — should not raise StopIteration.

        Task 3's body falls back to sentence case after one failed text
        extraction. Task 5 will swap that for a retry-then-fallback path,
        at which point this same test exercises the retry. We assert the
        contract that matters here: the function does not raise, and it
        returns a list of the right length.

        After Task 5, a stronger version of this test
        (test_zero_text_blocks_uses_retry_path) asserts the retry happens.
        """
        client = _make_mock_anthropic_client([
            {"content": []},
            {"text": "1: Hello\n2: World"},
        ])
        result = recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")
        assert len(result) == 2
        # Both the Task 3 fallback and the post-Task-5 retry produce
        # ["Hello", "World"] here — the fallback because to_sentence_case
        # is called per-element on each lowercase input, and the retry
        # because the second mocked call returns the parsed numbered text.
        assert result == ["Hello", "World"]
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/test_claude.py::TestResponseTextExtraction -v`
Expected: 3 passed. Task 3's body falls back to `to_sentence_case` per element when text extraction returns `None`, producing `["Hello", "World"]` for the third test (each input string starts a fresh sentence-case operation, so each gets its first letter capitalized).

If the third test fails with `StopIteration` or `AttributeError: 'NoneType' object has no attribute 'strip'`, the implementation is wrong — go back to Task 3 and verify both the `next(..., None)` guard and the `if response_text is not None` guard are in place.

- [ ] **Step 3: Run the full test file to make sure nothing regressed**

Run: `pytest tests/test_claude.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_claude.py
git commit -m "Test response-text extraction across text-only, multi-block, and zero-block cases"
```

---

## Task 5: Parse-failure retry once

**Files:**
- Modify: `tests/test_claude.py`
- Modify: `src/subtitle_modifier/claude.py`

- [ ] **Step 1: Write the retry test**

Append to `tests/test_claude.py`:

```python
class TestParseFailureRetry:
    def test_parse_failure_retries_once_and_succeeds(self):
        """First call returns garbage; second call returns valid output."""
        client = _make_mock_anthropic_client([
            {"text": "garbage that won't parse"},
            {"text": "1: Hello\n2: World"},
        ])
        result = recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")
        assert result == ["Hello", "World"]
        assert client.messages.create.call_count == 2
```

- [ ] **Step 2: Run the test to see what happens**

Run: `pytest tests/test_claude.py::TestParseFailureRetry::test_parse_failure_retries_once_and_succeeds -v`
Expected: FAIL — Task 3's body falls back to sentence case after one failed parse, so result will be `["Hello", "world"]` and `call_count` will be 1.

- [ ] **Step 3: Add the retry logic to `recase_batch_claude`**

In `src/subtitle_modifier/claude.py`, replace the `recase_batch_claude` body with the version including retry:

```python
def recase_batch_claude(texts, client, model):
    """Send a batch of texts to Claude for recasing.

    Mirrors recase_batch() in llm.py: parse-fail retry once, then
    sentence-case fallback for the whole batch.
    """
    if not texts:
        return []

    prompt = _build_prompt(texts)

    def _call():
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), None
        )
        return text, response.stop_reason

    response_text, stop_reason = _call()
    parsed = (
        _parse_response(response_text, len(texts))
        if response_text is not None
        else None
    )
    if parsed is not None:
        return parsed

    logger.warning(
        "Claude response parse failed (stop_reason=%s), retrying batch of %d",
        stop_reason, len(texts),
    )
    response_text, stop_reason = _call()
    parsed = (
        _parse_response(response_text, len(texts))
        if response_text is not None
        else None
    )
    if parsed is not None:
        return parsed

    logger.warning(
        "Claude response parse failed twice (stop_reason=%s), falling back to sentence case",
        stop_reason,
    )
    return [to_sentence_case(t) for t in texts]
```

- [ ] **Step 4: Run the retry test to verify it passes**

Run: `pytest tests/test_claude.py::TestParseFailureRetry -v`
Expected: PASS

- [ ] **Step 5: Add a stronger zero-text-blocks-via-retry test**

The test from Task 4 (`test_zero_text_blocks_does_not_raise`) returns the
same value (`["Hello", "World"]`) under both Task 3's fallback and Task 5's
retry, so it cannot distinguish the two paths. Add a sibling test that
asserts the retry path is genuinely exercised by checking `call_count`:

Append to `tests/test_claude.py` inside the existing
`TestResponseTextExtraction` class:

```python
    def test_zero_text_blocks_uses_retry_path(self):
        """After Task 5's retry lands, a zero-text-blocks first response
        should trigger a retry rather than going straight to fallback.
        Distinguishes paths via call_count (which the does-not-raise test
        cannot do)."""
        client = _make_mock_anthropic_client([
            {"content": []},
            {"text": "1: Hello\n2: World"},
        ])
        result = recase_batch_claude(["hello", "world"], client, "claude-haiku-4-5")
        assert result == ["Hello", "World"]
        assert client.messages.create.call_count == 2
```

- [ ] **Step 6: Run the full file to confirm the retry path is wired**

Run: `pytest tests/test_claude.py -v`
Expected: 6 passed. The new
`TestResponseTextExtraction::test_zero_text_blocks_uses_retry_path`
confirms two API calls were made (proving the retry path is active);
the rest of the suite remains green.

- [ ] **Step 7: Commit**

```bash
git add tests/test_claude.py src/subtitle_modifier/claude.py
git commit -m "Add parse-failure retry to Claude backend"
```

---

## Task 6: Fallback to sentence case after two failures

**Files:**
- Modify: `tests/test_claude.py`

The fallback behavior is already implemented (Task 5's body covers it). This task adds the explicit test.

- [ ] **Step 1: Write the fallback test**

Append to `tests/test_claude.py`:

```python
class TestParseFailureFallback:
    def test_two_parse_failures_fall_back_to_sentence_case(self):
        """Both calls return garbage; assert sentence-case fallback and
        exactly 2 API calls (no infinite retry)."""
        client = _make_mock_anthropic_client([
            {"text": "garbage one"},
            {"text": "garbage two"},
        ])
        result = recase_batch_claude(
            ["hello world", "foo bar"], client, "claude-haiku-4-5"
        )
        assert result == ["Hello world", "Foo bar"]
        assert client.messages.create.call_count == 2
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_claude.py::TestParseFailureFallback -v`
Expected: PASS (the implementation from Task 5 already handles this).

- [ ] **Step 3: Commit**

```bash
git add tests/test_claude.py
git commit -m "Test sentence-case fallback after two Claude parse failures"
```

---

## Task 7: Implement `convert_texts_claude` + full pipeline tests

**Files:**
- Modify: `src/subtitle_modifier/claude.py`
- Modify: `tests/test_claude.py`

This task implements the full pipeline (`convert_texts_claude`) by **literal copy** of `convert_texts_llm`'s body, with the inner `recase_batch` call swapped for `recase_batch_claude`. Then adds the wording-invariant and ASS-tag-preservation tests.

- [ ] **Step 1: Write the wording-invariant test**

Append to `tests/test_claude.py`:

```python
class TestWordingInvariant:
    def test_altered_wording_falls_back_to_sentence_case_per_event(self):
        """Claude returns a valid numbered format but adds a character
        (e.g. an apostrophe) — that single event falls back to sentence
        case while the others pass through unchanged.
        """
        client = _make_mock_anthropic_client([
            # Event 1 is fine; event 2 has an extra apostrophe that
            # violates the casing-only invariant.
            {"text": "1: Hello world\n2: It's foo"},
        ])
        result = convert_texts_claude(
            ["hello world", "its foo"], client, "claude-haiku-4-5"
        )
        # Event 1: passes through Claude's casing
        assert result[0] == "Hello world"
        # Event 2: wording was altered (apostrophe added), so falls back
        # to to_sentence_case("its foo") -> "Its foo"
        assert result[1] == "Its foo"
```

- [ ] **Step 2: Write the ASS-tag and `\N` preservation test**

Append to `tests/test_claude.py`:

```python
class TestPipelinePreservation:
    def test_ass_tags_and_newline_markers_round_trip(self):
        """Input with an ASS override tag and a \\N marker should
        round-trip through the full pipeline: Claude sees the lowercased
        plain text without tags or markers, and the output reinstates
        them at the correct positions.
        """
        client = _make_mock_anthropic_client([
            # Claude sees: "hello world\nfoo" lowercased with \N replaced
            # by space at strip time. The mock returns Claude-cased text.
            {"text": "1: Hello world foo"},
        ])
        result = convert_texts_claude(
            [r"{\an8}hello world\Nfoo"], client, "claude-haiku-4-5"
        )
        # Tag at start preserved; \N marker reinstated where the space was
        assert result == [r"{\an8}Hello world\Nfoo"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_claude.py::TestWordingInvariant tests/test_claude.py::TestPipelinePreservation -v`
Expected: FAIL with `NotImplementedError` (from `convert_texts_claude`).

- [ ] **Step 4: Implement `convert_texts_claude` as a literal copy of `convert_texts_llm`**

In `src/subtitle_modifier/claude.py`:

a. Add the imports needed for the pipeline at the top of the file (under the existing imports):

```python
from .converter import (
    reinsert_ass_tags,
    reinsert_newline_markers,
    strip_ass_tags,
    strip_newline_markers,
)
```

b. Replace the stub `convert_texts_claude` with the full body. **This is a literal copy of `convert_texts_llm` from `src/subtitle_modifier/llm.py:166-230` with `recase_batch(...)` replaced by `recase_batch_claude(...)`** — keep every step (including the `rstrip()` at line 210 of `llm.py`) verbatim:

```python
def convert_texts_claude(texts, client, model, batch_size=50):
    """Full Claude conversion pipeline for a list of subtitle event texts.

    For each text: strip ASS tags -> strip \\N markers -> lowercase ->
    batch Claude recase -> rstrip trailing whitespace -> validate wording
    invariant -> reinsert \\N markers -> reinsert ASS tags.

    Args:
        texts: Raw subtitle event texts (may contain ASS tags).
        client: An anthropic.Anthropic client instance.
        model: Claude model identifier.
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

    # Batch and recase via Claude
    recased = []
    for start in range(0, len(stripped), batch_size):
        batch = stripped[start : start + batch_size]
        batch_results = recase_batch_claude(batch, client, model)
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
                "Claude altered wording for event %d, falling back to sentence case. "
                "Input: %r | Output: %r",
                i,
                original_lowered[:100],
                result_text[:100],
            )
            result_text = to_sentence_case(original_lowered)

        result_text = reinsert_newline_markers(result_text, positions)
        results.append(reinsert_ass_tags(result_text, tags))

    return results
```

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_claude.py::TestWordingInvariant tests/test_claude.py::TestPipelinePreservation -v`
Expected: both PASS.

- [ ] **Step 6: Run the full Claude test file**

Run: `pytest tests/test_claude.py -v`
Expected: all 7 tests PASS (request-shape, 3 text-extraction, 1 retry, 1 fallback, 1 wording-invariant, 1 preservation).

- [ ] **Step 7: Run the full project test suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests PASS, including the existing `test_llm.py`, `test_converter.py`, etc.

- [ ] **Step 8: Commit**

```bash
git add src/subtitle_modifier/claude.py tests/test_claude.py
git commit -m "Implement convert_texts_claude pipeline with wording-invariant and ASS-tag preservation tests"
```

---

## Task 8: CLI integration

**Files:**
- Modify: `src/subtitle_modifier/cli.py`

This task adds the `--claude` flag group and the routing branch. There are no automated CLI tests in the existing project (matches the "light tests" decision from the brainstorm); verification is done via `--help` output and a no-args invocation that exits with the expected validation error.

- [ ] **Step 1: Add the `--claude*` argparse group**

In `src/subtitle_modifier/cli.py`, find the existing LLM mode argument group (starts around line 46 with `llm_group = parser.add_argument_group(...)`). After the `--llm-batch-size` argument is added (around line 71), add a new group:

```python
    # Claude mode arguments
    claude_group = parser.add_argument_group(
        "Claude mode",
        "Use the native Anthropic SDK for recasing.",
    )
    claude_group.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude mode (uses the native Anthropic SDK).",
    )
    claude_group.add_argument(
        "--claude-model",
        default="claude-haiku-4-5",
        help="Claude model ID (default: claude-haiku-4-5).",
    )
    claude_group.add_argument(
        "--claude-api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var).",
    )
    claude_group.add_argument(
        "--claude-batch-size",
        type=int,
        default=50,
        help="Number of subtitle events per Claude API call (default: 50).",
    )
```

- [ ] **Step 2: Add the mutual-exclusion check**

In `src/subtitle_modifier/cli.py`, find the spot just after `args = parser.parse_args(argv)` (around line 73) and the verbose-logging block. Before any backend setup, add:

```python
    if args.llm and args.claude:
        print("Error: --llm and --claude are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 3: Add the Claude routing branch**

Find the `if args.llm:` block (around line 105). Replace the `if args.llm: ... else:` structure with `if args.claude: ... elif args.llm: ... else:`, where the new Claude branch comes first:

```python
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

        def convert_fn(
            texts,
            _client=client,
            _model=args.claude_model,
            _bs=args.claude_batch_size,
        ):
            return convert_texts_claude(texts, _client, _model, batch_size=_bs)

    elif args.llm:
        # ... existing OpenAI branch unchanged ...
```

(Keep the body of the existing `args.llm` branch and the `else:` spaCy branch exactly as they are — just change the `if` keyword on the OpenAI branch to `elif` and indent the Claude branch in front.)

- [ ] **Step 4: Verify `--help` shows the new flags**

Run: `subtitle-modifier --help`
Expected output (excerpt):
```
Claude mode:
  Use the native Anthropic SDK for recasing.

  --claude              Enable Claude mode (uses the native Anthropic SDK).
  --claude-model CLAUDE_MODEL
                        Claude model ID (default: claude-haiku-4-5).
  --claude-api-key CLAUDE_API_KEY
                        Anthropic API key (default: ANTHROPIC_API_KEY env var).
  --claude-batch-size CLAUDE_BATCH_SIZE
                        Number of subtitle events per Claude API call (default: 50).
```

- [ ] **Step 5: Verify the mutual-exclusion check**

Create a tiny dummy SRT file for the test:

```bash
printf "1\n00:00:01,000 --> 00:00:02,000\nHELLO\n" > /tmp/_test.srt
```

Run: `subtitle-modifier --llm --llm-model foo --claude /tmp/_test.srt`
Expected: stderr contains `Error: --llm and --claude are mutually exclusive.` and exit code 1.

Run: `echo $?`
Expected: `1`

- [ ] **Step 6: Verify the missing-API-key check**

Run: `ANTHROPIC_API_KEY= subtitle-modifier --claude /tmp/_test.srt`
Expected: stderr contains `Error: no Anthropic API key.` and exit code 1.

Run: `echo $?`
Expected: `1`

- [ ] **Step 7: Verify the dispatch path picks the Claude backend (no network)**

We don't have CLI integration tests, so do a quick in-process check that
passing `--claude` results in a `convert_fn` derived from
`convert_texts_claude` (rather than the spaCy or `--llm` paths). Run:

```bash
ANTHROPIC_API_KEY=sk-fake python -c "
import sys
from unittest.mock import patch, MagicMock

# Mock anthropic.Anthropic so we don't construct a real client.
fake_anthropic = MagicMock()
fake_anthropic.Anthropic = MagicMock(return_value=MagicMock())
sys.modules['anthropic'] = fake_anthropic

# Mock process_file so we can inspect what convert_fn it received without
# actually opening a subtitle file.
captured = {}
def fake_process_file(path, output_path, nlp, convert_fn=None, dry_run=False):
    captured['convert_fn'] = convert_fn
    captured['nlp'] = nlp
    return []

with patch('subtitle_modifier.subtitle_io.process_file', side_effect=fake_process_file):
    from subtitle_modifier.cli import main
    main(['--claude', 'fake.srt'])

assert captured['nlp'] is None, 'spaCy nlp should be None in Claude mode'
assert captured['convert_fn'] is not None, 'convert_fn should be wired'
# convert_fn closes over convert_texts_claude — verify its qualname
import inspect
src = inspect.getclosurevars(captured['convert_fn']).nonlocals
print('convert_fn nonlocals:', sorted(src.keys()))
"
```

Expected: prints something like `convert_fn nonlocals: ['_bs', '_client', '_model']` and exits cleanly. If the command exits with `Error: --llm and --claude are mutually exclusive` or `Error: no Anthropic API key`, the dispatch is broken or the env var didn't reach the subprocess.

- [ ] **Step 8: Clean up the test fixture**

```bash
rm /tmp/_test.srt
```

- [ ] **Step 9: Run the full test suite to confirm no regressions in CLI parsing**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/subtitle_modifier/cli.py
git commit -m "Wire --claude flag and routing branch into CLI"
```

---

## Task 9: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md` Commands section**

In the Commands section's install block, add `pip install -e ".[claude]"` after the existing `[llm]` line:

```bash
# Install with LLM support
pip install -e ".[llm]"

# Install with Claude (native Anthropic SDK) support
pip install -e ".[claude]"
```

After the existing `--llm` example, add a runtime example:

```bash
# Run the CLI (Claude mode)
subtitle-modifier <file.ass> --claude --claude-model <model-name>
```

- [ ] **Step 2: Update `CLAUDE.md` Module responsibilities**

After the `llm.py` bullet, add a `claude.py` bullet:

```markdown
- **claude.py** — Claude conversion logic via the native Anthropic SDK. Mirrors `llm.py`'s structure but calls `client.messages.create()` (system as a top-level kwarg, not a message). Imports `_SYSTEM_PROMPT`, `_build_prompt`, and `_parse_response` from `llm.py` directly to avoid duplicating the prompt and parser. Validates the wording invariant per event with the same fallback as the OpenAI path.
```

- [ ] **Step 3: Update `CLAUDE.md` Key design decisions**

Add a bullet under "Key design decisions":

```markdown
- **Claude mode** sends `temperature=0` for deterministic recasing. This works on Haiku 4.5 (default), Sonnet 4.6, and Opus 4.6, but **returns a 400 on Opus 4.7**, which removed sampling parameters. Opus is overkill for this task; users who explicitly pass `--claude-model claude-opus-4-7` will see the API error and can switch models.
```

- [ ] **Step 4: Update `CLAUDE.md` Dependencies**

Add a bullet under Dependencies:

```markdown
- **anthropic** (optional) — native Anthropic SDK for `--claude` mode
```

- [ ] **Step 5a: Read `README.md` to find the `--llm` section**

Run: `cat README.md`

Identify three things: where the existing install instructions live (so the `[claude]` line can sit next to `[llm]`), where the existing `--llm` usage example lives, and the surrounding formatting (bullet style, code-fence language tags, heading levels). The next three steps mirror that style.

- [ ] **Step 5b: Add the install line for the `[claude]` extra**

Next to the existing `pip install -e ".[llm]"` line, add:

```bash
pip install -e ".[claude]"
```

- [ ] **Step 5c: Add a `--claude` usage example**

Mirroring the existing `--llm` example's formatting, add a `--claude` example that uses the default Haiku 4.5 model and the env var:

```bash
ANTHROPIC_API_KEY=sk-... subtitle-modifier subs.ass --claude
```

- [ ] **Step 5d: Add a one-liner contrasting `--claude` and `--llm`**

Near the new example or the existing `--llm` example, add a short note (one or two sentences) like:

> `--claude` uses the native Anthropic SDK and surfaces typed Anthropic errors. `--llm` is the generic OpenAI-compatible path, which can also reach Anthropic via Anthropic's OpenAI-compatible endpoint but loses Anthropic-native error typing.

- [ ] **Step 6: Verify the README still renders sensibly**

Run: `cat README.md | head -80`
Expected: the new content is in a logical place and matches the existing formatting style.

- [ ] **Step 7: Run the full test suite once more as a final sanity check**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document --claude mode in CLAUDE.md and README.md"
```

---

## Task 10: Final verification

This is a non-coding task that confirms the branch is ready to merge.

- [ ] **Step 1: Run the full test suite from a clean state**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 2: Verify `--help` is sensible**

Run: `subtitle-modifier --help`
Expected: `--claude*` flag group is present and well-formatted.

- [ ] **Step 3: Verify install from the `[claude]` extra still resolves**

Run: `pip install -e ".[claude]" --dry-run` (or `pip install -e ".[claude]"` again — idempotent).
Expected: no resolution errors.

- [ ] **Step 4: Verify the branch contents look right**

Run: `git log main..HEAD --oneline`
Expected: a clean, readable list of commits, one per task, with descriptive messages.

Run: `git diff main...HEAD --stat`
Expected: a small, focused diff: new `claude.py`, new `tests/test_claude.py`, modified `cli.py`, modified `pyproject.toml`, modified `CLAUDE.md`, modified `README.md`. No unrelated changes.

- [ ] **Step 5: Manual end-to-end smoke test (optional, requires real API key)**

If you have an `ANTHROPIC_API_KEY` and a small test SRT/ASS file:

```bash
ANTHROPIC_API_KEY=<your-key> subtitle-modifier <test.srt> --claude --dry-run
```

Expected: `Processing: <test.srt>` followed by per-event diffs showing the casing changes Claude produced. If the file has no all-caps lines, expect `(no changes)`.

This step is optional; a real key is not required to consider the implementation complete.

---

## Notes for the implementer

- **The "literal copy" contract for `convert_texts_claude`.** Task 7 calls out that the body must mirror `convert_texts_llm` step-for-step. If you find yourself "improving" or "deduplicating" while you copy, stop — the dedup is a known follow-up (see Future Work in the spec) and bundling it into this delivery is out of scope.
- **The retry path is exercised by Task 4's third test only after Task 5 lands.** Task 4 explicitly accepts either retry-success or sentence-case fallback as a valid outcome to keep the tasks independent. Don't be surprised by the assertion's `or`.
- **`temperature=0` and Opus 4.7.** This is a deliberate choice — see the spec's "Anthropic-API specifics" section. If a future user reports a 400 on Opus 4.7, the answer is "use a different model" (or the deferred option to add a `--claude-no-temperature` flag).
- **Frequent commits.** Each task ends in a commit. Don't batch.
