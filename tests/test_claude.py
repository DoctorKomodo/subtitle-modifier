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
