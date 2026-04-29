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
