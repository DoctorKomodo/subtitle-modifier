"""Unit tests for the LLM-based conversion pipeline (no real API calls)."""

from unittest.mock import MagicMock

import pytest

from subtitle_modifier.llm import (
    _build_prompt,
    _lowercase_preserving_markers,
    _parse_response,
    convert_texts_llm,
    recase_batch,
)


def _make_mock_client(response_text: str):
    """Create a mock OpenAI client that returns the given text."""
    client = MagicMock()
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


class TestLowercasePreservingMarkers:
    def test_no_markers(self):
        assert _lowercase_preserving_markers("HELLO WORLD") == "hello world"

    def test_single_marker(self):
        assert _lowercase_preserving_markers("HELLO\\NWORLD") == "hello\\Nworld"

    def test_multiple_markers(self):
        result = _lowercase_preserving_markers("A\\NB\\NC")
        assert result == "a\\Nb\\Nc"

    def test_marker_at_start(self):
        assert _lowercase_preserving_markers("\\NHELLO") == "\\Nhello"

    def test_marker_at_end(self):
        assert _lowercase_preserving_markers("HELLO\\N") == "hello\\N"

    def test_already_lowercase(self):
        assert _lowercase_preserving_markers("hello\\Nworld") == "hello\\Nworld"


class TestBuildPrompt:
    def test_single_line(self):
        assert _build_prompt(["hello world"]) == "1: hello world"

    def test_multiple_lines(self):
        result = _build_prompt(["hello", "world", "foo"])
        assert result == "1: hello\n2: world\n3: foo"

    def test_empty(self):
        assert _build_prompt([]) == ""

    def test_preserves_backslash_n(self):
        result = _build_prompt(["hello\\Nworld"])
        assert result == "1: hello\\Nworld"


class TestParseResponse:
    def test_single_line(self):
        result = _parse_response("1: Hello world", 1)
        assert result == ["Hello world"]

    def test_multiple_lines(self):
        result = _parse_response("1: Hello\n2: World\n3: Foo", 3)
        assert result == ["Hello", "World", "Foo"]

    def test_count_mismatch(self):
        assert _parse_response("1: Hello\n2: World", 3) is None

    def test_wrong_numbering(self):
        assert _parse_response("1: Hello\n3: World", 2) is None

    def test_with_colon_space(self):
        result = _parse_response("1: Hello world", 1)
        assert result == ["Hello world"]

    def test_without_space_after_colon(self):
        result = _parse_response("1:Hello world", 1)
        assert result == ["Hello world"]

    def test_preserves_backslash_n(self):
        result = _parse_response("1: Hello\\Nworld", 1)
        assert result == ["Hello\\Nworld"]

    def test_empty_response(self):
        assert _parse_response("", 1) is None

    def test_extra_text_ignored(self):
        """Extra non-numbered text before/after should not affect parsing."""
        result = _parse_response("Here you go:\n1: Hello\n2: World", 2)
        assert result == ["Hello", "World"]

    def test_code_fence_stripped(self):
        """LLMs sometimes wrap responses in markdown code blocks."""
        result = _parse_response("```\n1: Hello\n2: World\n```", 2)
        assert result == ["Hello", "World"]

    def test_code_fence_with_language(self):
        result = _parse_response("```text\n1: Hello\n2: World\n```", 2)
        assert result == ["Hello", "World"]


class TestRecaseBatch:
    def test_basic_recase(self):
        client = _make_mock_client("1: Hello world. John went to Paris.")
        result = recase_batch(
            ["hello world. john went to paris."], client, "test-model"
        )
        assert result == ["Hello world. John went to Paris."]

    def test_empty_input(self):
        client = _make_mock_client("")
        result = recase_batch([], client, "test-model")
        assert result == []
        client.chat.completions.create.assert_not_called()

    def test_multiple_items(self):
        client = _make_mock_client("1: Hello\n2: World")
        result = recase_batch(["hello", "world"], client, "test-model")
        assert result == ["Hello", "World"]

    def test_retry_on_parse_failure(self):
        """First response fails to parse, second succeeds."""
        client = MagicMock()
        bad_message = MagicMock()
        bad_message.content = "I don't understand"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]

        good_message = MagicMock()
        good_message.content = "1: Hello"
        good_choice = MagicMock()
        good_choice.message = good_message
        good_response = MagicMock()
        good_response.choices = [good_choice]

        client.chat.completions.create.side_effect = [bad_response, good_response]
        result = recase_batch(["hello"], client, "test-model")
        assert result == ["Hello"]
        assert client.chat.completions.create.call_count == 2

    def test_fallback_on_double_parse_failure(self):
        """Both attempts fail to parse -> falls back to sentence case."""
        client = MagicMock()
        bad_message = MagicMock()
        bad_message.content = "garbage"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]
        client.chat.completions.create.return_value = bad_response

        result = recase_batch(["hello world"], client, "test-model")
        assert result == ["Hello world"]
        assert client.chat.completions.create.call_count == 2


class TestConvertTextsLlm:
    def _mock_recase(self, inputs):
        """Simple mock that title-cases each word for testing."""
        return [text.title() for text in inputs]

    def test_basic_pipeline(self):
        client = _make_mock_client("1: Hello world.\n2: Goodbye.")
        result = convert_texts_llm(
            ["HELLO WORLD.", "GOODBYE."], client, "test-model", batch_size=50
        )
        assert result == ["Hello world.", "Goodbye."]

    def test_ass_tag_roundtrip(self):
        client = _make_mock_client("1: Hello world")
        result = convert_texts_llm(
            ["{\\i1}HELLO WORLD{\\i0}"], client, "test-model"
        )
        assert result == ["{\\i1}Hello world{\\i0}"]

    def test_backslash_n_preserved(self):
        client = _make_mock_client("1: Hello\\Nworld")
        result = convert_texts_llm(["HELLO\\NWORLD"], client, "test-model")
        assert result == ["Hello\\Nworld"]

    def test_wording_invariant_fallback(self):
        """When LLM changes wording, fall back to sentence case."""
        client = _make_mock_client("1: Hi there")  # changed "hello" to "hi there"
        result = convert_texts_llm(["HELLO WORLD"], client, "test-model")
        # Should fall back to to_sentence_case("hello world") = "Hello world"
        assert result == ["Hello world"]

    def test_batch_boundaries_exact(self):
        """Exactly batch_size items in one batch."""
        n = 3
        response = "\n".join(f"{i + 1}: Text {i + 1}" for i in range(n))
        client = _make_mock_client(response)
        texts = [f"TEXT {i + 1}" for i in range(n)]
        result = convert_texts_llm(texts, client, "test-model", batch_size=3)
        assert len(result) == n
        client.chat.completions.create.assert_called_once()

    def test_batch_boundaries_split(self):
        """N+1 items should produce 2 API calls with batch_size=N."""
        n = 3
        resp1 = "\n".join(f"{i + 1}: Text {i + 1}" for i in range(n))
        resp2 = "1: Text 4"
        client = MagicMock()

        msg1 = MagicMock()
        msg1.content = resp1
        choice1 = MagicMock()
        choice1.message = msg1
        response1 = MagicMock()
        response1.choices = [choice1]

        msg2 = MagicMock()
        msg2.content = resp2
        choice2 = MagicMock()
        choice2.message = msg2
        response2 = MagicMock()
        response2.choices = [choice2]

        client.chat.completions.create.side_effect = [response1, response2]

        texts = [f"TEXT {i + 1}" for i in range(n + 1)]
        result = convert_texts_llm(texts, client, "test-model", batch_size=n)
        assert len(result) == n + 1
        assert client.chat.completions.create.call_count == 2

    def test_empty_input(self):
        client = _make_mock_client("")
        result = convert_texts_llm([], client, "test-model")
        assert result == []
        client.chat.completions.create.assert_not_called()

    def test_whitespace_only_input(self):
        """Whitespace-only text should pass through."""
        client = _make_mock_client("1:    ")
        result = convert_texts_llm(["   "], client, "test-model")
        assert len(result) == 1

    def test_backslash_n_case_normalization(self):
        """LLM returning \\n instead of \\N should not trigger wording fallback."""
        client = _make_mock_client("1: Hello\\nworld")
        result = convert_texts_llm(["HELLO\\NWORLD"], client, "test-model")
        # Should accept \n as equivalent to \N, not fall back
        assert result == ["Hello\\nworld"]

    def test_mixed_ass_tags_and_entities(self):
        client = _make_mock_client("1: John went to Paris.")
        result = convert_texts_llm(
            ["{\\pos(320,50)}JOHN WENT TO PARIS."], client, "test-model"
        )
        assert result == ["{\\pos(320,50)}John went to Paris."]
