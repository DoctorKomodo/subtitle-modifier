"""Unit tests for the LLM-based conversion pipeline (no real API calls)."""

from unittest.mock import MagicMock

import pytest

from subtitle_modifier.converter import (
    reinsert_newline_markers,
    strip_newline_markers,
)
from subtitle_modifier.llm import (
    _build_prompt,
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


class TestBuildPrompt:
    def test_single_line(self):
        assert _build_prompt(["hello world"]) == "1: hello world"

    def test_multiple_lines(self):
        result = _build_prompt(["hello", "world", "foo"])
        assert result == "1: hello\n2: world\n3: foo"

    def test_empty(self):
        assert _build_prompt([]) == ""


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


class TestStripNewlineMarkers:
    def test_no_markers(self):
        text, positions = strip_newline_markers("hello world")
        assert text == "hello world"
        assert positions == []

    def test_single_marker(self):
        text, positions = strip_newline_markers("hello\\Nworld")
        assert text == "hello world"
        assert positions == [5]

    def test_multiple_markers(self):
        text, positions = strip_newline_markers("a\\Nb\\Nc")
        assert text == "a b c"
        assert positions == [1, 3]

    def test_marker_at_start(self):
        text, positions = strip_newline_markers("\\Nhello")
        assert text == " hello"
        assert positions == [0]

    def test_marker_at_end(self):
        text, positions = strip_newline_markers("hello\\N")
        assert text == "hello "
        assert positions == [5]

    def test_roundtrip(self):
        original = "hello\\Nworld\\Nfoo"
        text, positions = strip_newline_markers(original)
        restored = reinsert_newline_markers(text, positions)
        assert restored == original

    def test_space_before_marker(self):
        """Existing space before \\N is preserved."""
        text, positions = strip_newline_markers("hello \\Nworld")
        assert text == "hello  world"
        assert positions == [6]


class TestReinsertNewlineMarkers:
    def test_no_positions(self):
        assert reinsert_newline_markers("hello", []) == "hello"

    def test_single_position(self):
        assert reinsert_newline_markers("hello world", [5]) == "hello\\Nworld"

    def test_multiple_positions(self):
        assert reinsert_newline_markers("a b c", [1, 3]) == "a\\Nb\\Nc"

    def test_position_at_start(self):
        assert reinsert_newline_markers(" hello", [0]) == "\\Nhello"

    def test_position_at_end(self):
        assert reinsert_newline_markers("hello ", [5]) == "hello\\N"

    def test_space_before_marker_preserved(self):
        """Space adjacent to the \\N position is kept."""
        assert reinsert_newline_markers("hello  world", [6]) == "hello \\Nworld"


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

    def test_backslash_n_stripped_and_reinserted(self):
        """\\N is replaced with space for LLM, then reinserted at original position."""
        # "HELLO\\NWORLD" -> strip \N -> "hello world" -> LLM recases
        # LLM returns "Hello world" -> reinsert \N at pos 5 -> "Hello\\Nworld"
        client = _make_mock_client("1: Hello world")
        result = convert_texts_llm(["HELLO\\NWORLD"], client, "test-model")
        assert result == ["Hello\\Nworld"]

    def test_backslash_n_with_space(self):
        """\\N with adjacent space is handled correctly."""
        # "HELLO \\NWORLD" -> strip \N -> "hello  world" (double space) -> LLM recases
        client = _make_mock_client("1: Hello  world")
        result = convert_texts_llm(["HELLO \\NWORLD"], client, "test-model")
        assert result == ["Hello \\Nworld"]

    def test_backslash_n_after_sentence_end(self):
        """\\N after sentence-ending punctuation — LLM sees full context."""
        # "NO.\\NIS HE?" -> strip \N -> "no. is he?" -> LLM returns "No. Is he?"
        # -> reinsert \N at pos 3 -> "No.\\NIs he?"
        client = _make_mock_client("1: No. Is he?")
        result = convert_texts_llm(["NO.\\NIS HE?"], client, "test-model")
        assert result == ["No.\\NIs he?"]

    def test_backslash_n_never_sent_to_llm(self):
        """Verify the LLM prompt contains no \\N markers."""
        client = _make_mock_client("1: Hello world")
        convert_texts_llm(["HELLO\\NWORLD"], client, "test-model")
        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "\\N" not in user_content

    def test_wording_invariant_fallback(self):
        """When LLM changes wording, fall back to sentence case."""
        client = _make_mock_client("1: Hi there")  # changed "hello" to "hi there"
        result = convert_texts_llm(["HELLO WORLD"], client, "test-model")
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

    def test_trailing_whitespace_stripped(self):
        """LLM adding trailing spaces should not trigger wording fallback."""
        client = _make_mock_client("1: Hello world.  \n2: Goodbye.  ")
        result = convert_texts_llm(
            ["HELLO WORLD.", "GOODBYE."], client, "test-model", batch_size=50
        )
        assert result == ["Hello world.", "Goodbye."]

    def test_mixed_ass_tags_and_entities(self):
        client = _make_mock_client("1: John went to Paris.")
        result = convert_texts_llm(
            ["{\\pos(320,50)}JOHN WENT TO PARIS."], client, "test-model"
        )
        assert result == ["{\\pos(320,50)}John went to Paris."]

    def test_multiple_events_with_backslash_n(self):
        """Multiple events mixing \\N and plain text."""
        # Event 0: "A\\NB" -> strip \N -> "a b", Event 1: "C" -> "c"
        # LLM returns "A b", "C" -> reinsert \N at pos 1 -> "A\\Nb"
        client = _make_mock_client("1: A b\n2: C")
        result = convert_texts_llm(["A\\NB", "C"], client, "test-model")
        assert result == ["A\\Nb", "C"]
