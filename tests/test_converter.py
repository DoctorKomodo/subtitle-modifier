"""Unit tests for the converter module."""

import pytest
import spacy

from subtitle_modifier.converter import (
    convert_text,
    reinsert_ass_tags,
    strip_ass_tags,
    to_sentence_case,
)



@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


class TestStripAssTags:
    def test_no_tags(self):
        plain, tags = strip_ass_tags("Hello world")
        assert plain == "Hello world"
        assert tags == []

    def test_single_tag(self):
        plain, tags = strip_ass_tags("{\\i1}Hello world")
        assert plain == "Hello world"
        assert tags == [(0, "{\\i1}")]

    def test_multiple_tags(self):
        plain, tags = strip_ass_tags("{\\i1}Hello {\\b1}world{\\b0}")
        assert plain == "Hello world"
        assert tags == [(0, "{\\i1}"), (6, "{\\b1}"), (11, "{\\b0}")]

    def test_tag_in_middle(self):
        plain, tags = strip_ass_tags("Hello {\\i1}beautiful{\\i0} world")
        assert plain == "Hello beautiful world"
        assert tags == [(6, "{\\i1}"), (15, "{\\i0}")]

    def test_complex_tag(self):
        plain, tags = strip_ass_tags("{\\pos(320,50)}HELLO WORLD")
        assert plain == "HELLO WORLD"
        assert tags == [(0, "{\\pos(320,50)}")]


class TestReinsertAssTags:
    def test_no_tags(self):
        assert reinsert_ass_tags("hello world", []) == "hello world"

    def test_single_tag_at_start(self):
        result = reinsert_ass_tags("hello world", [(0, "{\\i1}")])
        assert result == "{\\i1}hello world"

    def test_multiple_tags(self):
        tags = [(0, "{\\i1}"), (6, "{\\b1}"), (11, "{\\b0}")]
        result = reinsert_ass_tags("hello world", tags)
        assert result == "{\\i1}hello {\\b1}world{\\b0}"

    def test_roundtrip(self):
        original = "{\\i1}Hello {\\b1}world{\\b0}"
        plain, tags = strip_ass_tags(original)
        modified = plain.lower()
        result = reinsert_ass_tags(modified, tags)
        assert result == "{\\i1}hello {\\b1}world{\\b0}"


class TestToSentenceCase:
    def test_all_uppercase(self):
        result = to_sentence_case("HELLO WORLD")
        assert result == "Hello world"

    def test_multiple_sentences(self):
        result = to_sentence_case("HELLO WORLD. HOW ARE YOU?")
        assert result == "Hello world. How are you?"

    def test_exclamation_and_question(self):
        result = to_sentence_case("STOP! WHAT ARE YOU DOING? I DON'T KNOW.")
        assert result == "Stop! What are you doing? I don't know."

    def test_standalone_i(self):
        result = to_sentence_case("I THINK I SHOULD GO")
        assert result == "I think I should go"

    def test_i_not_in_word(self):
        result = to_sentence_case("I THINK IT IS INTERESTING")
        assert result == "I think it is interesting"

    def test_i_in_contraction(self):
        result = to_sentence_case("I'M GOING AND I'LL BE BACK")
        assert result == "I'm going and I'll be back"

    def test_newline_mid_sentence(self):
        """Newline is not a sentence boundary."""
        result = to_sentence_case("WHY ARE WE SITTING\nHERE")
        assert result == "Why are we sitting\nhere"

    def test_newline_after_sentence_end(self):
        """Capitalize after newline only if previous line ended a sentence."""
        result = to_sentence_case("HELLO WORLD.\nHOW ARE YOU")
        assert result == "Hello world.\nHow are you"

    def test_standalone_i_after_newline(self):
        """Standalone I must be capitalized even after newline."""
        result = to_sentence_case("BILLY,\nI'M NOT TALKING ABOUT THIS.")
        assert result == "Billy,\nI'm not talking about this."

    def test_am_pm_abbreviation(self):
        result = to_sentence_case("5:00 A.M. TO 6:00 A.M.")
        assert result == "5:00 a.m. to 6:00 a.m."

    def test_acronym_abbreviation(self):
        result = to_sentence_case("D.E.A. SUSPECTED DRUGS")
        assert result == "D.e.a. suspected drugs"

    def test_honorific_period(self):
        result = to_sentence_case("MR. MCGARRY")
        assert result == "Mr. mcgarry"

    def test_real_sentence_after_multichar_word(self):
        result = to_sentence_case("HELLO WORLD. HOW ARE YOU?")
        assert result == "Hello world. How are you?"

    def test_ellipsis(self):
        result = to_sentence_case("WAIT... WHAT?")
        assert result == "Wait... What?"

    def test_empty_string(self):
        result = to_sentence_case("")
        assert result == ""


class TestConvertText:
    def test_simple_uppercase(self, nlp):
        result = convert_text("HELLO WORLD.", nlp)
        assert result.lower() == "hello world."
        assert result[0] == "H"  # First letter capitalized

    def test_preserves_ass_tags(self, nlp):
        result = convert_text("{\\i1}HELLO WORLD{\\i0}", nlp)
        assert "{\\i1}" in result
        assert "{\\i0}" in result
        # Wording unchanged
        plain_part = result.replace("{\\i1}", "").replace("{\\i0}", "")
        assert plain_part.lower() == "hello world"
        assert plain_part[0] == "H"

    def test_ass_tags_with_newline(self, nlp):
        result = convert_text("{\\i1}HELLO\\NWORLD{\\i0}", nlp)
        assert "{\\i1}" in result
        assert "{\\i0}" in result
        assert "\\N" in result
        assert result.lower() == "{\\i1}hello\\nworld{\\i0}"

    def test_empty_text(self, nlp):
        assert convert_text("", nlp) == ""
        assert convert_text("   ", nlp) == "   "

    def test_named_entities_capitalized(self, nlp):
        result = convert_text("JOHN WENT TO PARIS.", nlp)
        # Verify wording unchanged
        assert result.lower() == "john went to paris."
        # John and Paris should be capitalized (NER on title-cased text)
        assert "John" in result
        assert "Paris" in result

    def test_named_entity_after_newline(self, nlp):
        """Entities after \\N must not have offset-shifted capitalization."""
        result = convert_text("LEFT HAVANA THIS MORNING.\\NJOHN WENT TO PARIS.", nlp)
        assert result.lower() == "left havana this morning.\\njohn went to paris."
        # Havana should be cleanly capitalized, not "hAvana"
        if "havana" not in result.lower():
            pass  # entity not in text
        assert "hAvana" not in result, f"Offset bug: got {result!r}"
        assert "pAris" not in result, f"Offset bug: got {result!r}"

    def test_newline_mid_sentence(self, nlp):
        """\\N is a visual line break, not a sentence boundary."""
        result = convert_text("WHY ARE WE SITTING\\NHERE? YOU SAT DOWN.", nlp)
        assert "\\N" in result
        assert result.startswith("Why are we sitting\\N")
        # 'here' should NOT be capitalized — mid-sentence
        after_newline = result.split("\\N")[1]
        assert after_newline[0] == "h"

    def test_standalone_i_after_newline(self, nlp):
        """Standalone I must be capitalized after \\N."""
        result = convert_text("BILLY,\\NI'M NOT TALKING ABOUT THIS.", nlp)
        assert "\\NI'm" in result

    def test_multiple_sentences(self, nlp):
        result = convert_text("HELLO. GOODBYE.", nlp)
        assert result.startswith("Hello.")
        assert "Goodbye." in result

    def test_already_sentence_case(self, nlp):
        """Input that's already sentence case should not be mangled."""
        text = "John went to Paris."
        result = convert_text(text, nlp)
        assert result == text, (
            f"Sentence-case input was mangled!\n  Input:  {text}\n  Output: {result}"
        )

    def test_wording_never_changes(self, nlp):
        """Only casing should change, never the words themselves."""
        texts = [
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.",
            "HELLO WORLD! HOW ARE YOU?",
            "I THINK I SHOULD GO NOW.",
        ]
        for text in texts:
            result = convert_text(text, nlp)
            assert result.lower() == text.lower(), (
                f"Wording changed!\n  Input:  {text}\n  Output: {result}"
            )
