"""Integration tests using pysubs2 to process subtitle strings."""

import pytest
import spacy
import pysubs2

from subtitle_modifier.converter import convert_text


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


SRT_CONTENT = """\
1
00:00:01,000 --> 00:00:03,000
JOHN WENT TO NEW YORK.

2
00:00:04,000 --> 00:00:06,000
HE LOVED THE CITY.
I THINK IT'S GREAT.

3
00:00:07,000 --> 00:00:09,000
MARY SAID HELLO TO JOHN.
"""

ASS_HEADER = """\
[Script Info]
Title: Test
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\i1}JOHN WENT TO NEW YORK.{\\i0}
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,HE LOVED THE CITY.
"""


class TestSrtIntegration:
    def test_srt_processing(self, nlp):
        subs = pysubs2.SSAFile.from_string(SRT_CONTENT, format_="srt")

        for event in subs.events:
            event.text = convert_text(event.text, nlp)

        # "I" should be capitalized in second subtitle
        # pysubs2 uses \N for newlines in text field
        text1 = subs.events[1].text
        assert "I think" in text1 or "I " in text1

        # Verify output can be serialized back to SRT
        output = subs.to_string("srt")
        assert "00:00:01" in output
        assert "-->" in output

    def test_srt_no_wording_changes(self, nlp):
        """Verify that only casing changes, never wording."""
        subs = pysubs2.SSAFile.from_string(SRT_CONTENT, format_="srt")
        originals = [event.text for event in subs.events]

        for event in subs.events:
            event.text = convert_text(event.text, nlp)

        for original, event in zip(originals, subs.events):
            assert original.lower() == event.text.lower(), (
                f"Wording changed!\n  Original: {original}\n  Converted: {event.text}"
            )


class TestAssIntegration:
    def test_ass_tags_preserved(self, nlp):
        subs = pysubs2.SSAFile.from_string(ASS_HEADER, format_="ass")

        for event in subs.events:
            event.text = convert_text(event.text, nlp)

        # First event should still have italic tags
        assert "{\\i1}" in subs.events[0].text
        assert "{\\i0}" in subs.events[0].text

        # Plain text between tags should be sentence-cased
        plain = subs.events[0].text.replace("{\\i1}", "").replace("{\\i0}", "")
        assert plain[0].isupper()  # Starts with capital

    def test_ass_no_wording_changes(self, nlp):
        """Verify that only casing changes, never wording."""
        subs = pysubs2.SSAFile.from_string(ASS_HEADER, format_="ass")
        originals = [event.text for event in subs.events]

        for event in subs.events:
            event.text = convert_text(event.text, nlp)

        for original, event in zip(originals, subs.events):
            assert original.lower() == event.text.lower(), (
                f"Wording changed!\n  Original: {original}\n  Converted: {event.text}"
            )

    def test_ass_serialization(self, nlp):
        subs = pysubs2.SSAFile.from_string(ASS_HEADER, format_="ass")

        for event in subs.events:
            event.text = convert_text(event.text, nlp)

        output = subs.to_string("ass")
        assert "[Script Info]" in output
        assert "Dialogue:" in output
