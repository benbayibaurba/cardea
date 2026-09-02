import unittest

from cardea.thinking import OutputPhase, ThinkStreamDecoder


class ThinkStreamDecoderTests(unittest.TestCase):
    def test_complete_thinking_tokens_split_reasoning_and_answer(self):
        decoder = ThinkStreamDecoder()
        pieces = []
        for chunk in ("<think>", "look", "</think>\nRight ", "Dominance"):
            pieces.extend(decoder.feed(chunk))

        reasoning = "".join(text for phase, text in pieces if phase is OutputPhase.REASONING)
        answer = "".join(text for phase, text in pieces if phase is OutputPhase.ANSWER)
        self.assertEqual(reasoning, "look")
        self.assertEqual(answer.strip(), "Right Dominance")

    def test_delimiter_and_content_may_share_a_delta(self):
        decoder = ThinkStreamDecoder()
        pieces = decoder.feed("<think>look") + decoder.feed("</think>Right Dominance")

        self.assertEqual(
            pieces,
            [
                (OutputPhase.REASONING, "look"),
                (OutputPhase.ANSWER, "Right Dominance"),
            ],
        )

    def test_output_is_an_answer_when_the_server_suppresses_the_trace(self):
        decoder = ThinkStreamDecoder(thinking=False)
        self.assertEqual(
            decoder.feed("Right Dominance"),
            [(OutputPhase.ANSWER, "Right Dominance")],
        )

    def test_a_prefilled_opening_tag_still_starts_the_reasoning_phase(self):
        """The served template ends the prompt with "<think>", so the completion
        starts mid-trace and only the closing tag ever arrives. Treating that
        first delta as an answer filed the whole trace, closing tag included, as
        the answer -- which made every view read OTHER and every keyframe JSON
        unparseable."""
        decoder = ThinkStreamDecoder()
        pieces = []
        for chunk in ("Got it, let's look", " at the image.", "\n</think>", " LCA"):
            pieces.extend(decoder.feed(chunk))

        reasoning = "".join(t for phase, t in pieces if phase is OutputPhase.REASONING)
        answer = "".join(t for phase, t in pieces if phase is OutputPhase.ANSWER)
        self.assertEqual(reasoning, "Got it, let's look at the image.\n")
        self.assertEqual(answer.strip(), "LCA")

    def test_closing_marker_split_across_deltas(self):
        """A marker straddling two chunks must not leak into either phase."""
        decoder = ThinkStreamDecoder()
        pieces = decoder.feed("look</thi") + decoder.feed("nk>LCA")

        self.assertEqual(
            pieces,
            [(OutputPhase.REASONING, "look"), (OutputPhase.ANSWER, "LCA")],
        )

    def test_withheld_tail_is_emitted_when_the_stream_ends(self):
        """A trailing fragment that turned out not to be a marker is not lost."""
        decoder = ThinkStreamDecoder()
        self.assertEqual(decoder.feed("look</"), [(OutputPhase.REASONING, "look")])
        self.assertEqual(decoder.flush(), [(OutputPhase.REASONING, "</")])

    def test_flush_is_empty_when_nothing_is_withheld(self):
        decoder = ThinkStreamDecoder()
        decoder.feed("look</think>LCA")
        self.assertEqual(decoder.flush(), [])


if __name__ == "__main__":
    unittest.main()
