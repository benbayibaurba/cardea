import unittest

from cardea.outputs import CoronaryFindings, KeyframeSelection
from cardea.tasks import complexity, dominance, keyframe, report, view


class TaskOutputTests(unittest.TestCase):
    def test_complexity_prompt_is_deterministic(self):
        first = complexity._build_messages([])
        second = complexity._build_messages([])

        self.assertEqual(first, second)
        self.assertIn(
            "Could you please evaluate the SYNTAX score complexity level for this case?",
            first[0]["content"][-1]["text"],
        )

    def test_classification_tasks_return_canonical_strings(self):
        self.assertEqual(
            dominance._parse_answer("Right Dominance"),
            "Right Dominance",
        )
        self.assertEqual(
            complexity._parse_answer(
                "Normal to Intermediate Complexity: SYNTAX score 0-32"
            ),
            "Normal to Intermediate Complexity: SYNTAX score 0-32",
        )
        self.assertEqual(view._parse_answer("LCA"), "LCA")
        self.assertEqual(view._parse_answer("uncertain response"), "OTHER")

    def test_structured_tasks_return_validated_models(self):
        selection = keyframe._parse_answer(
            '{"selected_frame_idxs": [1, 2], "best_frame_idx": 2}'
        )
        findings = report._parse_answer(
            '{"lm": "normal", "lad": "normal", "lcx": "normal", "rca": "normal"}'
        )

        self.assertIsInstance(selection, KeyframeSelection)
        self.assertEqual(selection.best_frame_idx, 2)
        self.assertIsInstance(findings, CoronaryFindings)
        self.assertEqual(findings.lad, "normal")


if __name__ == "__main__":
    unittest.main()
