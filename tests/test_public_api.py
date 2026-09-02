import unittest

import cardea
from cardea import outputs
from cardea.tasks import complexity, dominance, keyframe, report, view


class PublicApiTests(unittest.TestCase):
    def test_task_modules_expose_the_common_execution_contract(self):
        for task_module in (complexity, dominance, keyframe, report, view):
            self.assertTrue(callable(task_module.run))
            self.assertTrue(callable(task_module.stream))
            self.assertFalse(hasattr(task_module, "build_messages"))
            self.assertFalse(hasattr(task_module, "parse"))
            self.assertFalse(hasattr(task_module, "TASK"))

    def test_output_contracts_have_one_public_namespace(self):
        self.assertIs(cardea.outputs, outputs)
        for name in (
            "Complexity",
            "CoronaryFindings",
            "Dominance",
            "KeyframeSelection",
            "View",
        ):
            self.assertTrue(hasattr(outputs, name))


if __name__ == "__main__":
    unittest.main()
