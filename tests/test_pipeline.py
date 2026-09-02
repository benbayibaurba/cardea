import unittest

from cardea.pipeline import StudyResult, run_pipeline


class PipelineContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_views_return_a_result_with_a_failure_reason(self):
        result = await run_pipeline([])

        self.assertIsInstance(result, StudyResult)
        self.assertEqual(result.fail_reason, "No LCA or RCA view was identified.")
        self.assertIsNone(result.dominance)
        self.assertIsNone(result.report)
        self.assertIsNone(result.complexity)


if __name__ == "__main__":
    unittest.main()
