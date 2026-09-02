import unittest

from cardea.streaming import stream_first_with_boxes
from cardea.task import TaskRunner


class _CandidateTask:
    def __init__(self):
        self._next = 0

    def stream(self, input_value, *, generation=None):
        index = self._next
        self._next += 1
        raw = (
            "<think>no spatial anchor</think>answer zero"
            if index == 0
            else '<think>{"img_idx": 0, "bbox_2d": [1, 2, 3, 4]}</think>answer one'
        )

        async def transport(messages, **kwargs):
            del messages, kwargs
            yield raw

        return TaskRunner(
            "candidate",
            lambda value: [],
            lambda answer: answer,
            transport_stream=transport,
        ).stream(input_value, generation=generation)


class SamplingTests(unittest.IsolatedAsyncioTestCase):
    async def test_resampling_selects_the_first_completed_box_rollout(self):
        snapshots = [
            snapshot
            async for snapshot in stream_first_with_boxes(
                _CandidateTask(),
                None,
                attempts=2,
            )
        ]
        final = snapshots[-1]
        self.assertEqual(final.attempt, 1)
        self.assertEqual(final.result.answer, "answer one")
        self.assertTrue(final.result.boxes)
        self.assertIsNone(final.notice)


if __name__ == "__main__":
    unittest.main()
