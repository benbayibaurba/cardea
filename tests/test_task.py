import unittest

from cardea.task import (
    AnswerDelta,
    Completed,
    GenerationOptions,
    OutputValidationError,
    ReasoningDelta,
    TaskRunner,
)


def fake_transport(chunks):
    async def stream(messages, **options):
        del messages, options
        for chunk in chunks:
            yield chunk

    return stream


class TaskContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_options_are_forwarded_to_the_model_client(self):
        received = {}

        async def transport(messages, **options):
            del messages
            received.update(options)
            yield "answer"

        runner = TaskRunner(
            "configured",
            lambda value: [],
            lambda answer: answer,
            transport_stream=transport,
        )
        generation = GenerationOptions(
            thinking=False,
            temperature=0.2,
            top_p=0.8,
            max_tokens=128,
        )

        await runner.run(None, generation=generation)

        self.assertEqual(
            received,
            {
                "thinking": False,
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 128,
            },
        )

    async def test_run_and_stream_have_the_same_completed_result(self):
        chunks = [
            "<think>",
            'inspect {"img_idx": 0, "bbox_2d": [1, 2, 3, 4], "label": "RCA"}',
            "</think>Right ",
            "Dominance",
        ]
        task = TaskRunner(
            "dominance",
            lambda value: [{"role": "user", "content": value}],
            lambda answer: answer.upper(),
            transport_stream=fake_transport(chunks),
        )

        streamed_events = [event async for event in task.stream("input")]
        completed = next(event.result for event in streamed_events if isinstance(event, Completed))
        direct = await task.run("input")

        self.assertEqual(direct, completed)
        self.assertTrue(any(isinstance(event, ReasoningDelta) for event in streamed_events))
        self.assertTrue(any(isinstance(event, AnswerDelta) for event in streamed_events))
        self.assertEqual(direct.answer, "RIGHT DOMINANCE")
        self.assertEqual(direct.answer_text, "Right Dominance")
        self.assertEqual(len(direct.boxes), 1)

    async def test_parse_failure_never_changes_the_declared_output_type(self):
        task = TaskRunner(
            "strict",
            lambda value: [],
            lambda answer: (_ for _ in ()).throw(ValueError("bad output")),
            transport_stream=fake_transport(["reasoning</think>unsupported"]),
        )

        with self.assertRaises(OutputValidationError) as raised:
            await task.run(None)

        self.assertEqual(raised.exception.answer_text, "unsupported")


if __name__ == "__main__":
    unittest.main()
