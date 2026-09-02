"""Shared execution contract for every CARDEA inference task."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .boxes import BoundingBox, parse_boxes
from .client import stream_text
from .thinking import OutputPhase, ThinkStreamDecoder

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class GenerationOptions:
    thinking: bool = True
    temperature: float = 0.5
    top_p: float = 0.95
    max_tokens: int = 4096


@dataclass(frozen=True)
class TaskResult(Generic[OutputT]):
    reasoning: str | None
    answer_text: str
    answer: OutputT
    raw_text: str
    boxes: tuple[BoundingBox, ...] = ()


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True)
class Completed(Generic[OutputT]):
    result: TaskResult[OutputT]


class OutputValidationError(ValueError):
    def __init__(self, task_name: str, answer_text: str):
        super().__init__(f"{task_name} returned an invalid answer: {answer_text!r}")
        self.task_name = task_name
        self.answer_text = answer_text


class TaskRunner(Generic[InputT, OutputT]):
    """Bind domain input, prompt construction and typed output parsing."""

    def __init__(
        self,
        name: str,
        build_messages: Callable[[InputT], list[dict[str, Any]]],
        parse_answer: Callable[[str], OutputT],
        transport_stream=stream_text,
    ):
        self.name = name
        self._build_messages = build_messages
        self._parse_answer = parse_answer
        self._transport_stream = transport_stream

    async def stream(
        self,
        input_value: InputT,
        *,
        generation: GenerationOptions | None = None,
    ) -> AsyncIterator[
        ReasoningDelta | AnswerDelta | Completed[OutputT]
    ]:
        generation = generation or GenerationOptions()
        decoder = ThinkStreamDecoder(thinking=generation.thinking)
        raw_parts = []
        reasoning_parts = []
        answer_parts = []

        async for delta in self._transport_stream(
            self._build_messages(input_value),
            thinking=generation.thinking,
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_tokens,
        ):
            raw_parts.append(delta)
            for event in self._decode(decoder.feed(delta), reasoning_parts, answer_parts):
                yield event

        for event in self._decode(decoder.flush(), reasoning_parts, answer_parts):
            yield event

        reasoning = "".join(reasoning_parts) or None
        answer_text = "".join(answer_parts).strip()
        try:
            answer = self._parse_answer(answer_text)
        except OutputValidationError:
            raise
        except Exception as exc:
            raise OutputValidationError(self.name, answer_text) from exc

        yield Completed(
            TaskResult(
                reasoning=reasoning,
                answer_text=answer_text,
                answer=answer,
                raw_text="".join(raw_parts),
                boxes=parse_boxes(reasoning),
            )
        )

    async def run(
        self,
        input_value: InputT,
        *,
        generation: GenerationOptions | None = None,
    ) -> TaskResult[OutputT]:
        result = None
        async for event in self.stream(input_value, generation=generation):
            if isinstance(event, Completed):
                result = event.result
        if result is None:
            raise RuntimeError(f"{self.name} stream ended without a completed result")
        return result

    @staticmethod
    def _decode(pieces, reasoning_parts, answer_parts):
        events = []
        for phase, text in pieces:
            if not text:
                continue
            if phase is OutputPhase.REASONING:
                reasoning_parts.append(text)
                events.append(ReasoningDelta(text))
            else:
                answer_parts.append(text)
                events.append(AnswerDelta(text))
        return events
