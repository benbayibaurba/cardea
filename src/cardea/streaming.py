"""Inference policies layered on top of the common task stream contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from .boxes import has_cob
from .task import (
    AnswerDelta,
    Completed,
    GenerationOptions,
    ReasoningDelta,
    TaskResult,
)

OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class TaskSnapshot(Generic[OutputT]):
    """Replaceable current state, suitable for a live UI."""

    reasoning: str | None
    answer_text: str
    result: TaskResult[OutputT] | None = None
    attempt: int = 0
    notice: str | None = None


@dataclass
class _AttemptState:
    reasoning: str = ""
    answer_text: str = ""
    reasoning_closed: bool = False
    result: object | None = None
    error: Exception | None = None
    done: bool = False


async def stream_snapshots(
    task,
    input_value,
    *,
    generation: GenerationOptions | None = None,
):
    """Turn deltas from one task execution into replaceable snapshots."""
    reasoning = ""
    answer_text = ""
    async for event in task.stream(input_value, generation=generation):
        result = None
        if isinstance(event, ReasoningDelta):
            reasoning += event.text
        elif isinstance(event, AnswerDelta):
            answer_text += event.text
        elif isinstance(event, Completed):
            result = event.result
            reasoning = result.reasoning or ""
            answer_text = result.answer_text
        yield TaskSnapshot(reasoning or None, answer_text, result=result)


async def stream_first_with_boxes(
    task,
    input_value,
    *,
    attempts: int,
    generation: GenerationOptions | None = None,
):
    """Run candidates concurrently and expose the first rollout with a CoB box.

    Snapshots can replace earlier snapshots. This makes candidate switching
    explicit and keeps that policy out of presentation code.
    """
    count = max(1, int(attempts))
    if count == 1:
        async for snapshot in stream_snapshots(
            task,
            input_value,
            generation=generation,
        ):
            yield snapshot
        return

    states = [_AttemptState() for _ in range(count)]
    changed = asyncio.Queue()

    async def produce(index):
        state = states[index]
        try:
            async for event in task.stream(input_value, generation=generation):
                if isinstance(event, ReasoningDelta):
                    state.reasoning += event.text
                elif isinstance(event, AnswerDelta):
                    state.reasoning_closed = True
                    state.answer_text += event.text
                elif isinstance(event, Completed):
                    state.reasoning_closed = True
                    state.result = event.result
                    state.reasoning = event.result.reasoning or ""
                    state.answer_text = event.result.answer_text
                await changed.put(index)
        except Exception as exc:  # noqa: BLE001 - one failed rollout must not cancel the others
            state.error = exc
        finally:
            state.done = True
            await changed.put(index)

    workers = [asyncio.create_task(produce(index)) for index in range(count)]
    try:
        while not all(state.done for state in states):
            await changed.get()
            selected = _select_live_attempt(states)
            if selected is not None:
                yield _snapshot(states[selected], selected)

        successful = [
            (index, state)
            for index, state in enumerate(states)
            if state.result is not None
        ]
        if not successful:
            error = next((state.error for state in states if state.error), None)
            raise error or RuntimeError("all sampling attempts ended without a result")

        selected, state = next(
            (
                (index, candidate)
                for index, candidate in successful
                if candidate.result.boxes
            ),
            successful[0],
        )
        notice = None if state.result.boxes else "No Chain-of-Box appeared in the sampled rollouts."
        yield _snapshot(state, selected, notice=notice)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def _select_live_attempt(states):
    for index, state in enumerate(states):
        if state.error:
            continue
        if not (state.reasoning_closed and not has_cob(state.reasoning)):
            return index
    return next((index for index, state in enumerate(states) if not state.error), None)


def _snapshot(state, index, notice=None):
    return TaskSnapshot(
        reasoning=state.reasoning or None,
        answer_text=state.answer_text,
        result=state.result,
        attempt=index,
        notice=notice,
    )
