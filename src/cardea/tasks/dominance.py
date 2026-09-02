"""Coronary dominance classification."""

from ..imaging import image_part
from ..prompts import GROUNDING_PROMPT
from ..task import GenerationOptions, TaskRunner

__all__ = ["run", "stream"]

_OPTIONS = ("Left Dominance", "Right Dominance")


# Intentional: no space before "Respond", and the images glue directly onto the
# end of this block. This matches the model's training-time prompt.
_INSTRUCT = (
    "Based on the SYNTAX SCORE standards, would you kindly determine whether this "
    "case presents as Left Dominance or Right Dominance?"
    "Respond one of the options without explanation.\noptions:\n- Left Dominance\n- Right Dominance"
)


def _build_messages(frames):
    text = f"{GROUNDING_PROMPT}\n{'<image>' * len(frames)}{_INSTRUCT}"
    return [{"role": "user", "content": [image_part(frame) for frame in frames] + [{"type": "text", "text": text}]}]


def _parse_answer(answer):
    for option in _OPTIONS:
        if option.lower() in answer.lower():
            return option
    raise ValueError("answer did not contain a supported dominance")


_runner = TaskRunner(
    name="dominance",
    build_messages=_build_messages,
    parse_answer=_parse_answer,
)


async def run(frames, *, generation: GenerationOptions | None = None):
    return await _runner.run(frames, generation=generation)


def stream(frames, *, generation: GenerationOptions | None = None):
    return _runner.stream(frames, generation=generation)
