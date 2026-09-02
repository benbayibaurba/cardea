"""SYNTAX complexity classification."""

from ..imaging import image_part
from ..prompts import GROUNDING_PROMPT
from ..task import GenerationOptions, TaskRunner

__all__ = ["run", "stream"]

_OPTIONS = (
    "Normal to Intermediate Complexity: SYNTAX score 0-32",
    "High Complexity: SYNTAX score > 32",
)


# The leading and trailing newlines match the training-time prompt.
_INSTRUCT = """
Please analyze the provided coronary angiography frames to determine the SYNTAX score complexity.
**Important Note**: The provided image sequence may contain non-diagnostic frames (e.g., poor contrast opacification, non-cardiac views, or artifacts). You must autonomously filter out these invalid frames and base your assessment ONLY on the high-quality, diagnostic frames showing clear vessel structures.

Please respond with exactly one of the following options without providing extra explanation:
Options:
- Normal to Intermediate Complexity: SYNTAX score 0-32
- High Complexity: SYNTAX score > 32

Could you please evaluate the SYNTAX score complexity level for this case?
"""


def _build_messages(frames):
    text = f"{GROUNDING_PROMPT}\n{'<image>' * len(frames)}\n\n{_INSTRUCT}"
    return [{"role": "user", "content": [image_part(frame) for frame in frames] + [{"type": "text", "text": text}]}]


def _parse_answer(answer):
    for option in _OPTIONS:
        if option.lower() in answer.lower():
            return option
    raise ValueError("answer did not contain a supported complexity")


_runner = TaskRunner(
    name="complexity",
    build_messages=_build_messages,
    parse_answer=_parse_answer,
)


async def run(frames, *, generation: GenerationOptions | None = None):
    return await _runner.run(frames, generation=generation)


def stream(frames, *, generation: GenerationOptions | None = None):
    return _runner.stream(frames, generation=generation)
