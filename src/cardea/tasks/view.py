"""Classify a keyframe as LCA, RCA, or OTHER."""

from ..imaging import image_part
from ..task import GenerationOptions, TaskRunner

__all__ = ["run", "stream"]

_VIEWS = ("LCA", "RCA", "OTHER")

_INSTRUCT = (
    "Identify if this image shows the LCA (Left Coronary Artery), "
    "RCA (Right Coronary Artery) or OTHER. "
    "Select 'OTHER' if the view is not sure, not coronary angiography, "
    "or if the vessel is not clear enough such that the contrast medium has not fully filled to the distal end."
)

_OPTIONS = "\noptions:\n -" + "\n- ".join(_VIEWS)
_PROMPT = f"Respond with just one of the options, no further details.\n<image>\n\n{_INSTRUCT}\n{_OPTIONS}"


def _build_messages(frame):
    return [{"role": "user", "content": [image_part(frame), {"type": "text", "text": _PROMPT}]}]


def _parse_answer(answer):
    normalized = answer.strip().upper()
    return normalized if normalized in _VIEWS else "OTHER"


_runner = TaskRunner(
    name="view",
    build_messages=_build_messages,
    parse_answer=_parse_answer,
)


async def run(frame, *, generation: GenerationOptions | None = None):
    return await _runner.run(frame, generation=generation)


def stream(frame, *, generation: GenerationOptions | None = None):
    return _runner.stream(frame, generation=generation)
