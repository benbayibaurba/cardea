"""Structured coronary findings over the major branches."""

import json

from ..imaging import image_part
from ..outputs import CoronaryFindings as _CoronaryFindings
from ..prompts import GROUNDING_PROMPT, format_instructions
from ..task import GenerationOptions, TaskRunner

__all__ = ["run", "stream"]


_INSTRUCT = """Analyze the provided coronary angiography images and generate a structured clinical report.

# Clinical Writing Guidelines
When describing the findings for each vessel, you MUST be specific and avoid vague qualitative statements like "moderate" or "severe" without numerical values.
For every vessel field, explicitly specify:
1. The exact anatomical segment (e.g., proximal, mid, distal, or specific branches like D1, OM1).
2. The severity of the stenosis MUST be described using estimated percentages or percentage ranges (e.g., "70% stenosis", "50-80% narrowing", "100% occlusion"). Do NOT use only qualitative words.
3. Special Cases: If all segments of the vessel are completely normal with no disease, you can simply write "normal". If the vessel is anatomically absent or not visualized, you can simply write "nan".

Examples of Good Writing:
- [Lesion present]: "The proximal segment is normal (0%), but there is a severe 80-90% stenosis in the mid segment, and the distal segment shows a mild 20-30% narrowing."
- [Completely Normal]: "normal"
- [Absent/Not Visualized]: "nan"

Examples of Bad Writing (DO NOT DO THIS):
- "Severe stenosis." (Terrible: completely misses both the exact segment and the numerical percentage)
- "There is a stenosis in the middle." (Bad: uses informal language, missing exact segment name, missing percentage)
- "Abnormal." (Unacceptable: completely vague and useless)

# Vessel Mapping
Map your observations to the following JSON fields based on the vessel segment numbers:
- "lad": Include segments 6 (proximal), 7 (mid), 8 (distal), 9 (1st diagonal), and 10 (2nd diagonal).
- "lcx": Include segments 11 (proximal), 13 (mid), 14/15 (distal), and 12a/12b (OM/Obtuse Marginal).
- "rca": Include segments 1 (proximal), 2 (mid), 3 (distal), 4 (PDA), and 16 (PLB).

# Formatting Requirement
Your final output MUST be a valid, parsable JSON object matching the requested schema. Do NOT write any conversational text before or after the JSON. Start your response directly with the `{` character and end with `}`.
"""


def _build_messages(frames):
    text = (
        f"{GROUNDING_PROMPT}\n{'<image>' * len(frames)}\n\n"
        f"{_INSTRUCT}\n"
        f"### OUTPUT FORMAT\n{format_instructions(_CoronaryFindings)}"
    )
    return [{"role": "user", "content": [image_part(frame) for frame in frames] + [{"type": "text", "text": text}]}]


def _parse_answer(answer):
    start, end = answer.find("{"), answer.rfind("}")
    payload = answer[start:end + 1] if start != -1 and end >= start else answer
    return _CoronaryFindings.model_validate(json.loads(payload))


_runner = TaskRunner(
    name="report",
    build_messages=_build_messages,
    parse_answer=_parse_answer,
)


async def run(frames, *, generation: GenerationOptions | None = None):
    return await _runner.run(frames, generation=generation)


def stream(frames, *, generation: GenerationOptions | None = None):
    return _runner.stream(frames, generation=generation)
