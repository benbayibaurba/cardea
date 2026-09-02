"""Select the best diagnostic frame from one clip."""

import re

from ..imaging import frame_to_square_pil, image_part
from ..outputs import KeyframeSelection as _KeyframeSelection
from ..prompts import format_instructions
from ..task import GenerationOptions, TaskRunner

__all__ = ["run", "stream"]

_EDGE = 128

_INSTRUCT = """You are an AI medical imaging assistant, specializing in helping physicians identify key diagnostic frames from coronary angiography (CAG) procedures.

## Task Context

You will be provided with a set of sequential frames extracted from a single DICOM video. **This sequence is confirmed to contain valid coronary angiography frames.**

## Objective

Your job is to analyze this sequence and **identify the key diagnostic frames**.
You must select and return the indexes of the frames that are **relatively** the most high-quality and diagnostically valuable within this specific sequence.

**Important Note on Relative Quality:**
"Diagnostic value" is **relative** to the provided video.
- You should aim to select the **best available frames** from this sequence.
- Even if the video contains noise or is not perfect, select the frames that show the clearest structure compared to the others.

## Criteria for Selection (High-Quality / Diagnostic Frames):

You should prioritize selecting frames that meet the following criteria:

    - **Adequate Opacification**: The contrast dye sufficiently fills and fully shows the coronary arteries (visualize the vessel tree clearly).

    - **Relevant Anatomy**: The image clearly displays the coronary arteries (LCA or RCA) rather than irrelevant structures (e.g., purely spine, diaphragm, or empty background).

    - **Proper Timing**: The frame captures the moment of maximum vessel filling (typically during diastole) without excessive motion blur.

## Criteria for Best Frame Selection (`best_frame_idx`):

From the selected frames, you must identify the single best frame based on the following priorities:

    1. **Full Vessel Opacification**: The contrast dye has completely filled the entire length of the vessel tree.

    2. **Peak Contrast Intensity**: The dye concentration is at its peak and has **not yet started to fade or washout**.

    3. **Complete Structure Visibility**: Due to camera panning or cardiac motion (heartbeat), the vessel might move out of the frame. You must select the frame where the **entire vessel is observable** within the field of view, avoiding frames where parts of the vessel are cut off.

## Selected Frame Indexes Output Format

{PARSER_PROMPT_INSTRUCT}
"""

_PROMPT = _INSTRUCT.format(
    PARSER_PROMPT_INSTRUCT=format_instructions(_KeyframeSelection)
)


def _build_messages(frames):
    content = [image_part(frame_to_square_pil(frame, _EDGE)) for frame in frames]
    header = "\n".join(f"img_idx: {index}\n<image>" for index in range(len(frames)))
    content.append({"type": "text", "text": header + _PROMPT})
    return [{"role": "user", "content": content}]


def _parse_answer(answer):
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    return _KeyframeSelection.model_validate_json(
        match.group(0) if match else answer
    )


_runner = TaskRunner(
    name="keyframe",
    build_messages=_build_messages,
    parse_answer=_parse_answer,
)


async def run(frames, *, generation: GenerationOptions | None = None):
    return await _runner.run(frames, generation=generation)


def stream(frames, *, generation: GenerationOptions | None = None):
    return _runner.stream(frames, generation=generation)
