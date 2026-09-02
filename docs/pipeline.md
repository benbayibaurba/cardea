# Pipeline walkthrough

This document maps the paper's two-pass CARDEA pipeline to the inference code
in this repository. All five tasks use the same served CARDEA model, with
task-specific input preparation, prompts, and output validation.

```text
DICOM clips
    -> keyframe selection
    -> view classification
    -> multi-view study curation (LCA + RCA)
    -> dominance + complexity + report
```

## Annotated implementation

```python
import asyncio

from cardea.imaging import (
    frame_to_square_pil,
    read_dcm_3d_array,
    subsample_frames,
)
from cardea.pipeline import STUDY_EDGE, pick_study_frames
from cardea.tasks import complexity, dominance, keyframe, report, view


async def screen_clip(path):
    # Pass 1a: sample one DICOM clip and select its best diagnostic frame.
    video = read_dcm_3d_array(path)
    if video is None:
        return None

    frames = subsample_frames(video)
    selection = (await keyframe.run(frames)).answer
    index = selection.best_frame_idx
    if not 0 <= index < len(frames):
        index = 0

    # Pass 1b: classify the selected frame as LCA, RCA, or OTHER.
    best_frame = frame_to_square_pil(frames[index], STUDY_EDGE)
    selected_view = (await view.run(best_frame)).answer
    return selected_view, best_frame


async def analyze_study(clip_paths):
    # Pass 1: select and classify one keyframe from every clip concurrently.
    screened = await asyncio.gather(*(screen_clip(path) for path in clip_paths))
    screened = [item for item in screened if item is not None]

    # Curate the multi-view study input: discard OTHER and require both LCA and RCA.
    lca = [frame for selected_view, frame in screened if selected_view == "LCA"]
    rca = [frame for selected_view, frame in screened if selected_view == "RCA"]
    study_frames, fail_reason = pick_study_frames(lca, rca)
    if not study_frames:
        print(fail_reason)
        return

    # Pass 2: interpret the same curated multi-view frames concurrently.
    results = await asyncio.gather(
        dominance.run(study_frames),
        complexity.run(study_frames),
        report.run(study_frames),
    )
    for name, result in zip(("dominance", "complexity", "report"), results):
        print(name, result.answer)


asyncio.run(analyze_study(["study/clip1.dcm", "study/clip2.dcm"]))
```

`subsample_frames()` keeps at most 50 frames per clip. Study curation requires
at least one LCA and one RCA keyframe and uses at most ten frames; when more
are available, selection preserves at least one frame from each view.
`run_pipeline()` performs this same composition and reports an incomplete
study through `StudyResult.fail_reason`.

## Task outputs

| Step | `TaskResult.answer` |
| --- | --- |
| `keyframe` | `KeyframeSelection` |
| `view` | `"LCA"`, `"RCA"`, or `"OTHER"` |
| `dominance` | `"Left Dominance"` or `"Right Dominance"` |
| `complexity` | `"Normal to Intermediate Complexity: SYNTAX score 0-32"` or `"High Complexity: SYNTAX score > 32"` |
| `report` | `CoronaryFindings` |

The two structured answers are Pydantic models defined in `cardea.outputs`:

```python
from pydantic import BaseModel


class KeyframeSelection(BaseModel):
    selected_frame_idxs: list[int]
    best_frame_idx: int


class CoronaryFindings(BaseModel):
    lm: str
    lad: str
    lcx: str
    rca: str
```

`selected_frame_idxs` contains the diagnostically useful indexes from the
sampled frames passed to the keyframe task. `best_frame_idx` identifies the
single best frame used by the next pipeline step. The report fields contain
the validated findings for the left main, LAD, LCX, and RCA, respectively.

Every completed task returns the same envelope:

- `reasoning`: decoded reasoning without thinking tags
- `answer_text`: final-answer text before task-specific validation
- `answer`: validated task output
- `raw_text`: unmodified endpoint response
- `boxes`: validated Chain-of-Box anchors parsed from the reasoning

## Chain-of-Box

For study-level tasks, `box.image_index` refers to a position in
`study_frames`, and coordinates use a 0-1000 scale:

```python
x1 = box.x1 / 1000 * image.width
y1 = box.y1 / 1000 * image.height
x2 = box.x2 / 1000 * image.width
y2 = box.y2 / 1000 * image.height
```

`run_pipeline()` performs one rollout per study-level task. The demo can
instead run multiple rollouts and display the first result containing a valid
Chain-of-Box anchor. If none contains a box, it falls back to the first
successful result. Resampling does not change clip screening or study
curation.

## Streaming

`stream()` accepts the same task input as `run()` and emits structured
reasoning and answer deltas followed by one completed result:

```python
from cardea.task import AnswerDelta, Completed, ReasoningDelta
from cardea.tasks import dominance

async for event in dominance.stream(study_frames):
    if isinstance(event, ReasoningDelta):
        print(event.text, end="")
    elif isinstance(event, AnswerDelta):
        print(event.text, end="")
    elif isinstance(event, Completed):
        result = event.result
```
