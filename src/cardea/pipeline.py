"""Wire the stages into the full run: clips in, study findings out."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from .imaging import frame_to_square_pil, read_dcm_3d_array, subsample_frames
from .outputs import Complexity, CoronaryFindings, Dominance, View
from .task import TaskResult
from .tasks import complexity, dominance, keyframe, report, view

STUDY_EDGE = 512      # resolution the view and study-level tasks see
MAX_STUDY_FRAMES = 10  # token budget for the study-level pass


@dataclass
class ClipResult:
    path: str
    best_frame: object | None
    view: View | None


async def analyze_clip(path):
    """Keyframe selection then view classification for one clip."""
    video = read_dcm_3d_array(path)
    if video is None:
        return ClipResult(path, None, None)
    frames = subsample_frames(video)
    pil = [frame_to_square_pil(f, STUDY_EDGE) for f in frames]

    keyframe_result = await keyframe.run(frames)
    selection = keyframe_result.answer
    idx = selection.best_frame_idx if 0 <= selection.best_frame_idx < len(pil) else 0
    view_result = await view.run(pil[idx])
    return ClipResult(path, pil[idx], view_result.answer)


def pick_study_frames(lca, rca, seed=None):
    """Choose the frames for the study pass from the LCA and RCA keyframes.

    Needs at least one of each view; otherwise returns ([], note) explaining what
    is missing. If there are more than MAX_STUDY_FRAMES, keep one random LCA and
    one random RCA, then sample the rest at random (without replacement). Pass a
    `seed` for a reproducible selection.
    """
    if not lca or not rca:
        if not lca and not rca:
            return [], "No LCA or RCA view was identified."
        missing = "LCA" if not lca else "RCA"
        return [], f"No {missing} view identified; both an LCA and an RCA view are needed."

    lca, rca = list(lca), list(rca)
    if len(lca) + len(rca) <= MAX_STUDY_FRAMES:
        return lca + rca, ""

    rng = random.Random(seed)
    keep = [lca.pop(rng.randrange(len(lca))), rca.pop(rng.randrange(len(rca)))]
    keep += rng.sample(lca + rca, MAX_STUDY_FRAMES - 2)
    rng.shuffle(keep)
    return keep, ""


async def select_study_frames(paths, seed=None):
    results = await asyncio.gather(*(analyze_clip(p) for p in paths))
    lca = [r.best_frame for r in results if r.view == "LCA" and r.best_frame is not None]
    rca = [r.best_frame for r in results if r.view == "RCA" and r.best_frame is not None]
    return pick_study_frames(lca, rca, seed)


@dataclass
class StudyResult:
    frames: list
    dominance: TaskResult[Dominance] | None = None
    report: TaskResult[CoronaryFindings] | None = None
    complexity: TaskResult[Complexity] | None = None
    fail_reason: str | None = None


async def run_pipeline(paths, seed=None):
    frames, note = await select_study_frames(paths, seed)
    if not frames:
        return StudyResult([], fail_reason=note)
    dom, rep, cx = await asyncio.gather(
        dominance.run(frames),
        report.run(frames),
        complexity.run(frames),
    )
    return StudyResult(frames, dom, rep, cx)
