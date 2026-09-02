"""Public output contracts for CARDEA inference tasks."""

from typing import Literal

from pydantic import BaseModel, Field

Dominance = Literal["Left Dominance", "Right Dominance"]
Complexity = Literal[
    "Normal to Intermediate Complexity: SYNTAX score 0-32",
    "High Complexity: SYNTAX score > 32",
]
View = Literal["LCA", "RCA", "OTHER"]


class KeyframeSelection(BaseModel):
    # This description is part of the training-time schema prompt. Keep verbatim.
    """selected high-quality keyframe indexes (不允許回傳空)."""

    selected_frame_idxs: list[int] = Field(
        description="A list of frame indexes from the sequence that are considered "
        "diagnostically valuable and high quality."
    )
    best_frame_idx: int = Field(
        description="The index of the single best frame selected from "
        "'selected_frame_idxs'. This frame must have full vessel opacification, "
        "peak contrast, and complete structure visibility."
    )


class CoronaryFindings(BaseModel):
    lm: str = Field(
        description="Observations regarding the Left Main (segment 5). You MUST include estimated stenosis percentages (e.g., 0%, 50-70...%)."
    )
    lad: str = Field(
        description="Observations regarding LAD (segments 6, 7, 8) and diagonals (9, 10). You MUST include estimated stenosis percentages (e.g., 0%, 50-70%...)."
    )
    lcx: str = Field(
        description="Observations regarding LCX (segments 11, 13, 14, 15) and OM/Obtuse Marginal (segments 12a/12b). You MUST include estimated stenosis percentages (e.g., 0%, 50-70%...)."
    )
    rca: str = Field(
        description="Observations regarding RCA (segments 1, 2, 3), PDA (4), and PLB (16). You MUST include estimated stenosis percentages (e.g., 0%, 50-70%...)."
    )


__all__ = [
    "Complexity",
    "CoronaryFindings",
    "Dominance",
    "KeyframeSelection",
    "View",
]
