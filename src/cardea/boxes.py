"""The box format CARDEA emits inside its <think> trace.

Boxes look like {"img_idx": 0, "bbox_2d": [x1, y1, x2, y2], "label": "..."},
with coordinates on a 0-1000 scale.
"""

import re
from dataclasses import dataclass

RATIO_BASE = 1000

# groups: image_n, x1, y1, x2, y2, and optional ref (label/caption)
BBOX_PATTERN = (
    r"\{(?=[^{}]*['\"]img_idx['\"]\s*:\s*(?P<image_n>\d+))"
    r"(?=[^{}]*['\"]bbox_2d['\"]\s*:\s*\[(?P<x1>\d+),\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\])"
    r"(?:(?=[^{}]*['\"](?:label|caption)['\"]\s*:\s*['\"](?P<ref>[^'\"]*)['\"])"
    r"|(?![^{}]*['\"](?!(?:img_idx|bbox_2d)['\"])[^'\"]+['\"]\s*:))[^{}]*\}"
)

_BOX = re.compile(BBOX_PATTERN)


@dataclass(frozen=True)
class BoundingBox:
    image_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None


def parse_boxes(text):
    """Return all valid Chain-of-Box anchors from a reasoning trace."""
    boxes = []
    for match in _BOX.finditer(text or ""):
        values = match.groupdict()
        box = BoundingBox(
            image_index=int(values["image_n"]),
            x1=int(values["x1"]),
            y1=int(values["y1"]),
            x2=int(values["x2"]),
            y2=int(values["y2"]),
            label=values.get("ref"),
        )
        if (
            all(0 <= value <= RATIO_BASE for value in (box.x1, box.y1, box.x2, box.y2))
            and box.x1 <= box.x2
            and box.y1 <= box.y2
        ):
            boxes.append(box)
    return tuple(boxes)


def has_cob(text):
    """True if a Chain-of-Box (CoB) box appears in the model's reasoning trace."""
    trace = (text or "").split("</think>")[0]
    return bool(parse_boxes(trace))
