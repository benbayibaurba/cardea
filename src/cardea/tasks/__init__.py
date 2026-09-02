"""Typed CARDEA inference tasks.

Each module exposes the same public execution surface:

    await task.run(input, generation=...)
    async for event in task.stream(input, generation=...): ...
"""

from . import complexity, dominance, keyframe, report, view

__all__ = ["complexity", "dominance", "keyframe", "report", "view"]
