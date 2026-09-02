"""Decode the Qwen thinking tokens emitted by CARDEA."""

from enum import Enum


class OutputPhase(str, Enum):
    REASONING = "reasoning"
    ANSWER = "answer"


class ThinkStreamDecoder:
    """Split streamed text into reasoning and answer phases.

    A thinking response starts in the reasoning phase, not the answer phase. The
    chat template shipped with the model prefills ``<think>`` at the end of the
    prompt, so the completion begins part-way through the trace and its first
    delta carries no opening tag; only the closing ``</think>`` ever arrives. An
    opening tag is still accepted, for a server or template that emits one.

    Reasoning is emitted as it arrives so a UI can show it live. That means the
    phase has to be decided before the closing tag is in hand, which is what
    ``thinking`` is for: pass False only when the server genuinely suppresses the
    trace, otherwise the trace is filed as the answer.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, thinking: bool = True):
        self._phase = OutputPhase.REASONING if thinking else OutputPhase.ANSWER
        self._opening_seen = False
        # Tail withheld because it could be the start of a marker split across
        # two deltas ("</thi" + "nk>").
        self._pending = ""

    def feed(self, delta: str):
        """Return newly decoded ``(phase, text)`` pieces."""
        if not delta:
            return []

        text = self._pending + delta
        self._pending = ""
        pieces = []

        if self._phase is OutputPhase.REASONING:
            if not self._opening_seen:
                before, marker, after = text.partition(self._OPEN)
                if marker:
                    self._opening_seen = True
                    text = before + after

            reasoning, marker, answer = text.partition(self._CLOSE)
            if not marker:
                reasoning, self._pending = self._withhold_partial_marker(reasoning)
                if reasoning:
                    pieces.append((OutputPhase.REASONING, reasoning))
                return pieces

            self._phase = OutputPhase.ANSWER
            if reasoning:
                pieces.append((OutputPhase.REASONING, reasoning))
            text = answer

        if text:
            pieces.append((OutputPhase.ANSWER, text))
        return pieces

    def flush(self):
        """Emit anything still withheld, once the stream has ended."""
        if not self._pending:
            return []
        text, self._pending = self._pending, ""
        return [(self._phase, text)]

    @classmethod
    def _withhold_partial_marker(cls, text: str):
        """Split off a trailing fragment that could begin a marker."""
        longest = max(len(cls._OPEN), len(cls._CLOSE)) - 1
        for size in range(min(longest, len(text)), 0, -1):
            tail = text[-size:]
            if cls._CLOSE.startswith(tail) or cls._OPEN.startswith(tail):
                return text[:-size], tail
        return text, ""
