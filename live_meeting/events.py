"""Typed audio/video and output events exchanged across the live_meeting pipeline.

These are the shared contracts every other module depends on. They import nothing heavy
(numpy is referenced only in type annotations, under TYPE_CHECKING).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    """A single decoded video frame at wall-clock-relative timestamp ``ts`` (seconds)."""

    ts: float
    image: "np.ndarray"  # HxWx3 uint8 RGB
    participant: Optional[str] = None


@dataclass(frozen=True)
class AudioChunk:
    """A block of PCM audio (s16le mono by default) at timestamp ``ts`` (seconds)."""

    ts: float
    pcm: bytes
    sample_rate: int = 16000


@dataclass(frozen=True)
class ProactiveEvent:
    """A model-initiated observation surfaced without an explicit user query."""

    ts: float
    kind: str
    description: str
    confidence: float = 1.0


@dataclass(frozen=True)
class QAResult:
    """The answer to a query asked at ``query_ts``, with measured end-to-end latency."""

    query_ts: float
    query: str
    answer: str
    latency_s: float


@dataclass
class Transcript:
    """Collected outputs of a meeting session: answered queries + proactive events."""

    qa: List[QAResult] = field(default_factory=list)
    proactive: List[ProactiveEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "qa": [asdict(q) for q in self.qa],
            "proactive": [asdict(p) for p in self.proactive],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            qa=[QAResult(**q) for q in d.get("qa", [])],
            proactive=[ProactiveEvent(**p) for p in d.get("proactive", [])],
        )


AVEvent = Union[VideoFrame, AudioChunk]
