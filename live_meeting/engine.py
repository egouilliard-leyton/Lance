"""The ``UnderstandingEngine`` interface plus a dependency-free ``FakeEngine``.

``UnderstandingEngine`` is the clean seam every transport/orchestrator talks to. Concrete
streaming backends (e.g. IXC2.5-OmniLive, see :mod:`live_meeting.ixc_engine`) implement it
behind heavy/lazy imports. ``FakeEngine`` is a pure-Python double used to make
orchestrator/integration tests deterministic without any model, GPU or network.

This module imports nothing heavy: only the shared contracts from :mod:`live_meeting.events`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .events import AudioChunk, ProactiveEvent, VideoFrame


class UnderstandingEngine(ABC):
    """Streaming multimodal understanding seam: push A/V in, ask questions, poll observations."""

    @abstractmethod
    def start_session(self, session_id: str) -> None:
        """Begin a new session identified by ``session_id``."""

    @abstractmethod
    def push_video(self, frame: VideoFrame) -> None:
        """Feed a single decoded video frame into the engine."""

    @abstractmethod
    def push_audio(self, chunk: AudioChunk) -> None:
        """Feed a single PCM audio chunk into the engine."""

    @abstractmethod
    def ask(self, query: str) -> str:
        """Ask a question about what's been observed so far; return the answer text."""

    @abstractmethod
    def poll_proactive(self) -> Optional[ProactiveEvent]:
        """Return the next model-initiated event if one is ready, else ``None``."""

    @abstractmethod
    def end_session(self) -> None:
        """Tear down the current session."""


class FakeEngine(UnderstandingEngine):
    """Deterministic, dependency-free ``UnderstandingEngine`` for tests.

    Answers come from ``qa_map`` (exact match, then case-insensitive substring match, then
    ``default_answer``). Proactive events come from ``proactive_script`` and are emitted in
    order, one per ``poll_proactive`` call, but only once enough A/V has been pushed to reach
    each event's timestamp (gated on ``last_ts``). The gating keeps integration tests
    deterministic regardless of how the orchestrator interleaves pushes and polls.
    """

    def __init__(
        self,
        qa_map: Optional[Dict[str, str]] = None,
        proactive_script: Optional[List[ProactiveEvent]] = None,
        default_answer: str = "(no answer)",
    ) -> None:
        self.qa_map: Dict[str, str] = dict(qa_map) if qa_map else {}
        self.proactive_script: List[ProactiveEvent] = list(proactive_script) if proactive_script else []
        self.default_answer = default_answer

        self.started: bool = False
        self.session_id: Optional[str] = None
        self.video_count: int = 0
        self.audio_count: int = 0
        self.last_ts: Optional[float] = None
        self._proactive_idx: int = 0

    def start_session(self, session_id: str) -> None:
        self.started = True
        self.session_id = session_id

    def push_video(self, frame: VideoFrame) -> None:
        self.video_count += 1
        self._track_ts(frame.ts)

    def push_audio(self, chunk: AudioChunk) -> None:
        self.audio_count += 1
        self._track_ts(chunk.ts)

    def ask(self, query: str) -> str:
        if not self.started:
            raise RuntimeError("ask() called before start_session()")
        if query in self.qa_map:
            return self.qa_map[query]
        lowered = query.lower()
        for key, value in self.qa_map.items():
            if key.lower() in lowered:
                return value
        return self.default_answer

    def poll_proactive(self) -> Optional[ProactiveEvent]:
        if self._proactive_idx >= len(self.proactive_script):
            return None
        event = self.proactive_script[self._proactive_idx]
        if self.last_ts is None or self.last_ts < event.ts:
            return None
        self._proactive_idx += 1
        return event

    def end_session(self) -> None:
        self.started = False

    def _track_ts(self, ts: float) -> None:
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts
