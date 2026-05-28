"""The :class:`MeetingOrchestrator` — the integration layer that wires a transport to an engine.

This module is the conductor of the live_meeting pipeline. It pulls paced ``AVEvent`` objects off a
:class:`~live_meeting.transport.MeetingTransport`, routes them into an
:class:`~live_meeting.engine.UnderstandingEngine` (video -> ``push_video``, audio -> ``push_audio``),
serves user queries submitted (thread-safely) via :meth:`MeetingOrchestrator.submit_query`, and
surfaces model-initiated :class:`~live_meeting.events.ProactiveEvent` observations.

It is deliberately CPU-only and import-light: only stdlib (``queue``, ``threading``, ``time``) plus
the shared contracts are imported. The interleaving (drain queries, then drain due proactive events,
after EVERY transport event) is what keeps integration/eval tests deterministic.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ._logging import get_logger
from .engine import UnderstandingEngine
from .events import AudioChunk, QAResult, Transcript, VideoFrame
from .transport import MeetingTransport

logger = get_logger("live_meeting.orchestrator")


@dataclass
class QueryRequest:
    """A user query submitted at wall-clock-relative timestamp ``ts`` (seconds)."""

    ts: float
    query: str


class MeetingOrchestrator:
    """Route a transport's A/V stream into an engine, serving queries and proactive events.

    Parameters
    ----------
    transport:
        Source of paced, timestamp-ordered :class:`AVEvent` objects.
    engine:
        The :class:`UnderstandingEngine` that ingests A/V and answers questions.
    session_id:
        Identifier passed to ``engine.start_session``.
    proactive_poll_every:
        Reserved API knob (currently unused): proactive events are drained after every transport
        event so tests stay deterministic regardless of pacing.
    clock:
        Monotonic clock callable returning seconds (injectable for tests); used to measure
        per-query latency.
    """

    def __init__(
        self,
        transport: MeetingTransport,
        engine: UnderstandingEngine,
        session_id: str = "poc1",
        proactive_poll_every: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.engine = engine
        self.session_id = session_id
        self.proactive_poll_every = proactive_poll_every
        self._clock = clock
        self._queries: "queue.Queue[str]" = queue.Queue()
        self._transcript = Transcript()

    def submit_query(self, query: str) -> None:
        """Enqueue a user query to be answered after the next transport event (thread-safe)."""
        self._queries.put(query)

    def transcript(self) -> Transcript:
        """Return the collected :class:`Transcript` (mutated in place during :meth:`run`)."""
        return self._transcript

    def _drain_queries(self, event_ts: float) -> None:
        """Answer every currently-enqueued query, measuring per-query latency via the clock."""
        while True:
            try:
                q = self._queries.get_nowait()
            except queue.Empty:
                break
            t0 = self._clock()
            ans = self.engine.ask(q)
            lat = self._clock() - t0
            self._transcript.qa.append(
                QAResult(query_ts=event_ts, query=q, answer=ans, latency_s=lat)
            )

    def _drain_proactive(self) -> None:
        """Drain all currently-due proactive events from the engine."""
        while True:
            event = self.engine.poll_proactive()
            if event is None:
                break
            self._transcript.proactive.append(event)

    def run(self, max_events: Optional[int] = None) -> Transcript:
        """Drive the pipeline to completion (or until ``max_events`` events) and return the transcript.

        For each transport event: route it into the engine, then drain pending queries, then drain
        due proactive events. ``engine.end_session()`` is always called in a ``finally``.
        """
        self.engine.start_session(self.session_id)
        count = 0
        try:
            for ev in self.transport.stream():
                if isinstance(ev, VideoFrame):
                    self.engine.push_video(ev)
                elif isinstance(ev, AudioChunk):
                    self.engine.push_audio(ev)
                else:  # pragma: no cover - defensive; transports only yield AVEvent
                    logger.debug("ignoring unknown event type: %r", type(ev))

                self._drain_queries(ev.ts)
                self._drain_proactive()

                count += 1
                if max_events is not None and count >= max_events:
                    break
        finally:
            self.engine.end_session()
        return self._transcript
