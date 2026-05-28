"""Tests for :class:`live_meeting.orchestrator.MeetingOrchestrator`.

CPU-only, no network: a :class:`RecordedFileTransport` (``realtime=False``) over synth fixtures feeds
a :class:`FakeEngine`. A fake monotonic ``clock`` makes latency measurements deterministic.
"""

from __future__ import annotations

import threading

from live_meeting.engine import FakeEngine
from live_meeting.events import ProactiveEvent
from live_meeting.orchestrator import MeetingOrchestrator, QueryRequest
from live_meeting.transport import RecordedFileTransport


def _transport(synth_video, synth_audio=None):
    vpath, _ = synth_video
    apath = synth_audio[0] if synth_audio else None
    return RecordedFileTransport(vpath, apath, fps=1.0, realtime=False)


def test_query_request_dataclass():
    qr = QueryRequest(ts=1.5, query="hi")
    assert qr.ts == 1.5 and qr.query == "hi"


def test_video_routing_counts(synth_video):
    engine = FakeEngine()
    orch = MeetingOrchestrator(_transport(synth_video), engine)
    orch.run()
    # 8-frame @1fps synth source, video-only -> exactly 8 push_video calls.
    assert engine.video_count == 8
    assert engine.audio_count == 0


def test_audio_routing_counts(synth_video, synth_audio):
    engine = FakeEngine()
    orch = MeetingOrchestrator(_transport(synth_video, synth_audio), engine)
    orch.run()
    assert engine.video_count == 8
    assert engine.audio_count > 0


def test_submit_query_before_run_is_served(synth_video):
    clock = iter([10.0, 10.25, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
                  99.0, 99.0, 99.0, 99.0, 99.0, 99.0])
    engine = FakeEngine(qa_map={"what is happening": "a slide"})
    orch = MeetingOrchestrator(
        _transport(synth_video), engine, clock=lambda: next(clock)
    )
    orch.submit_query("what is happening")
    transcript = orch.run()

    assert len(transcript.qa) == 1
    qa = transcript.qa[0]
    assert qa.query == "what is happening"
    assert qa.answer == "a slide"
    assert qa.latency_s >= 0
    # First event has ts 0.0 -> the query is attributed to it.
    assert qa.query_ts == 0.0


def test_proactive_event_surfaces_and_is_ordered(synth_video):
    engine = FakeEngine(
        proactive_script=[ProactiveEvent(ts=3.0, kind="slide_change", description="x")]
    )
    orch = MeetingOrchestrator(_transport(synth_video), engine)
    transcript = orch.run()

    assert len(transcript.proactive) == 1
    assert transcript.proactive[0].kind == "slide_change"
    # The event is only surfaced after the stream reaches ts >= 3.0 (FakeEngine gates on last_ts).
    ts = [p.ts for p in transcript.proactive]
    assert ts == sorted(ts)


def test_max_events_bounds_the_loop(synth_video):
    engine = FakeEngine()
    orch = MeetingOrchestrator(_transport(synth_video), engine)
    orch.run(max_events=2)
    assert engine.video_count == 2


def test_queries_from_another_thread_all_served(synth_video):
    engine = FakeEngine(default_answer="ok")
    orch = MeetingOrchestrator(_transport(synth_video), engine)

    queries = [f"q{i}" for i in range(5)]

    def submit_all():
        for q in queries:
            orch.submit_query(q)

    t = threading.Thread(target=submit_all)
    t.start()
    t.join()  # ensure all queries enqueued before run drains them

    transcript = orch.run()
    served = {qa.query for qa in transcript.qa}
    assert served == set(queries)
    assert len(transcript.qa) == 5


def test_end_session_always_called_even_on_error(synth_video):
    class Boom(FakeEngine):
        def push_video(self, frame):  # type: ignore[override]
            raise RuntimeError("boom")

    engine = Boom()
    orch = MeetingOrchestrator(_transport(synth_video), engine)
    try:
        orch.run()
    except RuntimeError:
        pass
    # end_session() flips started back to False even though run() raised.
    assert engine.started is False
