"""End-to-end CPU integration: RecordedFileTransport -> MeetingOrchestrator -> FakeEngine.

No network, no GPU. Synth video+audio fixtures drive the full pipeline; we assert the produced
transcript shape and that all timestamps are non-decreasing.
"""

from __future__ import annotations

from live_meeting.engine import FakeEngine
from live_meeting.events import ProactiveEvent
from live_meeting.orchestrator import MeetingOrchestrator
from live_meeting.transport import RecordedFileTransport


def test_full_pipeline_cpu(synth_video, synth_audio):
    vpath, _ = synth_video
    apath, _ = synth_audio

    transport = RecordedFileTransport(vpath, apath, fps=1.0, realtime=False)
    engine = FakeEngine(
        qa_map={"q1": "answer one", "q2": "answer two"},
        # ts=2.0 falls within the 0..7s stream, so it is surfaced exactly once.
        proactive_script=[ProactiveEvent(ts=2.0, kind="slide_change", description="advanced")],
    )
    orch = MeetingOrchestrator(transport, engine)
    orch.submit_query("q1")
    orch.submit_query("q2")

    transcript = orch.run()

    assert len(transcript.qa) == 2
    assert {qa.answer for qa in transcript.qa} == {"answer one", "answer two"}
    assert len(transcript.proactive) == 1
    assert transcript.proactive[0].kind == "slide_change"

    # Q/A query_ts non-decreasing.
    qa_ts = [qa.query_ts for qa in transcript.qa]
    assert qa_ts == sorted(qa_ts)
    # Proactive ts non-decreasing.
    pro_ts = [p.ts for p in transcript.proactive]
    assert pro_ts == sorted(pro_ts)
