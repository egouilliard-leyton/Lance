"""Behavioral tests for ``FakeEngine`` (the dependency-free understanding double)."""

import numpy as np
import pytest

from live_meeting.engine import FakeEngine
from live_meeting.events import AudioChunk, ProactiveEvent, VideoFrame


def _frame(ts: float) -> VideoFrame:
    return VideoFrame(ts=ts, image=np.zeros((2, 2, 3), dtype=np.uint8))


def _audio(ts: float) -> AudioChunk:
    return AudioChunk(ts=ts, pcm=b"\x00\x00", sample_rate=16000)


def test_ask_exact_match():
    eng = FakeEngine(qa_map={"what is happening": "A presentation is on screen."})
    eng.start_session("s1")
    assert eng.ask("what is happening") == "A presentation is on screen."


def test_ask_substring_match_case_insensitive():
    eng = FakeEngine(qa_map={"slide": "There is a slide."})
    eng.start_session("s1")
    assert eng.ask("Tell me about the SLIDE on screen") == "There is a slide."


def test_ask_default_answer():
    eng = FakeEngine(qa_map={"slide": "There is a slide."}, default_answer="dunno")
    eng.start_session("s1")
    assert eng.ask("what is the weather") == "dunno"


def test_default_answer_default_value():
    eng = FakeEngine()
    eng.start_session("s1")
    assert eng.ask("anything") == "(no answer)"


def test_ask_before_start_raises():
    eng = FakeEngine(qa_map={"q": "a"})
    with pytest.raises(RuntimeError):
        eng.ask("q")


def test_start_and_end_session_flags():
    eng = FakeEngine()
    assert eng.started is False
    eng.start_session("sess-42")
    assert eng.started is True
    assert eng.session_id == "sess-42"
    eng.end_session()
    assert eng.started is False


def test_push_counters_increment_and_track_ts():
    eng = FakeEngine()
    eng.start_session("s1")
    assert eng.video_count == 0
    assert eng.audio_count == 0
    assert eng.last_ts is None

    eng.push_video(_frame(0.0))
    eng.push_video(_frame(1.0))
    eng.push_audio(_audio(0.5))

    assert eng.video_count == 2
    assert eng.audio_count == 1
    assert eng.last_ts == 1.0

    eng.push_audio(_audio(2.5))
    assert eng.audio_count == 2
    assert eng.last_ts == 2.5


def test_poll_proactive_gated_by_last_ts_then_ordered_then_none():
    script = [
        ProactiveEvent(ts=1.0, kind="a", description="first"),
        ProactiveEvent(ts=3.0, kind="b", description="second"),
    ]
    eng = FakeEngine(proactive_script=script)
    eng.start_session("s1")

    # Nothing pushed yet -> gated.
    assert eng.poll_proactive() is None

    # Push to ts=1.0 -> first event becomes due (and only one per call).
    eng.push_video(_frame(1.0))
    first = eng.poll_proactive()
    assert first is script[0]
    # Second not yet due (last_ts=1.0 < 3.0).
    assert eng.poll_proactive() is None

    # Advance to ts=3.0 -> second due.
    eng.push_audio(_audio(3.0))
    second = eng.poll_proactive()
    assert second is script[1]

    # Script exhausted.
    assert eng.poll_proactive() is None


def test_poll_proactive_empty_script():
    eng = FakeEngine()
    eng.start_session("s1")
    eng.push_video(_frame(10.0))
    assert eng.poll_proactive() is None


def test_poll_proactive_emits_one_per_call_when_all_due():
    script = [
        ProactiveEvent(ts=0.0, kind="a", description="first"),
        ProactiveEvent(ts=0.0, kind="b", description="second"),
    ]
    eng = FakeEngine(proactive_script=script)
    eng.start_session("s1")
    eng.push_video(_frame(5.0))
    assert eng.poll_proactive() is script[0]
    assert eng.poll_proactive() is script[1]
    assert eng.poll_proactive() is None
