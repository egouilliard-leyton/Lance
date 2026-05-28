import dataclasses

import numpy as np
import pytest

from live_meeting.events import (
    AudioChunk,
    ProactiveEvent,
    QAResult,
    Transcript,
    VideoFrame,
)


def test_frozen_immutability():
    pe = ProactiveEvent(ts=1.0, kind="slide_change", description="slide advanced")
    assert pe.confidence == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        pe.kind = "other"  # type: ignore[misc]


def test_av_event_construction():
    vf = VideoFrame(ts=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8), participant="alice")
    ac = AudioChunk(ts=0.5, pcm=b"\x00\x00", sample_rate=16000)
    assert vf.participant == "alice"
    assert vf.image.shape == (2, 2, 3)
    assert ac.sample_rate == 16000


def test_transcript_round_trip():
    t = Transcript(
        qa=[QAResult(query_ts=1.0, query="q", answer="a", latency_s=0.5)],
        proactive=[ProactiveEvent(ts=2.0, kind="k", description="d", confidence=0.9)],
    )
    d = t.to_dict()
    assert d["qa"][0]["query"] == "q"
    assert d["proactive"][0]["confidence"] == 0.9
    assert Transcript.from_dict(d) == t


def test_transcript_empty_round_trip():
    assert Transcript.from_dict(Transcript().to_dict()) == Transcript()
