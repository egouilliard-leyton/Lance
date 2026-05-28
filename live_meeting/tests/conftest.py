"""Shared pytest fixtures for the live_meeting suite.

Heavy/inter-module imports happen lazily inside fixture bodies so test collection never depends
on modules that may not exist yet during incremental development.
"""

from __future__ import annotations

import pytest

from live_meeting.tests import fixtures as F


@pytest.fixture
def synth_video(tmp_path):
    """(path, intended_colors) for a tiny 8-frame @1fps mp4."""
    path = tmp_path / "meeting.mp4"
    colors = F.synth_mp4(path, n_frames=8, fps=1, size=(64, 64))
    return str(path), colors


@pytest.fixture
def synth_audio(tmp_path):
    """(path, n_samples) for an 8s mono 16kHz wav."""
    path = tmp_path / "meeting.wav"
    n = F.synth_wav(path, seconds=8.0, sr=16000)
    return str(path), n


@pytest.fixture
def fake_engine():
    from live_meeting.engine import FakeEngine

    return FakeEngine(qa_map={"what is happening": "A presentation is on screen."})


@pytest.fixture
def gt_spec(synth_video, synth_audio):
    from live_meeting.eval import ExpectedProactive, ExpectedQA, GroundTruthSpec

    vpath, _ = synth_video
    apath, _ = synth_audio
    return GroundTruthSpec(
        video_path=vpath,
        audio_path=apath,
        qa=[ExpectedQA(at_ts=1.0, query="what is happening", expected="presentation")],
        proactive=[ExpectedProactive(at_ts=3.0, kind="slide_change", tolerance_s=2.0)],
    )
