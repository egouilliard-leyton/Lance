"""Tests for :class:`live_meeting.transport.RecordedFileTransport`.

Inputs are synthesized at test time via the ``synth_video`` / ``synth_audio`` conftest fixtures
(no binaries committed). Frame identity is checked by *dominant color* classification
(``nearest_color_index``) rather than exact pixels, because mp4/H.264 is lossy.
"""

from __future__ import annotations

import sys

from live_meeting.events import AudioChunk, VideoFrame
from live_meeting.tests import fixtures as F
from live_meeting.transport import RecordedFileTransport


def test_import_is_light():
    """Importing transport must not pull in torch."""
    import live_meeting.transport  # noqa: F401

    assert "torch" not in sys.modules


def test_sampled_frame_indices_count(synth_video):
    vpath, colors = synth_video  # 8-frame @1fps source
    t = RecordedFileTransport(vpath, fps=1.0, realtime=False)
    try:
        indices = t._sampled_frame_indices()
    finally:
        t.close()
    assert len(indices) == 8
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7]


def test_deterministic_video_frames(synth_video):
    vpath, colors = synth_video
    t = RecordedFileTransport(vpath, fps=1.0, realtime=False)
    try:
        frames = [ev for ev in t.stream() if isinstance(ev, VideoFrame)]
    finally:
        t.close()

    assert len(frames) == 8

    # Monotonically increasing timestamps, one per second.
    ts = [f.ts for f in frames]
    assert ts == sorted(ts)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts == [float(i) for i in range(8)]

    # Per-frame dominant color matches the intended palette color via nearest_color_index.
    for i, frame in enumerate(frames):
        expected_idx = F.palette().index(F.frame_color(i))
        assert F.nearest_color_index(frame.image) == expected_idx
        assert frame.image.ndim == 3 and frame.image.shape[2] == 3
        assert frame.image.dtype.kind == "u"


def test_audio_chunks(synth_audio):
    apath, n_samples = synth_audio
    # video_path is required but audio-only iteration is what we assert here; use a tiny video too.
    t = RecordedFileTransport(
        video_path=apath,  # unused for audio iteration; reader is lazy
        audio_path=apath,
        fps=1.0,
        realtime=False,
        audio_block_bytes=4096,
    )
    try:
        chunks = list(t._iter_audio())
    finally:
        t.close()

    assert chunks, "expected at least one audio chunk"
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert all(c.sample_rate == 16000 for c in chunks)
    assert all(len(c.pcm) <= 4096 for c in chunks)
    # Timestamps are non-decreasing and start at 0.
    ts = [c.ts for c in chunks]
    assert ts == sorted(ts)
    assert ts[0] == 0.0
    # 8s of mono s16le @16kHz == 8 * 16000 * 2 bytes total.
    assert sum(len(c.pcm) for c in chunks) == n_samples * 2


def test_merged_stream_globally_ordered(synth_video, synth_audio):
    vpath, _ = synth_video
    apath, _ = synth_audio
    t = RecordedFileTransport(
        vpath, audio_path=apath, fps=1.0, realtime=False, audio_block_bytes=4096
    )
    try:
        events = list(t.stream())
    finally:
        t.close()

    # Mixed video + audio events present.
    assert any(isinstance(e, VideoFrame) for e in events)
    assert any(isinstance(e, AudioChunk) for e in events)

    # Globally non-decreasing timestamps across the mixed stream.
    ts = [e.ts for e in events]
    assert ts == sorted(ts)


def test_realtime_pacing_uses_injected_clock_and_sleep(synth_video):
    vpath, _ = synth_video

    sleeps: list[float] = []

    # Fake clock that does NOT advance with sleep -> proves no real wall time passes.
    fake_now = {"t": 100.0}

    def fake_clock() -> float:
        return fake_now["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # Intentionally do NOT advance fake_now: we are not advancing real time either.

    import time as _time

    t = RecordedFileTransport(
        vpath, fps=1.0, realtime=True, clock=fake_clock, sleep=fake_sleep
    )
    wall_start = _time.monotonic()
    try:
        frames = list(t.stream())
    finally:
        t.close()
    wall_elapsed = _time.monotonic() - wall_start

    # One sleep per emitted event.
    assert len(sleeps) == len(frames) == 8
    # Sleeps approximate the per-frame interval (clock never advanced, so each delay == ev.ts).
    assert sleeps == [float(i) for i in range(8)]
    # The fake sleep recorded the requested delays but real wall time did NOT advance ~8s.
    assert wall_elapsed < 1.0
