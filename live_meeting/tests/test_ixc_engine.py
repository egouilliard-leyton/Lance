"""Tests for ``IXCStreamingEngine`` against a mocked ``IXCBackendBoundary``.

Critically also asserts that importing ``live_meeting.ixc_engine`` pulls in NO heavy deps.
"""

import sys

import numpy as np
from unittest.mock import MagicMock

from live_meeting.events import AudioChunk, ProactiveEvent, VideoFrame
from live_meeting.ixc_engine import IXCBackendBoundary, IXCStreamingEngine


def _mock_backend() -> MagicMock:
    return MagicMock(spec=IXCBackendBoundary)


def _frame(ts: float) -> VideoFrame:
    return VideoFrame(ts=ts, image=np.zeros((2, 2, 3), dtype=np.uint8))


def test_import_is_heavy_dep_free():
    # Already imported at top of module; assert no heavy deps leaked in.
    import live_meeting.ixc_engine  # noqa: F401

    assert "torch" not in sys.modules
    assert "lmdeploy" not in sys.modules


def test_video_throttled_to_fps():
    backend = _mock_backend()
    eng = IXCStreamingEngine(backend, fps=1.0)
    eng.start_session("s1")

    # 30 frames spanning ts 0.0 .. ~2.9 (0.1s spacing) at fps=1 -> at most 4 put_frame calls.
    for i in range(30):
        eng.push_video(_frame(ts=i * 0.1))

    assert backend.put_frame.call_count <= 4
    # Sanity: first frame always forwarded.
    assert backend.put_frame.call_count >= 1


def test_video_first_frame_always_forwarded():
    backend = _mock_backend()
    eng = IXCStreamingEngine(backend, fps=1.0)
    eng.start_session("s1")
    eng.push_video(_frame(ts=100.0))
    backend.put_frame.assert_called_once()
    (image_arg, ts_arg), _ = backend.put_frame.call_args
    assert ts_arg == 100.0


def test_ask_sets_query_then_submits_with_returned_imgs():
    backend = _mock_backend()
    sentinel_imgs = ["img0", "img1"]
    backend.set_query.return_value = sentinel_imgs
    backend.submit_mm_query.return_value = "the answer"

    eng = IXCStreamingEngine(backend)
    eng.start_session("s1")
    result = eng.ask("what is on the slide?")

    backend.set_query.assert_called_once_with("what is on the slide?")
    backend.submit_mm_query.assert_called_once_with("what is on the slide?", sentinel_imgs)
    assert result == "the answer"


def test_poll_proactive_delegates_to_backend():
    backend = _mock_backend()
    event = ProactiveEvent(ts=1.0, kind="slide_change", description="advanced")
    backend.poll_event.return_value = event

    eng = IXCStreamingEngine(backend)
    eng.start_session("s1")
    assert eng.poll_proactive() is event
    backend.poll_event.assert_called_once()


def test_push_audio_forwards_pcm_unchanged():
    backend = _mock_backend()
    eng = IXCStreamingEngine(backend, audio_sample_rate=16000)
    eng.start_session("s1")

    pcm = b"\x01\x02\x03\x04"
    eng.push_audio(AudioChunk(ts=0.5, pcm=pcm, sample_rate=16000))

    backend.put_audio.assert_called_once_with(pcm, 0.5, 16000)
    # Exact same bytes object forwarded, not a copy/transform.
    (pcm_arg, _ts, _sr), _ = backend.put_audio.call_args
    assert pcm_arg is pcm


def test_start_and_end_delegate_to_backend():
    backend = _mock_backend()
    eng = IXCStreamingEngine(backend)
    eng.start_session("sess-7")
    backend.start.assert_called_once_with("sess-7")
    eng.end_session()
    backend.stop.assert_called_once()
