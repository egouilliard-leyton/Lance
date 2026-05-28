"""Tests for :class:`live_meeting.transport.RecallAITransport`.

The websocket is fully mocked via an injected ``ws_factory`` returning a scripted message
iterator. A tiny PNG is synthesized in-test with Pillow (no binaries committed).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from live_meeting.events import AudioChunk, VideoFrame
from live_meeting.transport import RecallAITransport


def _png_b64(color=(220, 20, 20), size=(8, 6)) -> str:
    """Return base64-encoded PNG bytes of a solid-color (h=size[1], w=size[0]) RGB image."""
    from PIL import Image

    w, h = size
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pcm_b64(nbytes=512) -> str:
    return base64.b64encode(b"\x01\x02" * (nbytes // 2)).decode("ascii")


class _FakeWS:
    """Iterable scripted websocket; records whether close() was called."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    def __iter__(self):
        yield from self._messages

    def close(self):
        self.closed = True


def test_empty_api_key_raises():
    with pytest.raises(ValueError):
        RecallAITransport(api_key="", bot_id="b", ws_url="wss://x")


def test_parse_message_png_to_video_frame():
    t = RecallAITransport(api_key="k", bot_id="b", ws_url="wss://x")
    msg = {
        "kind": "video_separate_png",
        "ts": 1.5,
        "data": _png_b64(color=(20, 200, 20), size=(8, 6)),
        "participant": "alice",
    }
    ev = t._parse_message(msg)
    assert isinstance(ev, VideoFrame)
    assert ev.ts == 1.5
    assert ev.participant == "alice"
    assert ev.image.shape == (6, 8, 3)  # (H, W, 3)
    assert ev.image.dtype == np.uint8


def test_parse_message_audio_to_audio_chunk():
    t = RecallAITransport(api_key="k", bot_id="b", ws_url="wss://x")
    msg = {"kind": "audio", "ts": 0.25, "data": _pcm_b64(512), "sample_rate": 16000}
    ev = t._parse_message(msg)
    assert isinstance(ev, AudioChunk)
    assert ev.ts == 0.25
    assert ev.sample_rate == 16000
    assert ev.pcm == b"\x01\x02" * 256


def test_parse_message_unknown_kind_returns_none():
    t = RecallAITransport(api_key="k", bot_id="b", ws_url="wss://x")
    assert t._parse_message({"kind": "something_else", "ts": 0.0}) is None
    assert t._parse_message({"ts": 0.0}) is None
    # The documented h264 stub also yields None (PyAV not installed).
    assert t._parse_message({"kind": "video_separate_h264", "ts": 0.0, "data": ""}) is None


def test_stream_over_scripted_ws_yields_then_stops():
    messages = [
        {"kind": "audio", "ts": 0.0, "data": _pcm_b64(256), "sample_rate": 16000},
        {"kind": "video_separate_png", "ts": 0.5, "data": _png_b64(size=(4, 4))},
        {"kind": "noise", "ts": 0.6},  # unknown -> skipped
        {"kind": "audio", "ts": 1.0, "data": _pcm_b64(256), "sample_rate": 16000},
    ]
    fake = _FakeWS(messages)

    captured = {}

    def factory(api_key, ws_url):
        captured["api_key"] = api_key
        captured["ws_url"] = ws_url
        return fake

    t = RecallAITransport(
        api_key="secret", bot_id="bot1", ws_url="wss://example", ws_factory=factory
    )
    events = list(t.stream())

    # Factory received credentials/url.
    assert captured == {"api_key": "secret", "ws_url": "wss://example"}

    # Unknown kind dropped; remaining 3 events yielded in script order.
    assert len(events) == 3
    assert isinstance(events[0], AudioChunk) and events[0].ts == 0.0
    assert isinstance(events[1], VideoFrame) and events[1].ts == 0.5
    assert isinstance(events[2], AudioChunk) and events[2].ts == 1.0

    # Stream stops cleanly when the ws is exhausted.
    t.close()
    assert fake.closed is True


def test_close_is_safe_without_stream():
    t = RecallAITransport(api_key="k", bot_id="b", ws_url="wss://x")
    # close() before any stream() must not raise.
    t.close()


def test_context_manager_closes_ws():
    fake = _FakeWS([{"kind": "audio", "ts": 0.0, "data": _pcm_b64(64)}])
    with RecallAITransport(
        api_key="k", bot_id="b", ws_url="wss://x", ws_factory=lambda a, u: fake
    ) as t:
        events = list(t.stream())
    assert len(events) == 1
    assert fake.closed is True
