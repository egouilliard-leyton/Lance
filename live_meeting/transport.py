"""Meeting transports: turn a video+audio source into a paced, ts-ordered ``AVEvent`` stream.

This module is the boundary between the live_meeting pipeline and the outside world (a recorded
file on disk, or the Recall.ai websocket). It is deliberately CPU-only and import-light:

* importing this module must NOT import torch/transformers/lmdeploy and must NOT open a socket;
* ``decord`` is lazily imported inside :class:`RecordedFileTransport` methods;
* ``websockets`` is lazily imported inside the default Recall websocket factory;
* ``PIL`` / ``imageio`` (PNG decode) is lazily imported inside :meth:`RecallAITransport._parse_message`.

``numpy`` is imported at top because it is light; ``np.ndarray`` type references use the
``from __future__ import annotations`` + ``TYPE_CHECKING`` pattern so no runtime cost is added.
"""

from __future__ import annotations

import time
import wave
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional

from live_meeting._logging import get_logger
from live_meeting.events import AudioChunk, AVEvent, VideoFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = get_logger("live_meeting.transport")


class MeetingTransport(ABC):
    """Abstract source of paced, timestamp-ordered audio/video events.

    Subclasses yield :class:`~live_meeting.events.AVEvent` instances from :meth:`stream`, ordered
    by their ``ts`` field (non-decreasing). Implementations are context managers: ``__enter__``
    returns ``self`` and ``__exit__`` always calls :meth:`close`.
    """

    def __enter__(self) -> "MeetingTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @abstractmethod
    def stream(self) -> Iterator[AVEvent]:
        """Yield paced, timestamp-ORDERED :class:`AVEvent` objects."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources (file readers, sockets)."""
        raise NotImplementedError


class RecordedFileTransport(MeetingTransport):
    """Replay a recorded video (+ optional wav audio) as a paced ``AVEvent`` stream.

    Video is decoded with decord (mirroring ``data/datasets_custom/validation_dataset.py``); audio
    is read with the stdlib :mod:`wave` module. The two streams are merged and yielded globally
    timestamp-ordered. When ``realtime=True`` events are paced against the injected ``clock`` /
    ``sleep`` callables so the wall-clock spacing matches the event timestamps; when
    ``realtime=False`` (the test mode) events are yielded back-to-back with no sleeping.

    Parameters
    ----------
    video_path:
        Path to a video file decodable by decord.
    audio_path:
        Optional path to a mono PCM ``wav`` file. If ``None``, no audio is emitted.
    fps:
        Target sampling rate (frames per second) drawn from the source video. See
        :meth:`_sampled_frame_indices` for the exact formula.
    realtime:
        If ``True`` pace the stream using ``clock``/``sleep``; if ``False`` yield immediately.
    audio_block_bytes:
        Maximum size (in bytes) of each emitted :class:`AudioChunk`'s ``pcm`` payload.
    clock:
        Monotonic clock callable returning seconds (injectable for tests).
    sleep:
        Sleep callable taking seconds (injectable for tests).
    """

    def __init__(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        fps: float = 1.0,
        realtime: bool = True,
        audio_block_bytes: int = 4096,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if audio_block_bytes <= 0:
            raise ValueError("audio_block_bytes must be > 0")
        self.video_path = video_path
        self.audio_path = audio_path
        self.fps = float(fps)
        self.realtime = realtime
        self.audio_block_bytes = int(audio_block_bytes)
        self._clock = clock
        self._sleep = sleep
        self._vr = None  # lazily created decord VideoReader

    # -- video ---------------------------------------------------------------

    def _reader(self):
        """Lazily build (and cache) the decord ``VideoReader`` for ``video_path``."""
        if self._vr is None:
            from decord import VideoReader, cpu  # lazy import: keep module import-light

            self._vr = VideoReader(self.video_path, ctx=cpu(0))
        return self._vr

    def _sampled_frame_indices(self) -> List[int]:
        """Indices into the source video to sample at ~``fps`` frames/sec.

        Formula (deterministic)::

            n        = len(vr)                       # source frame count
            src_fps  = vr.get_avg_fps()              # source frames per second
            duration = n / src_fps                   # seconds
            count    = max(1, floor(duration * fps)) # number of frames to emit
            idx[k]   = min(round(k / fps * src_fps), n - 1)   for k in [0, count)

        The k-th emitted frame therefore corresponds to wall-clock time ``ts = k / fps``.

        Worked example: an 8-frame source recorded at 1 fps (``src_fps == 1``, ``duration == 8``)
        sampled at ``fps == 1.0`` gives ``count == 8`` and ``idx == [0, 1, 2, 3, 4, 5, 6, 7]``,
        i.e. one frame per second at ts ``0.0, 1.0, ..., 7.0``.
        """
        vr = self._reader()
        n = len(vr)
        if n == 0:
            return []
        src_fps = float(vr.get_avg_fps())
        if src_fps <= 0:
            src_fps = self.fps
        duration = n / src_fps
        count = max(1, int(duration * self.fps))
        indices: List[int] = []
        for k in range(count):
            src_idx = int(round(k / self.fps * src_fps))
            if src_idx > n - 1:
                src_idx = n - 1
            indices.append(src_idx)
        return indices

    def _iter_video(self) -> Iterator[VideoFrame]:
        vr = self._reader()
        for k, src_idx in enumerate(self._sampled_frame_indices()):
            image = vr[src_idx].asnumpy()  # HxWx3 uint8 RGB
            yield VideoFrame(ts=k / self.fps, image=image)

    # -- audio ---------------------------------------------------------------

    def _iter_audio(self) -> Iterator[AudioChunk]:
        if not self.audio_path:
            return
        with wave.open(self.audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nchannels = wf.getnchannels()
            bytes_per_second = float(sample_rate * sampwidth * nchannels)
            # Align block size to a whole number of (interleaved) samples so we never split a frame.
            frame_bytes = sampwidth * nchannels
            block = max(frame_bytes, (self.audio_block_bytes // frame_bytes) * frame_bytes)
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
        offset = 0
        total = len(data)
        while offset < total:
            chunk = data[offset : offset + block]
            ts = offset / bytes_per_second if bytes_per_second else 0.0
            yield AudioChunk(ts=ts, pcm=chunk, sample_rate=sample_rate)
            offset += len(chunk)

    # -- merge + pacing ------------------------------------------------------

    def stream(self) -> Iterator[AVEvent]:
        """Yield merged video+audio events globally ordered by ``ts``.

        Events are gathered, sorted by ``ts`` (stable sort keeps relative order for equal ts), and
        then either yielded immediately (``realtime=False``) or paced against ``clock``/``sleep``
        (``realtime=True``) so that event ``i`` is yielded no earlier than ``start + ev.ts``.
        """
        events: List[AVEvent] = list(self._iter_video()) + list(self._iter_audio())
        events.sort(key=lambda e: e.ts)

        if not self.realtime:
            for ev in events:
                yield ev
            return

        start = self._clock()
        for ev in events:
            delay = ev.ts - (self._clock() - start)
            self._sleep(max(0.0, delay))
            yield ev

    def close(self) -> None:
        """Release the decord reader."""
        self._vr = None


# Recall.ai message schema (normalized) -------------------------------------
#
# :meth:`RecallAITransport._parse_message` consumes already-JSON-decoded ``dict`` messages with a
# ``kind`` discriminator. The normalized schema this module understands:
#
#   {"kind": "video_separate_png", "ts": <float seconds>, "data": <base64 PNG bytes>,
#    "participant": <optional str>}
#       -> decoded to an RGB np.ndarray and emitted as a VideoFrame.
#
#   {"kind": "audio", "ts": <float seconds>, "data": <base64 PCM s16le bytes>,
#    "sample_rate": <int, default 16000>}
#       -> emitted as an AudioChunk.
#
#   {"kind": "video_separate_h264", ...}
#       -> documented stub: H.264 decoding needs PyAV (not installed); returns None for now.
#
#   any other / missing "kind"
#       -> returns None (ignored by stream()).


class RecallAITransport(MeetingTransport):
    """Live transport backed by the Recall.ai websocket (real path is minimal & mockable).

    The websocket is created via the injectable ``ws_factory(api_key, ws_url) -> ws`` so the socket
    is fully mockable in tests. The returned ``ws`` object must be iterable over decoded ``dict``
    messages and expose a ``close()`` method. The default factory lazily imports ``websockets``.

    Parameters
    ----------
    api_key:
        Recall.ai API key. Must be non-empty.
    bot_id:
        Identifier of the Recall bot joined to the meeting.
    ws_url:
        Websocket URL to connect to.
    ws_factory:
        Optional callable ``(api_key, ws_url) -> ws`` building the websocket. Defaults to a thin
        wrapper around the synchronous ``websockets`` client.
    """

    def __init__(
        self,
        api_key: str,
        bot_id: str,
        ws_url: str,
        ws_factory: Optional[Callable[[str, str], object]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self.api_key = api_key
        self.bot_id = bot_id
        self.ws_url = ws_url
        self._ws_factory = ws_factory or self._default_ws_factory
        self._ws = None

    @staticmethod
    def _default_ws_factory(api_key: str, ws_url: str):
        """Default websocket factory: lazily import ``websockets`` (sync client).

        Kept intentionally minimal — the CPU test suite always injects a fake factory. Messages are
        expected to arrive as JSON text and are decoded to dicts by the caller side; here we return
        the raw connection object whose iteration yields JSON strings, wrapped so iteration yields
        decoded dicts.
        """
        import json

        import websockets  # lazy import: keep module import-light, no socket at import time
        from websockets.sync.client import connect

        conn = connect(ws_url, additional_headers={"Authorization": f"Token {api_key}"})

        class _JsonWS:
            def __init__(self, c):
                self._c = c

            def __iter__(self):
                for raw in self._c:
                    yield json.loads(raw)

            def close(self):
                self._c.close()

        return _JsonWS(conn)

    def _parse_message(self, msg: dict) -> Optional[AVEvent]:
        """Convert one normalized Recall.ai message ``dict`` into an :class:`AVEvent` (or ``None``).

        See the module-level schema docstring. Unknown / unsupported ``kind`` values return
        ``None`` so :meth:`stream` simply skips them.
        """
        import base64

        kind = msg.get("kind")
        if kind == "video_separate_png":
            import io

            from PIL import Image  # lazy import: keep module import-light

            import numpy as np

            raw = base64.b64decode(msg["data"])
            with Image.open(io.BytesIO(raw)) as im:
                image = np.asarray(im.convert("RGB"), dtype=np.uint8)
            return VideoFrame(
                ts=float(msg["ts"]),
                image=image,
                participant=msg.get("participant"),
            )
        if kind == "audio":
            pcm = base64.b64decode(msg["data"])
            return AudioChunk(
                ts=float(msg["ts"]),
                pcm=pcm,
                sample_rate=int(msg.get("sample_rate", 16000)),
            )
        if kind == "video_separate_h264":
            # Stub: decoding H.264 needs PyAV which isn't installed in the CPU environment.
            logger.debug("ignoring video_separate_h264 message (PyAV decode not available)")
            return None
        return None

    def stream(self) -> Iterator[AVEvent]:
        """Open the websocket, parse each message, and yield non-``None`` events until closed."""
        self._ws = self._ws_factory(self.api_key, self.ws_url)
        for msg in self._ws:
            ev = self._parse_message(msg)
            if ev is not None:
                yield ev

    def close(self) -> None:
        """Close the underlying websocket if it is open."""
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
