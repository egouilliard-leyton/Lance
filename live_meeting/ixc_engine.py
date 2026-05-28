"""IXC2.5-OmniLive streaming engine, with all heavy deps isolated behind a thin boundary.

This module MUST be import-light: importing it never pulls in torch, lmdeploy, IXC, funasr or
opens any process/queue. The single seam that touches those runtimes is the
:class:`IXCBackendBoundary` protocol; the real implementation (:class:`LmdeployIXCBackend`)
performs ALL heavy imports lazily, inside the methods that need them.

:class:`IXCStreamingEngine` is the CPU-only orchestration layer: it implements the
:class:`~live_meeting.engine.UnderstandingEngine` interface in terms of an
``IXCBackendBoundary``, adding frame-rate throttling. It is fully testable against a
``MagicMock(spec=IXCBackendBoundary)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from .engine import UnderstandingEngine
from .events import AudioChunk, ProactiveEvent, VideoFrame

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@runtime_checkable
class IXCBackendBoundary(Protocol):
    """The ONLY seam that touches the IXC/lmdeploy/multiprocessing runtime.

    Implementations own the model process(es), the frame/audio queues, and the query
    plumbing. Everything above this protocol is CPU-only and unit-testable with a mock.
    """

    def start(self, session_id: str) -> None:
        """Spin up the backend (model process, ASR/VAD threads, queues) for a session."""
        ...

    def put_frame(self, image: "np.ndarray", ts: float) -> None:
        """Enqueue a decoded RGB frame (HxWx3 uint8) at timestamp ``ts``."""
        ...

    def put_audio(self, pcm: bytes, ts: float, sample_rate: int) -> None:
        """Enqueue a PCM audio chunk (s16le mono) at timestamp ``ts``."""
        ...

    def set_query(self, text: str) -> list:
        """Register a query; block until the backend has selected frames. Return those images."""
        ...

    def submit_mm_query(self, text: str, imgs: list) -> str:
        """Run the multimodal query over ``text`` + ``imgs`` and return the answer text."""
        ...

    def poll_event(self) -> Optional[ProactiveEvent]:
        """Return the next proactive event if one is ready, else ``None``."""
        ...

    def stop(self) -> None:
        """Tear down the backend and release all resources."""
        ...


class LmdeployIXCBackend:
    """Real ``IXCBackendBoundary`` over IXC2.5-OmniLive's lmdeploy/multiprocessing runtime.

    ``__init__`` only stores config; it imports NOTHING heavy. Every torch/lmdeploy/IXC/funasr
    import is lazy, inside the method that uses it, so ``import live_meeting.ixc_engine`` stays
    cheap and GPU-free. Methods map onto IXC's reference runtime in
    ``online_demo/Backend/backend_ixc/client.py``:

      * ``start``           -> launch the IXC client/model process and shared ``vs_dict`` state,
                               the ``frame_list`` queue, ``mm_querys`` queue, ``llm_out_queue``,
                               and the VAD/ASR worker thread.
      * ``put_frame``       -> append ``(image, ts)`` to the backend's ``frame_list`` queue.
      * ``put_audio``       -> feed raw PCM (16kHz s16le mono) into the VAD/ASR thread input.
      * ``set_query``       -> set ``vs_dict['query'] = text``, wait on ``vs_dict['query_finish']``,
                               then return the backend-selected ``imgs``.
      * ``submit_mm_query`` -> ``mm_querys.put((text, imgs, time_dict))`` then block reading the
                               answer off ``llm_out_queue``.
      * ``poll_event``      -> non-blocking read of any proactive observation the model emitted.
      * ``stop``            -> signal shutdown and join the worker process/threads.

    Out of scope for CPU tests: this class is exercised only by the GPU smoke test. When the
    IXC package/weights are unavailable the lazy imports raise an informative error — but ONLY
    when a method is actually called, never at module import time.
    """

    def __init__(
        self,
        model_root: str,
        tp: int = 1,
        asr_model: str = "streaming_audio",
        tts_model: Optional[str] = None,
    ) -> None:
        # Config only — no heavy imports here, by design.
        self.model_root = model_root
        self.tp = tp
        self.asr_model = asr_model
        self.tts_model = tts_model
        self._started = False

    @staticmethod
    def _require_ixc():
        """Lazily import the IXC/lmdeploy runtime, raising an informative error if missing.

        Kept as a single chokepoint so every method imports through the same lazy path. The
        real import would be e.g. ``from online_demo.Backend.backend_ixc import client``.
        """
        try:  # pragma: no cover - exercised only on a real GPU box with weights present
            import torch  # noqa: F401
            import lmdeploy  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LmdeployIXCBackend requires the IXC2.5-OmniLive runtime (torch + lmdeploy + "
                "the IXC package and weights). Install them and provide a valid model_root; "
                "this backend is only meant to run on a GPU box."
            ) from exc

    def start(self, session_id: str) -> None:  # pragma: no cover - GPU smoke test only
        self._require_ixc()
        # Launch IXC client process, build frame_list / mm_querys / llm_out_queue, start VAD/ASR.
        raise NotImplementedError("LmdeployIXCBackend.start is GPU-only and not yet wired up.")

    def put_frame(self, image: "np.ndarray", ts: float) -> None:  # pragma: no cover
        # frame_list.put((image, ts))
        raise NotImplementedError("LmdeployIXCBackend.put_frame is GPU-only and not yet wired up.")

    def put_audio(self, pcm: bytes, ts: float, sample_rate: int) -> None:  # pragma: no cover
        # Feed PCM (16kHz s16le mono) into the VAD/ASR thread input.
        raise NotImplementedError("LmdeployIXCBackend.put_audio is GPU-only and not yet wired up.")

    def set_query(self, text: str) -> list:  # pragma: no cover
        # vs_dict['query'] = text; wait vs_dict['query_finish']; return selected imgs.
        raise NotImplementedError("LmdeployIXCBackend.set_query is GPU-only and not yet wired up.")

    def submit_mm_query(self, text: str, imgs: list) -> str:  # pragma: no cover
        # mm_querys.put((text, imgs, time_dict)); return llm_out_queue.get().
        raise NotImplementedError("LmdeployIXCBackend.submit_mm_query is GPU-only and not yet wired up.")

    def poll_event(self) -> Optional[ProactiveEvent]:  # pragma: no cover
        # Non-blocking read of any proactive observation emitted by the model.
        raise NotImplementedError("LmdeployIXCBackend.poll_event is GPU-only and not yet wired up.")

    def stop(self) -> None:  # pragma: no cover
        # Signal shutdown and join worker process/threads.
        raise NotImplementedError("LmdeployIXCBackend.stop is GPU-only and not yet wired up.")


class IXCStreamingEngine(UnderstandingEngine):
    """CPU-only ``UnderstandingEngine`` adapter over an :class:`IXCBackendBoundary`.

    Adds frame-rate throttling on top of the backend: at ``fps`` frames/sec, at most one frame
    is forwarded per ``1/fps`` window. Audio is forwarded unchanged. Everything else delegates
    straight to the backend.
    """

    def __init__(
        self,
        backend: "IXCBackendBoundary",
        fps: float = 1.0,
        audio_sample_rate: int = 16000,
    ) -> None:
        self.backend = backend
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self._min_interval = 1.0 / fps
        self._last_pushed_ts: Optional[float] = None

    def start_session(self, session_id: str) -> None:
        self._last_pushed_ts = None
        self.backend.start(session_id)

    def push_video(self, frame: VideoFrame) -> None:
        if self._last_pushed_ts is None or frame.ts - self._last_pushed_ts >= self._min_interval - 1e-6:
            self.backend.put_frame(frame.image, frame.ts)
            self._last_pushed_ts = frame.ts

    def push_audio(self, chunk: AudioChunk) -> None:
        # Forward PCM unchanged; the backend owns resampling/VAD/ASR.
        self.backend.put_audio(chunk.pcm, chunk.ts, chunk.sample_rate)

    def ask(self, query: str) -> str:
        imgs = self.backend.set_query(query)
        return self.backend.submit_mm_query(query, imgs)

    def poll_proactive(self) -> Optional[ProactiveEvent]:
        return self.backend.poll_event()

    def end_session(self) -> None:
        self.backend.stop()
