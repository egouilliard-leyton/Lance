"""live_meeting — PoC 1: live meeting understanding (IXC2.5-OmniLive + Recall.ai).

This package wraps an existing streaming video+audio model (IXC2.5-OmniLive) behind a
clean ``UnderstandingEngine`` interface and a live ``MeetingTransport`` (Recall.ai), plus an
evaluation harness. Heavy/external dependencies (torch/lmdeploy, the Recall websocket) are
isolated behind thin, mockable boundaries so the test suite runs CPU-only.

Only lightweight contracts are exported at package import time; importing this package never
pulls in torch, lmdeploy or opens a network connection.
"""

from .config import LiveMeetingConfig, parse_config
from .events import (
    AudioChunk,
    AVEvent,
    ProactiveEvent,
    QAResult,
    Transcript,
    VideoFrame,
)

__all__ = [
    "VideoFrame",
    "AudioChunk",
    "ProactiveEvent",
    "QAResult",
    "Transcript",
    "AVEvent",
    "LiveMeetingConfig",
    "parse_config",
]
