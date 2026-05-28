"""Configuration for the live_meeting PoC.

Uses a ``@dataclass`` + stdlib ``argparse`` (NOT HuggingFace ``HfArgumentParser``) so the
package stays import-light and CPU-only: ``HfArgumentParser`` lives in ``transformers``, a heavy
dependency that the bare CPU/CI environment does not have. The dataclass remains the single
config object passed around the package.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

ENGINES = ("fake", "ixc")
MODES = ("run", "eval", "simulate")


@dataclass
class LiveMeetingConfig:
    mode: str = "simulate"
    engine: str = "fake"
    # IXC engine
    ixc_model_root: str = ""
    tp: int = 1
    # streaming
    fps: float = 1.0
    audio_sample_rate: int = 16000
    realtime: bool = True
    # Recall.ai transport
    recall_api_key: str = field(default_factory=lambda: os.getenv("RECALL_API_KEY", ""))
    recall_bot_id: str = ""
    recall_ws_url: str = ""
    # recorded/eval inputs
    video_path: str = ""
    audio_path: str = ""
    eval_spec_path: str = ""
    output_path: str = "results/live_meeting"


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all LiveMeetingConfig fields as flags on ``parser``."""
    parser.add_argument("--mode", default="simulate", choices=list(MODES))
    parser.add_argument("--engine", default="fake", choices=list(ENGINES))
    parser.add_argument("--ixc_model_root", default="")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--recall_api_key", default=os.getenv("RECALL_API_KEY", ""))
    parser.add_argument("--recall_bot_id", default="")
    parser.add_argument("--recall_ws_url", default="")
    parser.add_argument("--video_path", default="")
    parser.add_argument("--audio_path", default="")
    parser.add_argument("--eval_spec_path", default="")
    parser.add_argument("--output_path", default="results/live_meeting")
    return parser


def config_from_namespace(ns: argparse.Namespace) -> LiveMeetingConfig:
    kwargs = {f.name: getattr(ns, f.name) for f in fields(LiveMeetingConfig) if hasattr(ns, f.name)}
    return LiveMeetingConfig(**kwargs)


def parse_config(argv: Optional[List[str]] = None) -> LiveMeetingConfig:
    parser = add_config_args(argparse.ArgumentParser(prog="live_meeting"))
    ns = parser.parse_args(argv)
    return config_from_namespace(ns)
