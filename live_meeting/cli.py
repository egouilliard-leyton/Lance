"""Command-line entry point for the live_meeting PoC.

Three subcommands, each reusing :func:`live_meeting.config.add_config_args` for its flags:

* ``simulate`` — replay a recorded file through a :class:`~live_meeting.engine.FakeEngine` and print
  the resulting transcript as JSON. CPU-only, no network.
* ``eval`` — score the pipeline against a :class:`~live_meeting.eval.GroundTruthSpec`, writing
  ``eval_report.json`` to the output dir and printing a human-readable report.
* ``run`` — the LIVE path: wire the IXC streaming engine to the Recall.ai websocket. Guarded so it
  fails fast (nonzero exit) when the IXC engine is requested without Recall credentials.

This module is import-light: the heavy engines/transports are imported lazily inside each dispatch
branch so ``import live_meeting.cli`` (and ``simulate``/``eval``) never touch torch/lmdeploy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from ._logging import get_logger
from .config import add_config_args, config_from_namespace

logger = get_logger("live_meeting.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="live_meeting")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "eval", "simulate"):
        sub = subparsers.add_parser(command)
        add_config_args(sub)
    return parser


def _cmd_simulate(cfg) -> int:
    from .engine import FakeEngine
    from .orchestrator import MeetingOrchestrator
    from .transport import RecordedFileTransport

    transport = RecordedFileTransport(
        cfg.video_path,
        cfg.audio_path or None,
        fps=cfg.fps,
        realtime=cfg.realtime,
    )
    engine = FakeEngine()
    orch = MeetingOrchestrator(transport, engine)
    transcript = orch.run()
    print(json.dumps(transcript.to_dict()))
    return 0


def _cmd_eval(cfg) -> int:
    from .engine import FakeEngine
    from .eval import EvalHarness, GroundTruthSpec
    from .transport import RecordedFileTransport

    spec = GroundTruthSpec.from_json(cfg.eval_spec_path)
    harness = EvalHarness(
        lambda s: RecordedFileTransport(
            s.video_path, s.audio_path or None, fps=cfg.fps, realtime=False
        ),
        FakeEngine(),
    )
    report = harness.run(spec)

    os.makedirs(cfg.output_path, exist_ok=True)
    out_path = os.path.join(cfg.output_path, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report.to_dict()))
    print(report.render())
    return 0


def _cmd_run(cfg) -> int:
    if cfg.engine == "ixc" and not cfg.recall_api_key:
        print(
            "error: `run --engine ixc` requires Recall.ai credentials; set RECALL_API_KEY "
            "(or pass --recall_api_key). This live path needs a GPU box + IXC weights too.",
            file=sys.stderr,
        )
        return 2

    from .ixc_engine import IXCStreamingEngine, LmdeployIXCBackend
    from .orchestrator import MeetingOrchestrator
    from .transport import RecallAITransport

    engine = IXCStreamingEngine(
        LmdeployIXCBackend(cfg.ixc_model_root, tp=cfg.tp)
    )
    transport = RecallAITransport(
        cfg.recall_api_key, cfg.recall_bot_id, cfg.recall_ws_url
    )
    orch = MeetingOrchestrator(transport, engine)
    transcript = orch.run()
    print(json.dumps(transcript.to_dict()))
    return 0


_DISPATCH = {
    "simulate": _cmd_simulate,
    "eval": _cmd_eval,
    "run": _cmd_run,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    cfg = config_from_namespace(ns)
    # Force the mode to match the chosen subcommand (config.py is not modified).
    cfg.mode = ns.command
    return _DISPATCH[ns.command](cfg)
