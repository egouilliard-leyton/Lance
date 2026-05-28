"""Tests for :func:`live_meeting.cli.main` across the simulate / eval / run subcommands.

CPU-only, no network: ``simulate`` and ``eval`` run a :class:`FakeEngine` over synth fixtures; the
live ``run`` path is only exercised for its credential guard (it never touches a GPU/socket).
"""

from __future__ import annotations

import json
import os

from live_meeting.cli import main


def test_simulate_returns_zero_and_prints_json(synth_video, synth_audio, capsys):
    vpath, _ = synth_video
    apath, _ = synth_audio
    rc = main(
        [
            "simulate",
            "--video_path",
            vpath,
            "--audio_path",
            apath,
            "--fps",
            "1.0",
            "--no-realtime",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert "qa" in payload and "proactive" in payload
    assert isinstance(payload["qa"], list)
    assert isinstance(payload["proactive"], list)


def test_eval_writes_report_json(tmp_path, synth_video, synth_audio, capsys):
    vpath, _ = synth_video
    apath, _ = synth_audio
    spec = {
        "video_path": vpath,
        "audio_path": apath,
        "qa": [
            {"at_ts": 1.0, "query": "what is happening", "expected": "presentation"}
        ],
        "proactive": [],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out_dir = tmp_path / "out"

    rc = main(
        [
            "eval",
            "--eval_spec_path",
            str(spec_path),
            "--output_path",
            str(out_dir),
            "--fps",
            "1.0",
        ]
    )
    assert rc == 0

    report_path = out_dir / "eval_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    for key in (
        "qa_total",
        "qa_accuracy",
        "precision",
        "recall",
        "f1",
        "latency_p50",
        "latency_p95",
        "latency_mean",
    ):
        assert key in report

    # render() output (with labels) is printed to stdout.
    out = capsys.readouterr().out
    assert "qa_accuracy" in out


def test_run_ixc_without_credentials_returns_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("RECALL_API_KEY", raising=False)
    rc = main(["run", "--engine", "ixc", "--recall_api_key", ""])
    assert rc != 0
    err = capsys.readouterr().err
    assert "RECALL_API_KEY" in err or "credentials" in err.lower()
