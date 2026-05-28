"""Offline evaluation harness for the live_meeting pipeline.

Given a :class:`GroundTruthSpec` (recorded inputs + expected Q/A and expected proactive events), the
:class:`EvalHarness` builds a transport, runs a :class:`~live_meeting.orchestrator.MeetingOrchestrator`
against an engine, and scores the produced :class:`~live_meeting.events.Transcript`:

* Q/A accuracy — the i-th produced answer is matched to the i-th expected answer using its ``match``
  mode (``contains`` | ``exact`` | ``regex``).
* Proactive precision/recall/F1 — greedy matching of expected events to produced events of the same
  ``kind`` within ``tolerance_s`` seconds.
* Latency p50/p95/mean — over the per-query end-to-end latencies recorded by the orchestrator.

This module is CPU-only and import-light: only stdlib (``json``, ``re``, ``statistics``) plus the
in-package contracts are imported.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ._logging import get_logger
from .engine import UnderstandingEngine
from .events import ProactiveEvent, Transcript
from .orchestrator import MeetingOrchestrator
from .transport import MeetingTransport

logger = get_logger("live_meeting.eval")

_MATCH_MODES = ("contains", "exact", "regex")


@dataclass
class ExpectedQA:
    """An expected answer to ``query`` asked around ``at_ts`` seconds, scored via ``match`` mode."""

    at_ts: float
    query: str
    expected: str
    match: str = "contains"


@dataclass
class ExpectedProactive:
    """An expected proactive event of ``kind`` around ``at_ts`` within ``tolerance_s`` seconds."""

    at_ts: float
    kind: str
    tolerance_s: float = 5.0


@dataclass
class GroundTruthSpec:
    """Recorded inputs plus the expected Q/A and proactive events to score against."""

    video_path: str
    audio_path: str
    qa: List[ExpectedQA] = field(default_factory=list)
    proactive: List[ExpectedProactive] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "GroundTruthSpec":
        """Load a spec from a JSON file with nested ``qa`` / ``proactive`` lists."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            video_path=d["video_path"],
            audio_path=d.get("audio_path", ""),
            qa=[ExpectedQA(**q) for q in d.get("qa", [])],
            proactive=[ExpectedProactive(**p) for p in d.get("proactive", [])],
        )


@dataclass
class EvalReport:
    """Scored metrics for a single evaluation run."""

    qa_total: int
    qa_correct: int
    qa_accuracy: float
    proactive_tp: int
    proactive_fp: int
    proactive_fn: int
    precision: float
    recall: float
    f1: float
    latency_p50: float
    latency_p95: float
    latency_mean: float

    def to_dict(self) -> dict:
        return {
            "qa_total": self.qa_total,
            "qa_correct": self.qa_correct,
            "qa_accuracy": self.qa_accuracy,
            "proactive_tp": self.proactive_tp,
            "proactive_fp": self.proactive_fp,
            "proactive_fn": self.proactive_fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_mean": self.latency_mean,
        }

    def render(self) -> str:
        """Render a human-readable, multi-line report containing every metric label."""
        return "\n".join(
            [
                "=== live_meeting eval report ===",
                f"qa_total:     {self.qa_total}",
                f"qa_correct:   {self.qa_correct}",
                f"qa_accuracy:  {self.qa_accuracy:.4f}",
                f"proactive_tp: {self.proactive_tp}",
                f"proactive_fp: {self.proactive_fp}",
                f"proactive_fn: {self.proactive_fn}",
                f"precision:    {self.precision:.4f}",
                f"recall:       {self.recall:.4f}",
                f"f1:           {self.f1:.4f}",
                f"latency_p50:  {self.latency_p50:.4f}",
                f"latency_p95:  {self.latency_p95:.4f}",
                f"latency_mean: {self.latency_mean:.4f}",
            ]
        )


def _qa_matches(produced: str, expected: ExpectedQA) -> bool:
    mode = expected.match
    if mode == "contains":
        return expected.expected.lower() in produced.lower()
    if mode == "exact":
        return produced.strip() == expected.expected.strip()
    if mode == "regex":
        return re.search(expected.expected, produced) is not None
    raise ValueError(f"unknown match mode: {mode!r} (expected one of {_MATCH_MODES})")


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile over ``values`` (``pct`` in [0, 100]); empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _score_proactive(
    expected: List[ExpectedProactive], produced: List[ProactiveEvent]
) -> tuple:
    """Greedy match expected -> produced; return (tp, fp, fn)."""
    matched = [False] * len(produced)
    tp = 0
    for exp in expected:
        for i, prod in enumerate(produced):
            if matched[i]:
                continue
            if prod.kind == exp.kind and abs(prod.ts - exp.at_ts) <= exp.tolerance_s:
                matched[i] = True
                tp += 1
                break
    fn = len(expected) - tp
    fp = matched.count(False)
    return tp, fp, fn


class EvalHarness:
    """Run the pipeline over a :class:`GroundTruthSpec` and score the result into an :class:`EvalReport`.

    Parameters
    ----------
    transport_factory:
        Callable ``(spec) -> MeetingTransport`` building a fresh transport for the spec's inputs.
    engine:
        The :class:`UnderstandingEngine` under evaluation.
    """

    def __init__(
        self,
        transport_factory: Callable[[GroundTruthSpec], MeetingTransport],
        engine: UnderstandingEngine,
    ) -> None:
        self.transport_factory = transport_factory
        self.engine = engine

    def run(self, spec: GroundTruthSpec) -> EvalReport:
        transport = self.transport_factory(spec)
        orch = MeetingOrchestrator(transport, self.engine)
        for expected in spec.qa:
            orch.submit_query(expected.query)
        transcript: Transcript = orch.run()

        # -- Q/A accuracy ----------------------------------------------------
        qa_total = len(spec.qa)
        qa_correct = 0
        for expected, produced in zip(spec.qa, transcript.qa):
            if _qa_matches(produced.answer, expected):
                qa_correct += 1
        qa_accuracy = (qa_correct / qa_total) if qa_total else 0.0

        # -- Proactive P/R/F1 ------------------------------------------------
        tp, fp, fn = _score_proactive(spec.proactive, transcript.proactive)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # -- Latency ---------------------------------------------------------
        latencies = [q.latency_s for q in transcript.qa]
        latency_p50 = _percentile(latencies, 50.0)
        latency_p95 = _percentile(latencies, 95.0)
        latency_mean = statistics.fmean(latencies) if latencies else 0.0

        return EvalReport(
            qa_total=qa_total,
            qa_correct=qa_correct,
            qa_accuracy=qa_accuracy,
            proactive_tp=tp,
            proactive_fp=fp,
            proactive_fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_mean=latency_mean,
        )
