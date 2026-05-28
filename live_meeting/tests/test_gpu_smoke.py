"""GPU smoke test for the LIVE path (deselected by default via the ``gpu`` marker).

This is the only test that touches the real IXC2.5-OmniLive runtime and the Recall.ai websocket. It
is deselected by ``addopts = -m "not gpu"`` in ``pytest.ini`` and additionally skips when the
required environment variables are unset, so it never runs in CI.

NOTE: no heavy imports at module level — everything is imported lazily inside the test body so plain
collection stays CPU-only and import-light.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.gpu
def test_ixc_live_answer_smoke():
    required = ["IXC_MODEL_ROOT", "RECALL_API_KEY", "RECALL_BOT_ID", "RECALL_WS_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(f"GPU smoke test requires env vars: {', '.join(missing)}")

    # Lazy imports: keep collection import-light (no torch/lmdeploy at module load).
    from live_meeting.ixc_engine import IXCStreamingEngine, LmdeployIXCBackend
    from live_meeting.orchestrator import MeetingOrchestrator
    from live_meeting.transport import RecallAITransport

    engine = IXCStreamingEngine(LmdeployIXCBackend(os.environ["IXC_MODEL_ROOT"]))
    transport = RecallAITransport(
        os.environ["RECALL_API_KEY"],
        os.environ["RECALL_BOT_ID"],
        os.environ["RECALL_WS_URL"],
    )
    orch = MeetingOrchestrator(transport, engine)
    orch.submit_query("What is happening in the meeting right now?")
    transcript = orch.run(max_events=200)

    assert transcript.qa, "expected at least one answered query"
    assert transcript.qa[0].answer.strip(), "expected a non-empty answer"
