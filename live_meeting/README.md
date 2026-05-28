# live_meeting — PoC 1: live meeting understanding

An AI that watches a **live meeting** (video + audio) continuously and (a) answers questions
about what's happening and (b) proactively surfaces notable events.

This is **PoC 1** of a two-PoC plan: it proves the product fast by **wrapping an existing model**
— [InternLM-XComposer2.5-OmniLive (IXC2.5-OmniLive)](https://github.com/InternLM/InternLM-XComposer)
— behind clean interfaces, with [Recall.ai](https://recall.ai) (a managed bot that joins
Zoom/Meet/Teams and streams raw frames + PCM audio) as the live transport. PoC 2 (a custom omni
streaming model) reuses these same interfaces, the transport, and the eval harness.

## Design

The heavy model (torch/lmdeploy) and the Recall.ai websocket are isolated behind thin, mockable
boundaries, so the **entire test suite runs CPU-only** with no GPU and no credentials. The real
end-to-end run happens on a GPU box (see "Live run" below).

```
Recall.ai bot ─► frames + PCM audio ─► UnderstandingEngine ─► answers / proactive events
   (transport.py)                          (engine.py / ixc_engine.py)
                         orchestrator.py wires transport ─► engine ─► Transcript
                         eval.py scores a recorded meeting against a ground-truth spec
```

| Concern | Interface | CPU fake (tested here) | Real adapter (GPU box) |
|---|---|---|---|
| Live A/V source | `MeetingTransport` | `RecordedFileTransport` (decord-decoded file → paced/deterministic stream) | `RecallAITransport` |
| Understanding | `UnderstandingEngine` | `FakeEngine` | `IXCStreamingEngine` + `LmdeployIXCBackend` |

All IXC contact is behind `IXCBackendBoundary`; all network is behind an injectable `ws_factory`.
Importing any `live_meeting` module pulls in **no** torch/lmdeploy and opens no socket.

> **Note on conventions:** this package uses a small stdlib logger (`live_meeting/_logging.py`)
> and `argparse`+dataclasses for config instead of the repo's `common.utils.logging.get_logger`
> / `HfArgumentParser`, because those transitively import `torch`/`transformers` and would break
> the CPU-only isolation guarantee.

## CPU-testable workflow (no GPU, no credentials)

```bash
# from the repo root
pip install -r live_meeting/requirements-dev.txt        # minimal, self-contained (no torch)
python -m pytest live_meeting/tests -v                  # gpu tests auto-deselected

# Simulate a meeting from a recorded file using the FakeEngine (prints a JSON transcript)
python -m live_meeting simulate \
    --video_path path/to/meeting.mp4 --audio_path path/to/meeting.wav \
    --engine fake --no-realtime

# Run the evaluation harness against a ground-truth spec (writes results/live_meeting/eval_report.json)
python -m live_meeting eval \
    --eval_spec_path path/to/spec.json --engine fake \
    --output_path results/live_meeting
```

### Ground-truth spec format (`eval`)

```json
{
  "video_path": "meeting.mp4",
  "audio_path": "meeting.wav",
  "qa": [
    {"at_ts": 5.0, "query": "what is on the shared screen?", "expected": "slide deck", "match": "contains"}
  ],
  "proactive": [
    {"at_ts": 12.0, "kind": "slide_change", "tolerance_s": 3.0}
  ]
}
```

`match` ∈ `contains` | `exact` | `regex`. The report includes QA accuracy, proactive
precision/recall/F1, and answer latency (p50/p95/mean).

## Live run (GPU box, real meeting)

Requires a ~40 GB-class GPU, the IXC2.5-OmniLive weights, and a Recall.ai API key.

```bash
# 1. Deploy the IXC2.5-OmniLive backend (see its repo: online_demo/Backend/backend_ixc)
huggingface-cli download internlm/internlm-xcomposer2d5-ol-7b
python examples/merge_lora.py            # in the IXC repo

# 2. Point this package at it and at Recall.ai
export IXC_MODEL_ROOT=/path/to/internlm-xcomposer2d5-ol-7b
export RECALL_API_KEY=...                # never commit this

# 3. (optional) GPU smoke test
python -m pytest live_meeting/tests/test_gpu_smoke.py -m gpu

# 4. Join a live meeting (Recall.ai bot must already be in the call)
python -m live_meeting run \
    --engine ixc --ixc_model_root "$IXC_MODEL_ROOT" \
    --recall_bot_id <bot-id> --recall_ws_url wss://<recall-stream-url>
```

## Known limitation → PoC 2

IXC2.5-OmniLive's streaming memory is **video-only** (audio rides separately as ASR text) and its
clip memory is **unbounded** — it grows for the length of the meeting, degrading latency on long
sessions. PoC 2 addresses exactly this: an **omni, bounded** memory module mounted on a Qwen-Omni
backbone. The interfaces here are intentionally engine-agnostic so PoC 2 drops in as another
`UnderstandingEngine`.
