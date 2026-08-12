# No-Nerf Gate — STANDARDIZED measurement protocol
# Canonical, confound-controlled bench against a live OpenAI-compat engine.
#
# Why this exists / design notes (2026-08-12):
#   Earlier DS4F numbers (11.5 vs 7.9 tok/s) were NOT comparable — they
#   differed by uncontrolled confounds (thinking state, output length, and
#   whether prefill was included). This module fixes those so gate results
#   mean something and are REPEATABLE for A/B.
#
#   Engine reality this is built for: the 0731 runai streamer SSE-batches
#   aggressively, so per-token timing (t_last - t_first) collapses to ~0 and
#   gives absurd tok/s. THEREFORE the decode metric here is the stable,
#   engine-agnostic one:
#       throughput_tok_s = completion_tokens / wall   (E2E, includes prefill;
#                                                       a LOWER BOUND)
#   with cold TTFT reported SEPARATELY (same-response first-token time), and
#   an explicit early_eos flag when the completion is too short to be a valid
#   decode sample. For the no-nerf gate what matters is a STABLE, identical
#   methodology across A/B, not a microsecond-pure decode number.
#
# Protocol invariants:
#   - Streamed; throughput = comp/wall (lower bound); ttft separate.
#   - Thinking state explicit per run (never mixed).
#   - Fixed prompt length (unique nonce defeats prefix cache) and FIXED output
#     length; prompts FORCE a long continuation (else early EOS dilutes).
#   - N rounds; report MEDIAN + min. Reasoning presence measured, not assumed.
# Stdlib only. Reuses core.unique_prompts.
from __future__ import annotations

import json
import statistics
import time
import urllib.request
from dataclasses import dataclass

from core import unique_prompts  # noqa: F401  (re-exported for CLI/logic use)

# Threshold: a decode sample is invalid if the model stopped before this many
# output tokens (early EOS dilutes throughput and makes A/B non-comparable).
MIN_VALID_COMPLETION = 80


# Continuation-forcing prompt suffix — defeats "one-line answer" early EOS.
FORCE_LONG = ("\\n\\nDo not wrap up or conclude. Continue writing at length, "
              "adding many more detailed paragraphs, until you are forced to stop.")

PRELUDE = ("Write a detailed, long-form essay. \\n\\nTopic: ")


@dataclass
class RunMetrics:
    ttft_s: float            # time to first content token (this response)
    throughput_tok_s: float  # completion/wall, E2E LOWER BOUND
    decode_ok: bool          # completion >= MIN_VALID_COMPLETION
    completion_tokens: int
    reasoning_present: bool
    wall_s: float            # full e2e wall time


def stream_generate(base_url: str, model: str, prompt: str, max_tokens: int,
                    api_key: str = "", enable_thinking: bool | None = None,
                    timeout: int = 900) -> RunMetrics:
    """One streamed generation; returns stable throughput + ttft + flags.

    The streamed prompt is appended with FORCE_LONG to hold output open.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + FORCE_LONG}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST", headers=headers)

    start = time.monotonic()
    t_first = None
    completion = 0
    reasoning = False
    buf = b""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            now = time.monotonic() - start
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning = True
                if delta.get("content") and t_first is None:
                    t_first = now
                u = obj.get("usage")
                if u and u.get("completion_tokens"):
                    completion = int(u["completion_tokens"])
    wall = time.monotonic() - start
    if completion < 1:
        raise RuntimeError(f"no completion tokens (max_tokens={max_tokens})")
    throughput = completion / max(wall, 1e-6)
    return RunMetrics(ttft_s=t_first if t_first is not None else wall,
                      throughput_tok_s=throughput,
                      decode_ok=completion >= MIN_VALID_COMPLETION,
                      completion_tokens=completion,
                      reasoning_present=reasoning, wall_s=wall)


def median_run(results: list) -> dict:
    """Robust aggregate over N runs (median, not mean — gates must resist a
    single slow outlier). Flags if any run was too short to be valid."""
    return {
        "n": len(results),
        "ttft_median_s": statistics.median(r.ttft_s for r in results),
        "ttft_min_s": min(r.ttft_s for r in results),
        "throughput_median_tok_s": statistics.median(
            r.throughput_tok_s for r in results),
        "throughput_min_tok_s": min(r.throughput_tok_s for r in results),
        "all_decode_ok": all(r.decode_ok for r in results),
        "completion_median": statistics.median(
            r.completion_tokens for r in results),
        "reasoning_present_any": any(r.reasoning_present for r in results),
        "wall_median_s": statistics.median(r.wall_s for r in results),
    }
