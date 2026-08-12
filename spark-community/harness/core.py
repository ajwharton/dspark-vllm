# No-Nerf Gate — Harness core
# Pure, importable, testable logic only. Heavy lifting (CLI/IO) lives in cli.py.
# Stdlib-only; no external deps so this runs in any venv.
from __future__ import annotations

import hashlib
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# Request generation — COLD: unique nonce per prompt defeats prefix caching    #
# --------------------------------------------------------------------------- #

# Concrete word pool for building prefill text. Synthetic, deterministic.
_WORDS = [
    "autumn", "basalt", "cedar", "delta", "ember", "fir", "glacier", "harbor",
    "ironsand", "juniper", "kelp", "lagoon", "mica", "neon", "obsidian",
    "pine", "quartz", "rift", "sierra", "tundra", "underwood", "volta",
    "windsock", "xeriscape", "yam", "zircon", "amber", "bramble", "current",
    "driftwood", "estuary", "falcon", "granite", "heather", "inlet", "jasper",
    "kinetic", "lichen", "meander", "noggin", "orchard", "pasture", "quarry",
    "riparian", "sedge", "tanager", "umbra", "vortex", "weir", "xenolith",
]

UUID_DISALLOW = "-"  # kept for clarity though nonce is now hash-based


def unique_prompts(n: int, approx_tokens: int, seed: int = 0) -> list[str]:
    """Generate `n` prompts, each ~approx_tokens, provably unique per prompt.

    The leading nonce line is a distinct UUID per prompt, so even identical
    body text would produce unique inputs — the prefix cache can never serve
    one prompt's prefill to another. Words rotate by (prompt_index + word_idx)
    so bodies also differ.
    """
    rng = __import__("random").Random(seed)
    prompts = []
    words_needed = max(1, int(approx_tokens * 0.75))  # ~1.3 tokens/word
    for i in range(n):
        # Deterministic nonce from (seed, i): distinct within a run, identical
        # across same-seed runs (reproducibility). NOT uuid4 — that would make
        # two runs of the same bench non-comparable.
        nonce = hashlib.sha1(f"{seed}:{i}".encode()).hexdigest()[:24]
        base = rng.randint(0, len(_WORDS) - 1)
        body = " ".join(
            _WORDS[(base + j) % len(_WORDS)] for j in range(words_needed)
        )
        prompts.append(f"nonce-{nonce}\n{body}\nGive a one-line answer.")
    return prompts


# --------------------------------------------------------------------------- #
# Client — OpenAI-compatible, streamed to measure true cold TTFT              #
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    metric: str
    value: float
    output_tokens: int = 0
    reasoning_present: Optional[bool] = None


class VLLMClient:
    """Minimal OpenAI-compat chat client. Non-stream for decode throughput,
    stream for cold TTFT (time-to-first-token). stdlib urllib only."""

    def __init__(self, base_url: str, model: str, timeout: int = 600,
                 max_tokens: int = 64):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _post(self, payload: dict, stream: bool) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if not stream:
                return json.loads(resp.read().decode())
            # Streamed: capture time-to-first-token, then read the rest.
            start = time.monotonic()
            first_byte_t = None
            acc = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                if first_byte_t is None:
                    first_byte_t = time.monotonic() - start
                acc += chunk
            # Best-effort reassemble final JSON from SSE data lines.
            data = acc.decode(errors="replace")
            usage = _extract_stream_usage(data)
            return {"ttft_s": first_byte_t, "usage": usage}

    def decode_throughput(self, prompt: str) -> Sample:
        """Non-stream call; report output tokens / wall time."""
        t0 = time.monotonic()
        res = self._post(
            {"model": self.model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": self.max_tokens, "stream": False},
            stream=False,
        )
        wall = time.monotonic() - t0
        out = int(res["usage"]["completion_tokens"])
        elapsed_gen = max(wall - res.get("_ttft", 0), 1e-6)
        return Sample("decode_tok_s", out / elapsed_gen, output_tokens=out)

    def cold_ttft(self, prompt: str) -> Sample:
        """Streamed call; report time to first token (seconds)."""
        res = self._post(
            {"model": self.model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}},
            stream=True,
        )
        return Sample("ttft_s", res.get("ttft_s", float("nan")))

    def reasoning_probe(self, turns: int = 3) -> list[Sample]:
        """Multi-turn probe: does reasoning persist across resends (trap 04/20)?

        Sends a reasoning-prompt first turn, then conversations each with the
        FULL prior history re-sent. Flags whether a reasoning/thinking field
        appears in any assistant message of any turn.
        """
        history: list[dict] = []
        saw_reasoning: Optional[bool] = None
        samples = []
        for i in range(turns):
            if i == 0:
                history.append({"role": "user", "content":
                                "Let's reason step by step: what is 973*47? "
                                "Show your full reasoning then the answer."})
            else:
                history.append({"role": "user", "content": "Continue, restating your reasoning."})
            res = self._post(
                {"model": self.model, "messages": history, "max_tokens": 64,
                 "stream": False},
                stream=False,
            )
            msg = res["choices"][0]["message"]
            # vLLM reasoning models expose reasoning in a dedicated field.
            reason_field = msg.get("reasoning_content") or msg.get("reasoning") or ""
            present = bool(reason_field and len(reason_field.strip()) > 0)
            if saw_reasoning is None and present:
                saw_reasoning = True
            history.append({"role": "assistant", "content": msg.get("content", "")})
            samples.append(Sample("reasoning_present", 1.0 if present else 0.0,
                                  reasoning_present=present))
        return samples


def _extract_stream_usage(sse_text: str) -> dict:
    """Grab usage from the last SSE data line if present."""
    try:
        for line in reversed(sse_text.splitlines()):
            if line.startswith("data:"):
                obj = json.loads(line[5:].strip())
                if obj.get("usage"):
                    return obj["usage"]
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
# Gate logic — pure, testable                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    metric: str
    baseline_median: float
    candidate_median: float
    delta_pct: float
    threshold_pct: float
    lower_is_better: bool
    failed: bool = False

    @property
    def summary(self) -> str:
        verb = "lower" if self.lower_is_better else "higher"
        head = "FAIL" if self.failed else "pass"
        return (f"[{head}] {self.metric}: cand {self.candidate_median:.4f} vs "
                f"base {self.baseline_median:.4f} ({self.delta_pct:+.2f}%), "
                f"guard {verb} by <= {self.threshold_pct}%")


def median(xs: list[float]) -> float:
    return statistics.median(xs)


def compare_median(baseline: list[float], candidate: list[float],
                   metric: str, threshold_pct: float,
                   lower_is_better: bool) -> GateResult:
    """Classic two-sample median comparison with a regression guard.

    lower_is_better=True, threshold T: candidate FAILS when its median exceeds
    baseline's by more than T% (e.g. TTFT got 12% worse → FAIL at 10%).
    """
    b = statistics.median(baseline) if baseline else float("nan")
    c = statistics.median(candidate) if candidate else float("nan")
    delta = (c - b) / b * 100.0 if b else float("nan")
    if lower_is_better:
        failed = c > b * (1 + threshold_pct / 100.0)
    else:  # higher is better (throughput): FAIL when it drops >T%
        failed = c < b * (1 - threshold_pct / 100.0)
    return GateResult(metric, b, c, delta, threshold_pct, lower_is_better, failed)


def gate_blocked(results: list[GateResult]) -> list[GateResult]:
    return [r for r in results if r.failed]
