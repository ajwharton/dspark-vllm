# No-Nerf Gate — capability detectors (Tony / Keys / Mia specific probes)
#
# Purpose (Andrew, 2026-08-12): the ecosystem DS4F/GB10 forks each patched a
# SPECIFIC capability into their build. To decide whether stock vLLM 0.27 "works
# as is" on dual-Spark GB10s for DS4F, we must DETECT each capability on a stock
# endpoint, not infer it statically. Each probe returns a verdict:
#   PASS (capability present/healthy on stock) | FAIL (absent/broken -> patch
#   needed) | UNKNOWN (not observable on this endpoint, reported honestly).
#
# Probes:
#   probe_spec_acceptance   (Tony #4: draft shared-expert drop -> accept 60->25%)
#   probe_concurrency       (Keys #3: request-stable main-KV / no cudagraph hang)
#   probe_warm_ttft         (Mia: prefix-cache warm TTFT path)
#
# All engine-agnostic (OpenAI-compat + /metrics where available). Stdlib only.
from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass

from core import unique_prompts


def _post_json(url: str, payload: dict, api_key: str = "",
               timeout: int = 900) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_text(url: str, api_key: str = "", timeout: int = 60) -> str:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


@dataclass
class ProbeVerdict:
    probe: str
    verdict: str          # PASS | FAIL | UNKNOWN
    detail: str
    value: float | None = None
    threshold: float | str | None = None


def _base_url(base: str) -> str:
    return base.rstrip("/")


# --------------------------------------------------------------------------- #
# Tony #4 — DSpark speculative acceptance health (60% -> 25% silent collapse)
# --------------------------------------------------------------------------- #
def probe_spec_acceptance(base: str, api_key: str = "",
                          hard_fail_below: float = 0.30,
                          pass_at_or_above: float = 0.40) -> ProbeVerdict:
    """Read vLLM spec-decode acceptance counters from /metrics.

    vLLM exposes accepted/drafted spec tokens under a metrics name containing
    \"spec_decode\" and \"accept\". acceptance = accepted/drafted. The Tony bug
    collapses acceptance from ~0.6 to ~0.25 SILENTLY — no error, wrong weights.
    """
    try:
        text = _get_text(_base_url(base) + "/metrics", api_key)
    except Exception as e:
        return ProbeVerdict("spec_acceptance", "UNKNOWN",
                            f"no /metrics (err={e})")
    lines = [ln for ln in text.splitlines()
             if "spec_decode" in ln.lower()]
    drafted = accepted = None
    for ln in lines:
        body = ln.lstrip("#")
        if "drafted" in body.lower():
            v = _last_metric_value(body)
            if v is not None:
                drafted = v
        elif "accepted" in body.lower():
            v = _last_metric_value(body)
            if v is not None:
                accepted = v
    if accepted is None:
        return ProbeVerdict("spec_acceptance", "UNKNOWN",
                            "no spec-decode accept counters exposed")
    if drafted is None or drafted == 0:
        return ProbeVerdict("spec_acceptance", "UNKNOWN",
                            "drafted counter is 0 (spec decode not active?)")
    rate = accepted / drafted
    if rate < hard_fail_below:
        v = "FAIL"
    elif rate >= pass_at_or_above:
        v = "PASS"
    else:
        v = "UNKNOWN"
    return ProbeVerdict("spec_acceptance", v,
                        f"acceptance={rate:.3f} (accepted={accepted},"
                        f" drafted={drafted})", value=rate,
                        threshold=f"PASS>={pass_at_or_above}, FAIL<{hard_fail_below}")


def _last_metric_value(line: str):
    import re
    m = re.search(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?\s*$", line.strip())
    return float(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# Keys #3 — concurrency: no cudagraph hang, correct output routing, N complete
# --------------------------------------------------------------------------- #
def probe_concurrency(base: str, model: str, api_key: str, n_conc: int = 6,
                      max_tokens: int = 48, timeout_s: int = 180) -> ProbeVerdict:
    """Fire N concurrent distinct-prompt generations.

    Detects request-stable main-KV bugs: a cudagraph hang (incomplete/timeout)
    or cross-request contamination (a response not matching its own prompt's
    topic/nonce). FAIL if any request hangs, errors, or is cross-contaminated.
    """
    url = _base_url(base) + "/v1/chat/completions"
    prompts = unique_prompts(n_conc, 64, seed=7)
    results: list = [None] * n_conc

    def work(i: int):
        nonce = prompts[i].splitlines()[0]  # unique leading nonce
        try:
            r = _post_json(url, {"model": model,
                                 "messages": [
                                     {"role": "user",
                                      "content": f"{prompts[i]}\\nReply with the "
                                                 f"exact nonce: {nonce}"}],
                                 "max_tokens": max_tokens}, api_key,
                           timeout=timeout_s)
            text = (r.get("choices") or [{}])[0].get("message", {}).get(
                "content", "")
            results[i] = {"ok": True, "nonce": nonce, "text": text}
        except Exception as e:
            results[i] = {"ok": False, "nonce": nonce, "error": str(e)}

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout_s + 30)

    incomplete = [i for i, r in enumerate(results) if not r or not r["ok"]]
    contaminations = []
    for i, r in enumerate(results):
        if r and r["ok"] and r["nonce"] not in r["text"]:
            contaminations.append(i)
    if incomplete:
        return ProbeVerdict("concurrency", "FAIL",
                            f"{len(incomplete)}/{n_conc} requests "
                            f"incomplete/timeout (hang or error): {incomplete}")
    if contaminations:
        return ProbeVerdict("concurrency", "FAIL",
                            f"cross-request contamination on ids {contaminations}")
    return ProbeVerdict("concurrency", "PASS",
                        f"{n_conc} concurrent requests all completed with "
                        f"correct per-request nonce routing")


# --------------------------------------------------------------------------- #
# Mia — prefix-cache warm TTFT (reuse of identical prefix)
# --------------------------------------------------------------------------- #
def probe_warm_ttft(base: str, model: str, api_key: str,
                    warm_ratio_below: float = 0.40) -> ProbeVerdict:
    """A warm-prefix 2nd request should be far faster than the cold first one.

    Detects whether the endpoint delivers any prefix-cache/TTFT optimization
    (Mia's warm-cache TTFT headline). UNKNOWN if warm is not meaningfully
    faster (could mean no cache, or no overlap for this endpoint).
    """
    url = _base_url(base) + "/v1/chat/completions"
    prefix = ("nonce-warmcache-2026\\n" + " ".join(["quartz"] * 40) +
              "\\nContinue in detail at length.\\n")
    samples = {}
    for kind in ("cold", "warm"):
        ts = []
        for _ in range(2):
            p = prefix + (f"\\nRound-{kind}-{_}" if kind == "cold"
                          else "\\nRound-warm-continuation")
            payload = {"model": model,
                       "messages": [{"role": "user", "content": p}],
                       "max_tokens": 8, "stream": True,
                       "stream_options": {"include_usage": True}}
            start = time.monotonic()
            t_first = None
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
            with urllib.request.urlopen(req, timeout=300) as resp:
                buf = b""
                while True:
                    c = resp.read(4096)
                    if not c:
                        break
                    buf += c
                    now = time.monotonic() - start
                    text = buf.decode(errors="replace")
                    if t_first is None and '"content":"' in text:
                        t_first = now
            ts.append(t_first if t_first is not None else float("inf"))
        samples[kind] = statistics.median(ts)
    if samples["cold"] in (float("inf"), 0):
        return ProbeVerdict("warm_ttft", "UNKNOWN", "cold TTFT not observed")
    ratio = samples["warm"] / samples["cold"]
    if ratio < warm_ratio_below:
        v = "PASS"
    else:
        v = "UNKNOWN"
    return ProbeVerdict("warm_ttft", v,
                        f"warm/cold TTFT ratio={ratio:.3f} "
                        f"({samples['cold']:.3f}s cold,"
                        f" {samples['warm']:.3f}s warm)",
                        value=ratio, threshold=f"PASS<{warm_ratio_below}")
