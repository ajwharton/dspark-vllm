# NO-NERF-GATE — acceptance rule + reference-bench spec

The enforcement mechanism for "take upstream mainline vLLM fixes only where
they do not nerf the system" on N x DGX-Spark (GB10). Every runtime-affecting
change must pass this gate on a reference bench BEFORE merge.

## 0. Principles
- Measured, falsifiable, cold-cache. No warm-cache / identical-prompt
  compression of TTFT claims (that is how long-context "speedup" claims get
  inflated — the prefix cache does the work, not the patch).
- Deterministic seed + pinned model commit + pinned runtime, so regressions
  are attributable.
- Fail = merge blocker. A regressing PR is held behind a feature toggle and
  reported upstream with measured evidence — never silently absorbed.

## 1. Reference bench topology
- N x DGX-Spark (GB10) over the NVLink/IB mesh used by the recipe (N>=2).
- Model: deepseek-v4-flash-0731 DSpark (fp8 nvfp4_ds_mla KV), pinned commit.
- Runtime under test (candidate vs stable baseline) on the SAME bench pair.
- Runs: >=3, report median; interleaved A/B, not serial (thermal/driver drift).

## 2. Gate metrics (median; each vs stable baseline)
| Metric | Baseline | Guard (regression fails) | Notes |
|--------|----------|--------------------------|-------|
| Decode throughput (tok/s sustained) | measured | within -5% | steady-state, >= few hundred tokens |
| Cold TTFT (unique prefill, cache bypassed) | measured | within -10% | nonce-per-query defeats prefix cache |
| Warm TTFT / cache-hit floor | measured | report only | not a gate; context-length honesty |
| KV-quant fidelity (nvfp4_ds_mla) | measured | within tolerance | detect silent fidelity drop |
| Reasoning preservation, multi-turn | pass | PASS | reasoning must persist across resend |
| DSpark acceptance rate | measured | within -X% | catches the silent 12-tensor-drop class |
| Concurrency stability >= N req | pass | PASS | catches #40969-class hang |

## 3. Harness requirements (build once, public)
- Scriptable cold + warm bench (unique-nonce prefill generator + warm replay).
- Acceptance-rate probe; multi-turn reasoning-preservation probe (trap 04/20).
- Perplexity (KV fidelity) probe. Interleaved A/B runner + median + gate
  (non-zero exit on FAIL). Reproducibility manifest (seeds, commits, images).

Tooling: see `harness/` in this layer (stdlib-only, self-testing).

## 4. Candidate intake workflow
1. Open PR against `main` (rolling) with the PR's claimed metric + this bench.
2. CI-lab runs the gate on the reference bench (`../.github/workflows/no-nerf-gate.yml`).
3. Gate result attached to PR; merge only on PASS; FAIL -> posted upstream with
   the measured diff and held behind a toggle.
