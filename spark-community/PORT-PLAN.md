# PORT-PLAN — sequenced milestones + CI-lab topology

Companion to `PORT-INVENTORY.md` and `NO-NERF-GATE.md`. Everything is gated by
the no-nerf reference bench on N-node GB10.

## Ordering principle
Correctness and concurrency first (silent, high impact, block real use);
speculation and kernels last (high churn, hardware-bound). A step is "done"
only when its guarding gate metric passes on N-node GB10.

## Milestones
### M1 — Provenance & baseline (blocked on a dedicated bench pair)
Freeze the proven recipe base -> `stable`. Reproduce the reference bench on the
chosen N-node pair; record baseline for every gate metric.
Exit: attributable baseline manifest (`harness/manifest.yaml`). Nothing merges
before this.

### M2 — Correctness roll-up (items #4, #5)
#4 DSpark draft-loader silent 12-tensor drop fix.
  Acceptance: DSpark acceptance restored (60%-class, not 25%); decode not regressed.
#5 Cold-start agent garble fix. Acceptance: first-turn clean; reasoning preserved.

### M3 — Multi-node concurrency (item #3)
Request-stable main-KV slot map + ragged query_start_loc for real
independent-arrival batching. Acceptance: concurrency >= N stable; no
#40969-class hang; decode held.

### M4 — DSpark normalization / speculation (items #1, #6)
Reconcile the documented divergences (draft count, hf_config_override, k=7).
#6 CUDA-graph capture-size fix first. Acceptance: acceptance + reasoning
preserved; no divergence-dependent behavior.

### M5 — Kernel layer provisioning (items #8, #9)
GB10 kernel layer + unmerged flashinfer#3817 topk=256 (carry until released).
Acceptance: decode + fidelity meet/exceed baseline under the gate.

### M6 — Rolling-line convergence + upstream intake
With stable load-bearing, bring `main` (rolling) up: current upstream vLLM +
only the not-upstreamable delta, gated. Acceptance: rolling passes the same
gate as stable before promotion.

## CI-lab topology
- Self-hosted GitHub Actions runners, label `dgx-spark-2node` (see
  `.github/workflows/no-nerf-gate.yml`). GitHub has no free hosted GB10 runners.
- The bench is an interleaved A/B between a "stable" endpoint and a
  "rolling"/candidate endpoint served on the same bench pair during a quiet
  window — bench must not fight live serving.
- Env vars the workflow needs once CI is live:
  `GATE_STABLE_URL`, `GATE_ROLLING_URL`, `GATE_MODEL`.
- Runner install/register per node is the standard GitHub Actions self-hosted
  flow; register only after the bench pair is chosen (M1). Preflight on each
  node: `docker run --rm --gpus all` works; harness self-test passes.

## Cross-cutting
- Every step logs a gate result into `harness/manifest.yaml`.
- Record both the upstream claim and our measured result when they differ.
- No merge of a runtime change without the gate (or an explicit maintainer risk
  verdict documented in the PR).
