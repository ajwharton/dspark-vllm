# CONTRIBUTING (community layer)

Contributing to the **Spark community roll-up** (this `spark-community/`
layer). For changes to pure upstream vLLM behavior, also follow vLLM's own
root `CONTRIBUTING.md`; this file governs the Spark delta.

## The rule
We take upstream fixes only when they don't nerf the system. Every
runtime-affecting change must pass the no-nerf gate (`NO-NERF-GATE.md`):
measured, cold-cache, on N-node GB10.

## Branch policy
- `main` (rolling): current upstream vLLM + the Spark delta.
- `stable`: forked from the proven working base; merges only via the gate.
- Feature PRs: short-lived branches off `main`, merged by squash.

## To open a PR
1. Check `PORT-INVENTORY.md` — is this already rolled up? Known broken upstream?
2. Reproduce any upstream claim on N-node GB10 yourself (or get a CI-lab run).
3. Attach the no-nerf gate result for your change (paste the benchmark output).
4. Open the PR against `main` unless it is a stable-only fix.

A runtime change without a gate result is reviewed for risk; a maintainer may
request the bench before merge.

## DCO
Commits must be signed (`git commit -s`) under the Developer Certificate of
Origin. Keeps provenance clean for a community-maintained layer.

## Conduct
Every change in this layer answers to issues #1 and #2 regardless of source
any patch modernization: N-node correctness and reasoning preservation.
```
