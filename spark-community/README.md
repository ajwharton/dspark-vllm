# spark-community — the DGX Spark roll-up layer

This directory is the **community roll-up** living inside a fork of upstream
vLLM. It exists because the DGX Spark (GB10 / N-node) DeepSeek-V4 DSpark
capability was never cleanly in upstream vLLM — it is a scattered patch
ecosystem across past forks (Anemll, tonyd2wild, drowzeys/Keys, fraserprice,
rafaelcaricio, and others). This layer re-assembles it in ONE governed place and
sets the rule for taking upstream fixes.

**The rule (the "no-nerf gate"):** we take upstream mainline vLLM fixes only
when they do not nerf the working system on GB10 / N x DGX-Spark. Every
runtime-affecting change must pass a measured, cold-cache reference bench.

## Why a fork of vLLM, not a from-scratch repo
Being a true fork on GitHub's fork network means upstream vLLM stays one click
away: we can pull mainline fixes and push them through the gate without
re-assembling history. The Spark delta lives in this layer only, leaving the
upstream tree pristine so upstream merges stay clean.

## Contents
- `PORT-INVENTORY.md` — the roll-up map: every capability, origin, upstream
  status, and its nerf risk. Read this first.
- `NO-NERF-GATE.md` — the acceptance rule + reference-bench spec that gates
  every merge.
- `PORT-PLAN.md` — sequenced milestones (M1–M6) and the CI-lab / bench topology.
- `MAINTAINERS.md` — governance and merge rights.
- `CONTRIBUTING.md` — how to contribute, including the gate requirement.
- `harness/` — the gate tooling (stdlib-only, self-testing).

## Two-track policy
| Track | Base | Purpose |
|-------|------|---------|
| `stable` | proven working base | stays shippable / N-node safe today (advances only via a passing no-nerf gate) |
| `main` (rolling) | current upstream vLLM | converges under the no-nerf gate; auto-syncs upstream daily (see `.github/workflows/upstream-sync.yml`) |

## Endgame: merge back into upstream vLLM
This layer is a proving ground, not a permanent parallel. Every Spark delta is
developed and measured here under the gate so that, as local-AI-on-DGX-Spark
adoption grows, the working capability can be advocated and merged back into
upstream vLLM main with clean, reproducible evidence — retiring this fork
rather than defending it. This is WHY we take upstream fixes that don't nerf:
minimizing divergence keeps the eventual upstreaming honest and small.

## Risks — why this hasn't been done before (honest, recorded up front)
Nobody consolidated this ecosystem yet, and the reasons are structural, not
oversight. Reading them as risks up front is how we avoid being surprised:
- **Fragmented, divergent bases.** Each past fork made different assumptions
  (draft sizing, `hf_config_override`, k values). There is no git-level merge;
  reconciling them is design work, not cherry-picking.
- **Hardware-scarce validation.** Reproducing a claim needs a dedicated N-node
  GB10 bench; most contributors have one box under load, so claims are hard to
  verify and distrust builds. The gate only works if a bench exists.
- **Single-maintainer, territorial patches.** Working fixes live with their
  authors; consolidation needs them to cede or collaborate. Governance and
  credit must be handled carefully or the "one place" won't actually converge.
- **Silent regression risk.** "Newer base lost to recipe" in a community
  bakeoff — moving toward current upstream can nerf the very capability people
  rely on, invisibly. The gate is the guard, but it must be built before we move.
- **Beyond-the-fork blockers.** Some pieces are vendor or unmerged upstream
  (e.g. flashinfer#3817 sm120 topk=256); final upstreamization depends on
  parties outside this repo.
- **Early, fragile market.** Local-AI-on-Spark is nascent; the pool of active
  maintainers is small, so momentum is thin. A few gate-passing reference runs
  and clear credit are what keep people contributing.

None of these block the start; all of them shape how we proceed (evidence
over claims, credit carefully, gate before moving).

## Quick start
```
cd spark-community/harness && python3 -m unittest test_core -v   # verify gate tooling
```
See `PORT-INVENTORY.md` for what is rolled up and `NO-NERF-GATE.md` before
opening a PR.

## Status
- [x] fork created (upstream + this layer)
- [x] governance + gate tooling seeded
- [ ] stable track populated
- [ ] rolling track ported
- [ ] no-nerf CI-lab wired (N-node self-hosted runner)
- [ ] first public release
