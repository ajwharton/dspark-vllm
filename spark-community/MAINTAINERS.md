# MAINTAINERS

Governance for the DGX Spark community layer of this fork. Small, pragmatic,
evidence-gated.

## Merge rights
- Maintainers have write + merge. Runtime-affecting merges are gated on the
  no-nerf bench (or an explicit maintainer risk verdict when a bench pair is
  unavailable).
- `stable` has a freeze-on-regression rule: merge only with a passing gate;
  regressions must be reverted or toggled.

## Responsibilities
- Triage upstream PR candidates against `PORT-INVENTORY.md`.
- Run or request no-nerf bench runs for runtime changes.
- Report regressions upstream (vLLM) with measured evidence.
- Keep `PORT-INVENTORY.md` current — it is the map of the roll-up.

## Initial ownership
- Lead/founding: Andrew (ajwharton)
- Tooling/infra: Vulcan (AI maintainer: gate harness, CI-lab)
- Open to community maintainers who demonstrate sustained, gate-passing work.

## Decision rule
Disputes resolve by evidence: reproduce both claims on the reference bench.
If hardware cannot reproduce, say so — do not guess. Escalation is upstream
(vLLM) with measured diffs.

## Onboarding
1. Nominated by an existing maintainer.
2. Multiple gate-passing PRs + participation in upstream reporting.
3. Community vote or founding-maintainer sign-off.
