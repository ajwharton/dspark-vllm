# Agentic Code-Creation Governance

Status: adopted (2026-08-12). Applies to every contribution to this repo,
whether authored by a human engineer or by an agent. It binds contributors who
"pick up" the roll-up (e.g. upstream-authors whose patches we modernize) just
as it binds the maintainers' own tooling.

## Purpose

Set the non-negotiable bar for code written into this project. The bar exists
because human engineers — including our near-term contributors — must be able
to review, trust, and revert every change quickly. Small, tested, attributable
changes are the unit of safe work, and they are what make a community-
maintained roll-up possible.

## Scope

Binds all agent-authored and human-authored changes: feature code, fixes,
refactors, configuration, and scripts. It is not optional and is not relaxed
for automated or "low-risk" changes.

## Gates (mandatory)

A change must pass every gate before it is finished.

1. Minimal diff. The change is the smallest set of edits that satisfies the
   task. No extra edits, no opportunistic cleanup, no renames or reformatting
   of unrelated code.

2. Smallest reasonable fix. For a bug or correction, fix the root cause with
   the fewest moving parts. Do not build a framework to solve a one-line
   problem.

3. Resist refactoring and oversized changes. Refactor only when the task
   itself requires it. When a large change is genuinely needed, split it into
   small, independently reviewable pieces with a clear sequence. A 500-line
   diff is a smell, not a badge of honor.

4. Human reviewability. Optimize for a human being able to read the diff in a
   few minutes and understand exactly what changed and why. If a human cannot
   review it quickly, it is too big or too clever.

5. Tests are mandatory. Every behavioral change ships with a test that would
   fail without the change. New or risky paths live in the canonical test
   suite, not in ad-hoc scripts. A change without a test is not done.

6. Provenance and reversibility. Every change is attributable (who, when, why)
   and reversible. For risky operations, prefer rename-before-replace so a
   rollback is a rename, not surgery. Never commit credentials. Every commit
   is signed under the DCO.

7. Falsifiable evidence. Claims made during review — throughput, correctness,
   "this fixes it" — are supported by measured evidence, not assertion. If you
   state a result, produce the command and output that proves it. Runtime-
   affecting claims must show a no-nerf gate result (`NO-NERF-GATE.md`).

8. Humility and scope pushback. Prefer the simplest correct change. Speak up
   when scope creeps, whether the creep comes from a human or from another
   agent. Difficulty of review is the author's failure, not the reviewer's.

9. Capability gating behind optional flags (community unblock). Any behavior
   tied to a specific model, model family, or node count must NOT be
   hard-wired into the shared path. Ship it as an OPTIONAL, DEFAULT-OFF
   startup flag so a generic user's setup never breaks and the capability
   stays available to those who need it. Default must always preserve the
   existing, known-good behavior of upstream vLLM. This is how we roll
   single-model and N-node-specific work up into a community build without
   regressing anyone. Port by capability (PIP), not by author or by
   "my machine": if a change only helps a particular model/nodes, it is a
   flag, gated behind a documented CLI switch with a no-nerf gate result
   recorded in the manifest. `main`/`rolling` stays generic + safe; the
   flag surfaces the specific capability.

## What failure looks like

- A diff that also "cleans up" unrelated files.
- Fixing a bug by replacing the surrounding function or rewriting the module.
- Shipping behavior with no test.
- A claim ("it is faster", "this fixes it", "acceptance is fine") with no
  command and output.
- Changes so large the reviewer cannot identify the actual change.

## Enforcement

- Every pull request and every agent handoff runs the review checklist below.
- A change that fails a gate is returned for revision. It is not merged to buy
  time.
- Agents record a checkpoint after the change, so the work is auditable and
  self-learning.

## Review checklist (run on every change before merge)

- [ ] Minimal diff (nothing unrelated)
- [ ] Smallest reasonable fix (no framework for a one-liner)
- [ ] No drive-by refactor
- [ ] Human can review in minutes
- [ ] Test added to canonical suite, demonstrates the fix
- [ ] Provenance recorded (author, rationale, revert path); DCO signed
- [ ] Any performance/correctness claim backed by measured output (gate result
      for runtime-affecting changes)
- [ ] No scope creep, and pushback was registered if scope was inflated
