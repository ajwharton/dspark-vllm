# Patch package — Port-In Patches (PIPs)

The roll-up is delivered as a set of **Port-In Patches (PIPs)** that carry the
DS4F / DSpark / multi-node capability onto the fork's bases (`stable` and
`main`/rolling). This directory is the package. Every PIP is auditable,
provenance-bound, and no-nerf-gated before merge.

## Why a package, not ad-hoc diffs

The source forks (tonyd2wild, drowzeys/Keys, fraserprice, rafaelcaricio, Mia,
jasl, Moet/Moet-GB10, Anemll) all re-integrate the *same* capabilities on
*different* bases. Bundling them raw would re-create the divergence this fork
exists to resolve. Each PIP therefore:

- pins one capability (not an author bundle);
- records source provenance (repo + commit) OR measured evidence;
- names its target base (stable vs rolling) for this port;
- carries an acceptance gate metric from NO-NERF-GATE.md;
- moves `spec -> extracted -> applied -> gated -> merged` in order.

## Ordering (from PORT-PLAN.md)

M2 correctness (PIP-200, PIP-201) -> M3 concurrency (PIP-300) ->
M4 speculation (PIP-400) -> M5 kernels (PIP-500). Higher milestone PIPs depend
on lower ones; do not apply out of order.

## Workflow

    ./apply.sh dry-run main      # check what applies cleanly to a base
    ./apply.sh main              # apply eligible PIPs in dependency order
    <run the no-nerf gate + the PIP's acceptance probe>

## Merge rule

A PIP becomes `merged` only when its gate/acceptance metric passes on N-node
GB10, or an explicit maintainer risk verdict is recorded in the PR. Record the
result in `../NO-NERF-GATE.md` and `manifest.yaml`.

Registry of PIPs and status: `PIP-INDEX.md`. Per-PIP detail: `PIP-NNN.md`.
