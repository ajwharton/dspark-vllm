# PIP INDEX — Port-In Patch registry

Status legend: spec | extracted | applied | gated | merged
Order mirrors PORT-PLAN milestones. Apply in PIPS dependency order below.

| PIP | Milestone | Capability | Source (provenance) | Acceptance | Status |
|-----|-----------|------------|----------------------|------------|--------|
| 200 | M2 | DSpark draft-loader silent 12-tensor drop fix | community correctness patch (tonyd2wild family; audit-confirmed) | dspark_acceptance 60%-class, decode not regressed | spec |
| 201 | M2 | Cold-start reasoning garble / thinking-routing (trap 04/20) | measured on temper DS4F-0731 live; Moet docs | reasoning N/N turns, first-turn clean | measured-spec (gate 3/3 PASS 2026-08-12) |
| 300 | M3 | Request-stable main-KV slot map + ragged query_start_loc | drowzeys/Keys concurrency patch | concurrency >= 6 stable, no #40969 hang | spec |
| 400 | M4 | DSpark normalization (draft count, hf_config_override, k=7) + CUDA-graph-size #6 | divergence map PORT-INVENTORY; Moet-GB10 k=2 measured | acceptance + reasoning preserved | spec |
| 500 | M5 | GB10 kernel layer (B12X, CUTLASS sm121a, FlashInfer sparse-MLA sm120, DeepGEMM) + flashinfer#3817 topk=256 | Moet-GB10 kernels/cubins-sm120; unmerged upstream | decode + fidelity >= baseline | spec |

## Notes
- PIP-201 is the first with live measured evidence (fired the tray gate pass on
  the faithful single-node DS4F rig, 2026-08-12). It is the template for how a
  PIP moves from spec to measured-spec: reproduce, then pin.
- PIP-200 remains the top M2 risk (silent correctness regression). Extraction
  of the concrete draft-loader diff is the immediate next task.
- Detailed per-PIP writeups: `pip-NNN.md`; unified diffs land as `pip-NNN.patch`.
