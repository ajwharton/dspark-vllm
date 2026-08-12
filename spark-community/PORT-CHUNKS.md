# PORT-CHUNKS — grounded chunking plan against `stable` base

Status: adopted 2026-08-12. Companion to `PORT-PLAN.md` and `PORT-INVENTORY.md`.
This file is the *grounded* decomposition: it anchors every PIP to a REAL seam
on the actual `stable` base (not to the file layout each upstream artifact
happened to use), so contributors port capability, never blindly apply foreign
patches onto foreign seams.

## 0. The base (now grounded)

- Branch: **`stable`** @ `5030cb2`
- = upstream vLLM **v0.25.1** @ `752a3a504485790a2e8491cacbb35c137339ad34`
  + Anemll **0.1.1** DSpark NVFP4 overlay (13 files, from
  `anemll/dspark-vllm-gx10` upstream.lock + `overlay/vllm/*`; overlay LICENSE
  under `LICENSES/anemll-overlay.LICENSE.txt`).
- This is the exact line our frozen prod pin
  `ghcr.io/anemll/dspark-vllm-gx10` (vLLM `0.25.2.dev0+g752a3a504.d20260714`)
  is built from. `stable` is the PIP landing ground; `main` stays generic
  upstream rolling.

### Proven base facts (verified against real tree, not assumed)
- `vllm/models/deepseek_v4/nvidia/dspark.py` -> EXISTS in base (roots at 0.25.1).
  This is the DSpark **spec/draft weight loader seam** on stable.
- `vllm/parser/abstract_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`
  -> EXIST in base. Reasoning seam.
- `vllm/v1/spec_decode/dspark.py`, `vllm/v1/spec_decode/dspark_proposer.py`
  -> **DO NOT EXIST** in base. The tonyd2wild / Keys artifacts that reference
  these target a *different* (newer) fork layout, not ours.

## 1. Chunking doctrine

Chunk = one discrete, seam-anchored, reviewable edit (small diff), each gated
by `NO-NERF-GATE` on the temper rig before it moves to `merged`. Chunks are
independent; order follows the risk/sequence plan in `PORT-PLAN.md`.

## CHUNK-0 (M2, correctness #4) — draft shared-expert load audit on the REAL seam
- Seam: `vllm/models/deepseek_v4/nvidia/dspark.py::DSpark....load_weights`.
- Hollow finding: the generic `stacked_params_mapping`
  (`("gate_up_proj","w1",0),("gate_up_proj","w3",1)` + `.shared_experts.w2`
  -> `.down_proj`) ALREADY catches `shared_experts.w1/.w3` on this base. The
  silent 12-tensor-drop bug (PIP-200) lives in the `spec_decode` layout the
  tonyd2wild patch targets, which is NOT our seam.
- So PIP-200 is **baseline-modulated**: likely already-covered on `stable`.
  DECISION REQUIRES A GATE, not a static read.
- Chunk-0 deliverable: load a real DS4F checkpoint on a `stable` build, assert
  zero uninitialized shared-expert tensors, and gate DSpark acceptance
  (~60%-class healthy). Outcome A: already-covered -> record PIP-200 as
  "already-in-base, verify-only", no code change. Outcome B: tensors still dropped
  -> add the missing rows to OUR stacked mapping (small diff), gate, merge.
- Risk: LOW. Reversible. This is the cheapest correctness chunk to start.

## CHUNK-1 (M2, concurrency #3) — Keys concurrency port to the REAL seam
- Artifact seam: `vllm/v1/spec_decode/dspark_proposer.py` (does not exist here)
  + `vllm/models/deepseek_v4/nvidia/dspark.py` (exists here).
- Port target: the main-KV slot map + ragged `query_start_loc` concurrency must
  be re-expressed against STABLE's `nvidia/dspark.py` + its v1 worker seam
  (`vllm/v1/worker/gpu_model_runner.py`), NOT applied.
- Chunk-1 deliverable: concurrency capability present + no cudagraph-hang
  class regression (measured), gated. Expect a real, seam-rewritten diff
  (~the 3 hunk-groups of PIP-300, adapted).

## CHUNK-2 (M2, reasoning) — Moet reasoning route to the REAL seam
- Seam: `vllm/parser/abstract_parser.py` + `vllm/reasoning/deepseek_v3_reasoning_parser.py`
  (both EXIST on base). PIP-201 artifact was sliced from Moet 0.24; base is
  0.25.1 — assess drift first; may be closer than on `main`.
- Gate: harness reasoning 3/3 (api_key + enable_thinking path already added).
- Flag note (governance #9): thinking enablement is model-specific -> must stay
  an optional, default-off flag.

## CHUNK-3+ (deferred)
- PIP-400/500 (still `spec`). GGUF seams + DSpark k=2 tuning are rig work, not
  stable-tree code; tracked in `PORT-PLAN.md`.

## 2. Blockers / by design
- The three PIP artifacts cannot be `git apply`'d onto `main` (drift) NOR page for
  the strong majority of `stable` (foreign seam). They are CAPABILITY references:
  read the intent, re-implement on our seam, gate. `apply.sh` remains the
  correctness checker once a reroll exists; it must be run from REPO ROOT
  (cwd-relative git apply otherwise false-cleans).
- `stable` does not yet build/run standalone on temper — gating chunk-0 first
  requires a `stable`-source container (build or bind-mount). Not started.

## 3. Expected baselines to hold (no-nerf)
- DSpark acceptance ~60%-class (collapse to ~25% = the bug we guard).
- Decode ~11.5 tok/s long-run warm (temper, Moet k=2 faithful) / harness
  scoped variant; cold-TTFT median ~1.35s @256 unique tok; reasoning 3/3.
