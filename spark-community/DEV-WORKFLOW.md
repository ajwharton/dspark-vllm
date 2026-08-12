# DEV-WORKFLOW — single-node dev rig + staged N-node validation

How this fork's experimentation is run. Read before contributing hardware time.

## Model
1. **Develop on a single node (the "rig")** for coherence and speed. A single
   DGX-Spark/GB10 (and the wider fleet of *single*-node Spark users, who are a
   first-class audience here) exercises the stack without needing a 2-node/full
   fp8 setup. One node, fully driven, is where porting, reasoning-preservation,
   and harness development happen.
2. **Proxy-resemblance:** because full DeepSeek-V4-Flash is N-node/fp8-heavy and
   won't fit one node, the rig runs a **single-node MoE model that resembles
   DS4F's architecture** (MoE + speculative decode + reasoning) as the dev
   stand-in. The stack (serving path, KV quant, reasoning-preservation trap
   04/20, the no-nerf harness, port patches) is developed and gated *single-node*
   against this proxy.
3. **Stage the final run for Grok.** The authoritative N-node (>=2), full-DS4F
   gate is a high-stakes, hardware-bound run. It is **staged for Grok to
   orchestrate at the end** from a prepared handoff: pinned repro, bench
   topology, expected results, and the gate spec. Vulcan prepares + validates
   the rig and harness and writes the handoff; Grok executes the final
   N-node staging runs. This mirrors the existing doctrine for delegating risky
   host runs via a prepared handoff document.

## Proxy model roles (two, kept distinct)
- **(a) DS4F-faithful proxy:** a single-node DeepSeek-V4-Flash-DSpark GGUF
  (~22GB) — real DS4F MoE + DSpark architecture, single-node-loadable. Used for
  DS4F-behavior/coherence and reasoning-preservation development (its limitation
  is that GGUF loads via the llama.cpp path, not vLLM's safetensors DS4 path).
- **(b) vLLM-stack proxy:** a safetensors MoE (DeepSeek-family or equivalent)
  that loads through vLLM's real path, to exercise vLLM's DSpark/spec decode +
  KV-quant + tracing on one node. (Selection to be verified on the rig.)

## Gate coverage
- The single-node gate runs N=1 metrics (reasoning preservation, decode/TTFT,
  KV fidelity, concurrency>=1) via `harness/`.
- The N>=2 gate (multinode concurrency, dual-node throughput) is the staged,
  Grok-orchestrated run. Both use the same `harness/` and `NO-NERF-GATE.md`
  rule; only the bench topology differs.

## Rig handoff artifact
Before any final N-node run, `rig/handoff.md` (prepared by Vulcan) contains:
pinned model+runtime commits, image SHAs, env, the exact benchmark commands,
expected gate baselines, and the named executor (Grok). Nothing about the final
run is left to improvisation.
