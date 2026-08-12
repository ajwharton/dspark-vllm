# PORT-INVENTORY — the roll-up map

Everything this fork intends to consolidate, with origin and upstream status.
Basis: community recipes' attribution lists + upstream vLLM tracking
(verified 2026-08-12). This is the single source of truth for what is rolled
up and what is missing. **Read this before proposing a port.**

Legend — Upstream status: `merged` = cleanly in upstream vLLM;
`partial` = partly absorbed / fragmented; `fork` = only in a fork (we must
carry it); `unmerged` = PR open upstream, not merged/released.

| # | Capability | Origin | Upstream | Nerf risk | In stable? | In rolling? |
|---|-----------|--------|----------|-----------|------------|-------------|
| 1 | DSpark speculative decoding (fused draft / Markov head) | rafaelcaricio, fraserprice, DeepSeek DeepSpec | partial | HIGH (divergent + unmerged mods) | via base | port |
| 2 | `nvfp4_ds_mla` KV cache | drowzeys (Keys), GLM Quantrio | partial | MEDIUM | via base | port |
| 3 | Real N-node concurrency (request-stable main-KV slot map, ragged query_start_loc) | drowzeys "Keys Concurrency Patch" | fork | HIGH | via base | port |
| 4 | DSpark draft weight loader — silent 12-tensor drop (draft shared expert); acceptance 60%→25% | tonyd2wild Patch 4 | fork | HIGH (silent throughput loss) | needed | needed |
| 5 | Cold-start agent garble fix | tonyd2wild Patch 3 | fork | LOW | needed | needed |
| 6 | CUDA-graph capture-size fix | Wpnx330 | unmerged | MEDIUM | needed | needed |
| 7 | Long-context engine-death fix; slot-corruption instrumentation | tonyd2wild | fork | MEDIUM | needed | needed |
| 8 | GB10 kernel layer (B12X MoE, CUTLASS/cuTe sm121a, FlashInfer sparse-MLA sm120, DeepGEMM) | fork/vendor | fork | MEDIUM | via base | port |
| 9 | flashinfer sm120 topk=256 (deepseek_v4 dsc test) | flashinfer#3817 | unmerged (not released) | MEDIUM | skip | carry |

## Known broken upstream (tracked; do not ship on prod today)
- vLLM #51009 — DSpark acceptance collapse
- vLLM #40969 — CUDA-graph hang ~6 concurrent requests
- vLLM #51758 / #50796 — SM120 DeepGEMM

## Documented divergences to reconcile
- The Anemll fork branch differs: no `hf_config_override` branch; draft sized
  from `num_speculative_tokens` not `dspark_block_size`; k=7 works there only.
  => "same patch, different behavior" across forks must be normalized on roll-up.
- Community bakeoff (2026-07-29): a newer vLLM base LOSES to the optimized
  recipe fork on the same N-node GB10 hardware (recipe peak ~84 tok/s).
  => Evidence, not claim: verify a newer base before trusting it.

## Port order (risk-first)
1. Correctness: #4 draft-loader fix, #5 garble.    (silent, high impact)
2. Concurrency: #3 slot-map / ragged offsets.       (blocks real multi-request)
3. Speculation: #1 DSpark normalization (#6 cu-graph first).
4. Kernels: #8, #9 last.                            (sm120/DeepGEMM maturation)

Each step must pass the no-nerf gate (`NO-NERF-GATE.md`) on N-node GB10 before
the next is accepted.
