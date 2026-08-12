# PORT-INVENTORY — the roll-up map

Everything this fork intends to consolidate, with origin and upstream status.
Basis: community recipes' attribution lists + upstream vLLM tracking
(verified 2026-08-12). This is the single source of truth for what is rolled
up and what is missing. **Read this before proposing a port.**

## Source & capability map (audit v1 — confirmed repos, 2026-08-12)

Where the capability actually lives. The critical fact: **the same capability
is re-integrated across many independent repos**, so a roll-up must proceed by
*capability* (pick ONE canonical implementation), never by *source* (which
would port the same thing 3–5 times and conflict).

| Source repo | Role | Capabilities it carries |
|-------------|------|--------------------------|
| `rafaelcaricio/vllm` (PR #1) | origin of DSpark-in-vLLM | DSpark integration (spec decode) |
| `fraserprice/dspark-vllm` + HF model | DSpark model + runtime (Blackwell/CUDA-13.2) | DSpark model, sparse-MLA |
| `local-inference-lab/rtx6kpro`, `voipmonitor/vllm` (B12X) | sm120/B12X kernel stack | B12X MoE, sparse-MLA sm120 (serves Fraser's model) |
| `drowzeys/Keys-Concurrency-…` + `Keys---Full-GLM-5.2-Quantrio-…` | concurrency patch + `nvfp4_ds_mla` origin | DSpark 2-node concurrency, NVFP4 KV lineage |
| `tonyd2wild/DeepSeek-v4-Flash-0731-…-2x-DGX-Spark` (39 forks) | NVFP4 1M recipe + correctness | NVFP4 integration, garble fix, draft-loader fix |
| `Anemll/dspark-vllm-gx10` | **our prod base** (img a8394849) | pinned DSpark runtime; divergent DSpark behavior |
| `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` | 2-node launch packaging | worker-first launch, her six v0.27 backports |
| "jasl fork" (4x, RDMA, MTP) | NVLink/RDMA multi-node | MTP path, RDMA tuning |
| upstream `vllm-project/vllm` | canonical line | its OWN DS4/DSpark path (target to converge on) |

**Observed duplication across sources:** DSpark integration appears in 5
places (rafaelcaricio, fraserprice, B12X, Anemll, upstream); NVFP4 KV in 3
(drowzeys, tonyd2wild, Anemll); concurrency in 3 (drowzeys, tonyd2wild, Anemll).
=> Roll up by capability, not by author. Sequence risk-first and gate each.

## Capability table

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
