# SPARK-COMMUNITY

This fork of upstream vLLM carries the **DGX Spark (GB10 / N-node) DeepSeek-V4
DSpark community roll-up**. The Spark delta lives in [`spark-community/`](spark-community/);
the rest of the tree is pristine upstream vLLM (clean upstream sync via the fork
network).

Read first: [`spark-community/README.md`](spark-community/README.md) and
[`spark-community/PORT-INVENTORY.md`](spark-community/PORT-INVENTORY.md).

**The rule:** upstream mainline fixes are taken only when they do not nerf the
working system on GB10 / N x DGX-Spark — enforced by the no-nerf gate
([`spark-community/NO-NERF-GATE.md`](spark-community/NO-NERF-GATE.md)).

**Endgame:** this is a proving ground, not a permanent parallel. Every Spark
delta is developed, hardened, and measured here under the gate so that, as
local-AI-on-DGX-Spark adoption grows, the working capability can be advocated
and merged back into upstream vLLM main with clean, reproducible evidence —
retiring the fork rather than defending it.
