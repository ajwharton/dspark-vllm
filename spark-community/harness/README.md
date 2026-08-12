# No-Nerf Gate harness.

Public reference-bench harness that enforces the fork's "upstream fixes only
where they do not nerf the system" rule on a 2-node DGX-Spark / GB10 reference
bench (rolling line, architecture C). Pure core logic is stdlib-only and
importable/testable; IO lives in cli.py.

## Why cold

Warm/identical-prompt benches compress TTFT by exploiting the prefix cache and
inflate throughput claims (this is how "80% faster at 256k" got published but
did not survive an honest cold measurement). This harness:
- `gen`: writes prompts each with a unique nonce so the prefix cache can never
  serve one prompt's prefill to another (true cold prefill).
- `bench`: interleaved A/B (baseline vs candidate) over the same prompt set to
  keep thermal/driver drift symmetric; reports medians and a regression gate.
- `reasoning`: multi-turn reasoning-preservation probe (trap 04/20) — catches
  the default template silently dropping thinking on conversation resend.

## Usage (requires 2x GB10 endpoints serving the SAME model commit)

  python -m harness.cli gen      --prompts 8 --approx-tokens 4096
  python -m harness.cli bench    --base-url http://stable:8000 \
                                 --cand-url http://rolling:8000 \
                                 --model deepseek-v4-flash-0731-dspark \
                                 --metric ttft --prompts 3 --threshold 10
  python -m harness.cli reasoning --url http://rolling:8000 --model <m> --turns 3

## Gate semantics (see ../docs/spark-no-nerf-gate-spec.md)

- lower_is_better (ttft): candidate FAILS if its median is >baseline*(1+T/100).
- higher_is_better (decode): candidate FAILS if its median is <baseline*(1-T/100).
- exit code 1 = FAIL (blocks merge); 0 = pass.

## Test without hardware

  python -m unittest harness.test_core -v

## Manifest

Copy harness/manifest.template.yaml to manifest.yaml before any run; record
model/runtime commits, seeds, env vars so results are attributable.
