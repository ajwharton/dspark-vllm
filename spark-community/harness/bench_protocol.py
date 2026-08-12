#!/usr/bin/env python3
"""No-Nerf Gate — standardized bench CLI (confound-controlled).

Usage:
  python3 bench_protocol.py --base http://localhost:8000 \
      --model deepseek-v4-flash-0731 --key change-me \
      --mode decode --thinking off --out-tokens 256 --prompt-tokens 256 --rounds 3

Modes:
  decode  : true generation-only decode tok/s (streamed, prefill excluded)
  ttft    : cold time-to-first-token (unique nonce prompts)
  full    : both, standardized
Rules: thinking state explicit; fixed lengths; median + min reported.
"""
from __future__ import annotations

import argparse
import sys

from core import unique_prompts
from measure import median_run, stream_generate


def main() -> int:
    ap = argparse.ArgumentParser(description="standardized no-nerf bench")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--mode", choices=["decode", "ttft", "full"], default="full")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    thinking = True if args.thinking == "on" else False
    prompts = unique_prompts(args.rounds, args.prompt_tokens, seed=args.seed)

    if args.mode in ("ttft", "full"):
        print(f"== cold TTFT (thinking={args.thinking}, prompt~{args.prompt_tokens}t) ==")
        res = [stream_generate(args.base, args.model, p, 8, api_key=args.key,
                               enable_thinking=thinking, timeout=900)
               for p in prompts]
        a = median_run(res)
        print(f"  ttft_median={a['ttft_median_s']:.2f}s min={a['ttft_min_s']:.2f}s "
              f"(n={a['n']})")

    if args.mode in ("decode", "full"):
        print(f"== decode (thinking={args.thinking}, out={args.out_tokens}t, "
              f"prompt~{args.prompt_tokens}t) ==")
        res = [stream_generate(args.base, args.model, p, args.out_tokens,
                               api_key=args.key, enable_thinking=thinking,
                               timeout=900)
               for p in prompts]
        a = median_run(res)
        print(f"  throughput_median={a['throughput_median_tok_s']:.2f} tok/s "
              f"min={a['throughput_min_tok_s']:.2f} "
              f"(completion_med={a['completion_median']}, n={a['n']})")
        print(f"  all_decode_ok={a['all_decode_ok']} "
              f"reasoning_present_any={a['reasoning_present_any']} "
              f"wall_median={a['wall_median_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
