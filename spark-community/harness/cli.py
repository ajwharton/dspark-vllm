# No-Nerf Gate — CLI entrypoints. Thin IO over core.py logic.
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .core import (VLLMClient, compare_median, gate_blocked, median,
                   unique_prompts)


def _out_path(arg: str) -> Path:
    p = Path(arg)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cmd_gen(args: argparse.Namespace) -> int:
    prompts = unique_prompts(args.prompts, args.approx_tokens, seed=args.seed)
    _out_path(args.out).write_text(json.dumps(prompts))
    print(f"wrote {len(prompts)} unique prompts -> {args.out}", file=sys.stderr)
    return 0


def _run_batch(client, prompts, metric):  # -> list[float] values
    vals = []
    for p in prompts:
        s = client.cold_ttft(p) if metric == "ttft" else client.decode_throughput(p)
        vals.append(s.value)
    return vals


def cmd_bench(args: argparse.Namespace) -> int:
    """Interleaved A/B: baseline vs candidate, set of unique prompts each."""
    gen = unique_prompts(args.prompts, args.approx_tokens, seed=args.seed)
    base = VLLMClient(args.base_url, args.model, api_key=args.api_key)
    cand = VLLMClient(args.cand_url, args.model, api_key=args.api_key)
    # Interleave by running alternately across the shared prompt set, seeded.
    rng = random.Random(args.seed)
    order = list(range(len(gen)))
    rng.shuffle(order)
    base_vals, cand_vals = [], []
    for idx in order:
        cand_vals.append(
            cand.cold_ttft(gen[idx]).value if args.metric == "ttft"
            else cand.decode_throughput(gen[idx]).value)
        base_vals.append(
            base.cold_ttft(gen[idx]).value if args.metric == "ttft"
            else base.decode_throughput(gen[idx]).value)
    lower_is_better = args.metric == "ttft"
    res = compare_median(
        baseline=base_vals, candidate=cand_vals, metric=args.metric,
        threshold_pct=args.threshold, lower_is_better=lower_is_better)
    print(res.summary)
    if args.json:
        print(json.dumps({
            "metric": res.metric, "baseline_median": res.baseline_median,
            "candidate_median": res.candidate_median, "delta_pct": res.delta_pct,
            "threshold_pct": res.threshold_pct, "failed": res.failed,
            "baseline_n": len(base_vals), "candidate_n": len(cand_vals),
        }))
    return 1 if res.failed else 0


def cmd_reasoning(args: argparse.Namespace) -> int:
    chat_kwargs = {"enable_thinking": True} if args.think else None
    client = VLLMClient(args.url, args.model, api_key=args.api_key,
                        chat_template_kwargs=chat_kwargs)
    samples = client.reasoning_probe(turns=args.turns)
    present = sum(1 for s in samples if s.reasoning_present)
    print(f"reasoning present in {present}/{len(samples)} turns")
    return 0 if present == len(samples) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness", description="No-Nerf gate bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--prompts", type=int, default=8)
    g.add_argument("--approx-tokens", type=int, default=4096)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="requests.jsonl")
    g.set_defaults(fn=cmd_gen)

    b = sub.add_parser("bench")
    b.add_argument("--base-url", required=True)
    b.add_argument("--cand-url", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--api-key", default="")
    b.add_argument("--metric", choices=["ttft", "decode"], required=True)
    b.add_argument("--prompts", type=int, default=3)
    b.add_argument("--approx-tokens", type=int, default=4096)
    b.add_argument("--threshold", type=float, default=10.0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("reasoning")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--api-key", default="")
    r.add_argument("--turns", type=int, default=3)
    r.add_argument("--think", action="store_true",
                   help="request enable_thinking (required for DS4F reasoning field)")
    r.set_defaults(fn=cmd_reasoning)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
