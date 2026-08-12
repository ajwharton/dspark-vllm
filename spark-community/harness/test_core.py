# No-Nerf gate — core logic tests (stdlib unittest, no hardware required).
import unittest

from .core import compare_median, gate_blocked, unique_prompts


class TestUniquePrompts(unittest.TestCase):
    def test_prompts_are_unique(self):
        prompts = unique_prompts(8, approx_tokens=64, seed=1)
        self.assertEqual(len(prompts), len(set(prompts)),
                         "prompts must be pairwise distinct (nonce + body)")

    def test_length_is_reasonably_scaled(self):
        prompts = unique_prompts(2, approx_tokens=4000, seed=2)
        # ~4000 tokens * 0.75 words/token + nonce line
        for p in prompts:
            self.assertGreater(len(p.split()), 2500)

    def test_deterministic_across_seed(self):
        a = unique_prompts(3, 128, seed=7)
        b = unique_prompts(3, 128, seed=7)
        self.assertEqual(a, b)


class TestGate(unittest.TestCase):
    def test_ttft_worse_fails(self):
        r = compare_median([1.0, 1.0, 1.0], [1.5, 1.5, 1.5], "ttft_s",
                           threshold_pct=10, lower_is_better=True)
        self.assertTrue(r.failed)

    def test_ttft_better_passes(self):
        r = compare_median([1.5, 1.5, 1.5], [1.0, 1.0, 1.0], "ttft_s",
                           threshold_pct=10, lower_is_better=True)
        self.assertFalse(r.failed)

    def test_throughput_drop_fails(self):
        r = compare_median([100, 100, 100], [80, 80, 80], "decode_tok_s",
                           threshold_pct=10, lower_is_better=False)
        self.assertTrue(r.failed)  # dropped 20% (>10% guard)

    def test_within_guard_passes(self):
        r = compare_median([100, 100, 100], [94, 94, 94], "decode_tok_s",
                           threshold_pct=10, lower_is_better=False)
        self.assertFalse(r.failed)  # -6% within guard

    def test_gate_blocked_aggregates(self):
        bad = compare_median([1], [2], "m", 10, True)
        good = compare_median([2], [1], "m", 10, True)
        self.assertEqual(gate_blocked([good, bad]), [bad])


if __name__ == "__main__":
    unittest.main()
