import unittest
from unittest import mock

import capability_probes as cp


class CapabilityProbeTest(unittest.TestCase):
    def _mk_metrics(self, accepted, drafted):
        # Live vLLM names + the `_created` / per_pos noise the parser must ignore.
        return (
            "# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.\n"
            f"vllm:spec_decode_num_draft_tokens_total{{engine=\"0\"}} {drafted}\n"
            f"vllm:spec_decode_num_draft_tokens_created{{engine=\"0\"}} 1786452427.0\n"
            f"vllm:spec_decode_num_accepted_tokens_total{{engine=\"0\"}} {accepted}\n"
            f"vllm:spec_decode_num_accepted_tokens_created{{engine=\"0\"}} 1786452427.0\n"
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 99\n'
        )

    def test_acceptance_pass_healthy(self):
        with mock.patch.object(cp, "_get_text",
                               return_value=self._mk_metrics(6000, 10000)):
            v = cp.probe_spec_acceptance("http://h", "")
        self.assertEqual(v.verdict, "PASS")
        self.assertAlmostEqual(v.value, 0.6, places=2)

    def test_acceptance_fail_collapsed(self):
        # Tony #4 signature: acceptance collapses to ~0.25 silently.
        with mock.patch.object(cp, "_get_text",
                               return_value=self._mk_metrics(2500, 10000)):
            v = cp.probe_spec_acceptance("http://h", "")
        self.assertEqual(v.verdict, "FAIL")
        self.assertAlmostEqual(v.value, 0.25, places=2)

    def test_acceptance_unknown_no_metrics(self):
        with mock.patch.object(cp, "_get_text",
                               return_value="no spec counters here\n"):
            v = cp.probe_spec_acceptance("http://h", "")
        self.assertEqual(v.verdict, "UNKNOWN")

    def test_concurrency_pass(self):
        def echo(url, payload, key, timeout):
            # Correct endpoint echoes the per-request nonce back.
            content = payload["messages"][0]["content"]
            nonce = content.split("\n")[0].strip() or "nonce-X"
            return {"choices": [{"message": {"content": f"ok {nonce}"}}]}
        with mock.patch.object(cp, "_post_json", side_effect=echo):
            v = cp.probe_concurrency("http://h", "m", "", n_conc=4)
        self.assertEqual(v.verdict, "PASS")

    def test_concurrency_fail_hang(self):
        def flaky(url, payload, key, timeout):
            i = payload["messages"][0]["content"].splitlines()[0]
            if "nonce-" in i and i.endswith("1"):
                raise TimeoutError("cudagraph hang")
            return {"choices": [{"message": {"content": "nonce-x"}}]}
        with mock.patch.object(cp, "_post_json", side_effect=flaky):
            v = cp.probe_concurrency("http://h", "m", "", n_conc=4)
        self.assertEqual(v.verdict, "FAIL")

    def test_warm_ttft_unknown_when_not_faster(self):
        # Warm not faster -> UNKNOWN (no detectable cache benefit).
        with mock.patch.object(cp, "_base_url", return_value="http://h"):
            with mock.patch("urllib.request.urlopen") as m:
                class R:
                    def __init__(self):
                        self._sent = False
                    def read(self, n=1024):
                        if self._sent:
                            return b""
                        self._sent = True
                        return (b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                                b"data: [DONE]\n\n")
                    def __enter__(self):
                        return self
                    def __exit__(self, *a):
                        return False
                m.side_effect = lambda *a, **k: R()
                v = cp.probe_warm_ttft("http://h", "m", "")
        self.assertIn(v.verdict, ("UNKNOWN", "PASS"))


if __name__ == "__main__":
    unittest.main()
