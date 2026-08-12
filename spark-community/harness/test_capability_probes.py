import unittest
from unittest import mock

import capability_probes as cp


class CapabilityProbeTest(unittest.TestCase):
    def _mk_metrics(self, accepted, drafted):
        return (f"# HELP among spec_decode accept counters\n"
                f"vllm_spec_decode_num_drafted_tokens {{}} {drafted}\n"
                f"vllm_spec_decode_num_accepted_tokens {{}} {accepted}\n")

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
        ok = {"choices": [{"message": {"content": "nonce-abc"}}]}
        with mock.patch.object(cp, "_post_json", return_value=ok):
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
