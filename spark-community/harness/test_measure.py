import json
import time
import unittest
from unittest import mock

import measure


class FakeURLResponse:
    """Emits a scripted SSE stream as read() would. Supports partial feeds."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._i = 0

    def read(self, n=4096):
        if self._i >= len(self._chunks):
            return b""
        c = self._chunks[self._i]
        self._i += 1
        return c

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _usage_line(n):
    # Real engine emits usage on a trailing data line with empty choices.
    return f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n".encode()


def _content(data):
    return f"data: {json.dumps({'choices': [{'delta': {'content': data}}]})}\n\n".encode()


class MeasureTest(unittest.TestCase):
    def test_valid_long_generation_flagged_ok(self):
        chunks = []
        for i in range(20):
            time.sleep(0.002)
            chunks.append(_content("x"))
        chunks.append(_usage_line(200))
        chunks.append(b"data: [DONE]\n\n")
        with mock.patch("urllib.request.urlopen", return_value=FakeURLResponse(chunks)):
            r = measure.stream_generate("http://h", "m", "p", 256)
        self.assertGreater(r.ttft_s, 0)
        self.assertGreater(r.throughput_tok_s, 0)
        self.assertTrue(r.decode_ok)              # 200 >= MIN_VALID_COMPLETION
        self.assertFalse(r.reasoning_present)
        self.assertEqual(r.completion_tokens, 200)

    def test_short_early_eos_flagged(self):
        chunks = [content := _content("hi there only a few tokens")]
        chunks.extend([_usage_line(12), b"data: [DONE]\n\n"])
        with mock.patch("urllib.request.urlopen", return_value=FakeURLResponse(chunks)):
            r = measure.stream_generate("http://h", "m", "p", 256)
        self.assertFalse(r.decode_ok)             # early EOS must be flagged
        self.assertGreater(r.throughput_tok_s, 0)

    def test_reasoning_detected(self):
        def rc(d):
            return f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': d}}]})}\n\n".encode()
        chunks = [rc("let me think"), _content("a"), _content("b"),
                  _usage_line(200), b"data: [DONE]\n\n"]
        with mock.patch("urllib.request.urlopen", return_value=FakeURLResponse(chunks)):
            r = measure.stream_generate("http://h", "m", "p", 256)
        self.assertTrue(r.reasoning_present)

    def test_run_lower_than_min_raises(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeURLResponse([_usage_line(0), b"data: [DONE]\n\n"])):
            with self.assertRaises(RuntimeError):
                measure.stream_generate("http://h", "m", "p", 256)

    def test_median_run(self):
        a = measure.median_run([
            measure.RunMetrics(1.0, 5.0, True, 256, False, 60.0),
            measure.RunMetrics(2.0, 7.0, True, 256, False, 30.0),
            measure.RunMetrics(3.0, 9.0, False, 30, True, 60.0),
        ])
        self.assertAlmostEqual(a["throughput_median_tok_s"], 7.0)
        self.assertAlmostEqual(a["ttft_median_s"], 2.0)
        self.assertFalse(a["all_decode_ok"])      # one short run flagged
        self.assertTrue(a["reasoning_present_any"])


if __name__ == "__main__":
    unittest.main()
