# No-Nerf Gate — nvfp4_ds_mla port contract (rule-9, default-off)
#
# Purpose: stock vLLM 0.27 `main` lacks the `nvfp4_ds_mla` KV cache dtype that
# prod Anemll (and the fork `stable` branch) carries. This test pins the port
# CONTRACT so the capability cannot silently regress:
#   1. `nvfp4_ds_mla` is a valid kv-cache dtype (config enum + torch dtype map).
#   2. THE Issue-#22 INVARIANT: every fp8 dispatch site routes BOTH fp8_ds_mla
#      AND nvfp4_ds_mla to the fast FP8 kernel path — never a bare
#      `== "fp8_ds_mla"` that would drop nvfp4 to the slow BF16 path (the exact
#      long-context decode regression the original patch fixed).
#      If a bare fp8-only comparison reappears, this test FAILS.
# Rule-9: adding a dtype value is inherently default-off (must be named at
# launch); `auto` and every other dtype path are unchanged when it is absent.
#
# Engine-agnostic / stdlib-only: reads source, imports nothing heavy, so it runs
# in the canonical harness suite without a vLLM build.
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # spark-community/harness -> repo root


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


CACHE_PY = "vllm/config/cache.py"
TORCH_UTILS_PY = "vllm/utils/torch_utils.py"
FLASHMLA_PY = "vllm/v1/attention/backends/mla/flashmla_sparse.py"
ATTN_PY = "vllm/models/deepseek_v4/attention.py"


class Nvfp4DsMlaPortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = _read(CACHE_PY)
        cls.torch_utils = _read(TORCH_UTILS_PY)
        cls.flashmla = _read(FLASHMLA_PY)
        cls.attn = _read(ATTN_PY)

    def test_dtype_is_in_cache_enum(self):
        self.assertIn('"nvfp4_ds_mla"', self.cache)

    def test_dtype_maps_to_uint8(self):
        self.assertIn('"nvfp4_ds_mla": torch.uint8', self.torch_utils)

    def test_backend_supported_list_includes_dtype(self):
        # The FlashMLA sparse backend must advertise support for it. Parse the
        # list body assigned after `supported_kv_cache_dtypes ... = [`.
        body = self.flashmla.split("supported_kv_cache_dtypes", 1)[1]
        start = body.find("= [")
        self.assertGreater(start, -1, "no supported_kv_cache_dtypes list found")
        head = body[start + 3: body.find("]", start)]
        self.assertIn('"nvfp4_ds_mla"', head)
        self.assertIn('"fp8_ds_mla"', head)

    def test_issue22_route_never_drops_nvfp4_to_bf16(self):
        # THE invariant: no bare fp8-only dispatch may remain in the fp8
        # forward-routing of the FlashMLA sparse backend. Every site must route
        # BOTH fp8_ds_mla and nvfp4_ds_mla to the FP8 path.
        # A bare line matching  kv_cache_dtype == "fp8_ds_mla"  (or
        #  cache_dtype == "fp8_ds_mla") would send nvfp4 to _forward_bf16_kv.
        bad = []
        for i, line in enumerate(self.flashmla.splitlines(), 1):
            s = line.strip().rstrip(",")
            if s.endswith('== "fp8_ds_mla"') or s.endswith('not in ("fp8_ds_mla")'):
                bad.append((i, line))
        self.assertEqual(bad, [], f"bare fp8-only dispatch would BF16-drop nvfp4:\n{bad}")

    def test_issue22_forward_dispatch_guards_both(self):
        # The hot forward path must branch on fp8 for BOTH dtypes.
        self.assertIn(
            'use_fp8_cache = self.kv_cache_dtype in ('
            '"fp8_ds_mla", "nvfp4_ds_mla")',
            self.flashmla,
        )

    def test_get_kv_cache_shape_accepts_both(self):
        self.assertIn(
            'if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):',
            self.flashmla,
        )

    def test_dsv4_attention_resolver_accepts_nvfp4(self):
        # Live dual-node crash: assert kv_cache_dtype.startswith("fp8")
        # rejected nvfp4_ds_mla before backend selection finished.
        self.assertIn('kv_cache_dtype == "nvfp4_ds_mla"', self.attn)
        self.assertIn('("fp8_ds_mla", "nvfp4_ds_mla")', self.attn)
        self.assertNotIn(
            'assert kv_cache_dtype.startswith("fp8"),',
            self.attn,
        )


if __name__ == "__main__":
    unittest.main()
