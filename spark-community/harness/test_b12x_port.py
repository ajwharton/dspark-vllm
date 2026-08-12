# No-Nerf Gate — flashinfer_b12x / B12X_MXFP4 port contract (rule-9, default-off)
#
# Purpose: stock vLLM 0.27 `main` accepted `--moe-backend flashinfer_b12x` at
# argparse but the MXFP4 oracle rejected it (falsify 2026-08-12). This test
# pins the oracle wiring that routes the explicit flag to B12xExperts, and
# pins the default-off invariant: B12X must NOT appear in auto priority lists.
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
EXPERT = REPO / "vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py"
QUANT = REPO / "vllm/model_executor/layers/quantization/mxfp4.py"


class TestB12xPortContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = ORACLE.read_text()
        cls.expert = EXPERT.read_text()
        cls.quant = QUANT.read_text()
        cls.oracle_ast = ast.parse(cls.oracle)

    def test_expert_file_exists_and_defines_b12x_experts(self):
        self.assertTrue(EXPERT.is_file(), "b12x_mxfp4_moe.py must exist on main")
        self.assertIn("class B12xExperts", self.expert)
        self.assertIn("fp4_e8m0_k32", self.expert)

    def test_enum_has_b12x(self):
        self.assertIn('B12X_MXFP4 = "B12X_MXFP4"', self.oracle)

    def test_map_flashinfer_b12x_to_enum(self):
        # The falsify's exact failure: map_mxfp4_backend("flashinfer_b12x")
        self.assertIn('"flashinfer_b12x": [Mxfp4MoeBackend.B12X_MXFP4]', self.oracle)

    def test_backend_to_kernel_cls_returns_b12x_experts(self):
        self.assertIn("from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import", self.oracle)
        self.assertIn("return [B12xExperts]", self.oracle)

    def test_not_in_auto_priority_lists(self):
        # Rule-9: default-off. Auto must not pick B12X.
        for fn in ("_get_priority_backends", "_get_priority_backends_for_gpt_oss"):
            node = next(
                n for n in self.oracle_ast.body
                if isinstance(n, ast.FunctionDef) and n.name == fn
            )
            src = ast.get_source_segment(self.oracle, node) or ""
            self.assertNotIn("B12X_MXFP4", src, f"{fn} must not auto-select B12X")

    def test_weight_convert_passthrough_both_paths(self):
        # Both convert_* functions must leave native MXFP4 tensors alone.
        self.assertGreaterEqual(self.oracle.count("if mxfp4_backend == Mxfp4MoeBackend.B12X_MXFP4:"), 2)

    def test_kernel_builder_takes_layer_and_postloads(self):
        self.assertIn("layer: torch.nn.Module | None = None", self.oracle)
        self.assertIn("experts.process_weights_after_loading(layer)", self.oracle)
        self.assertGreaterEqual(self.quant.count("layer=layer,"), 2)


if __name__ == "__main__":
    unittest.main()
