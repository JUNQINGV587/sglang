"""HiSparse + radix prefix cache on Qwen4-Exp QSA: argument-gate regression test.

``--enable-hisparse`` historically required ``--disable-radix-cache`` for every
model (``hisparse_hook.validate_hisparse``).  For QSA that requirement is not
structural: the QSA host tier is a pure address-translation mask, every logical
slot stays readable, and the radix tree never touches K/V buffers, so the two
can run together behind ``SGLANG_QSA_HISPARSE_RADIX=1``.

These tests pin the gate: off by default (the assertion still fires), open for
QSA when the env switch is set, and *never* open for the other HiSparse models.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _View:
    """Stand-in for the resolving view hisparse_hook reads through."""

    def __init__(self, *, disable_radix_cache: bool) -> None:
        self.enable_hisparse = True
        self.disable_radix_cache = disable_radix_cache
        self.enable_mixed_chunk = False
        self.kv_cache_dtype = "bfloat16"
        self.speculative_algorithm = None
        self.speculative_adaptive = False
        self.speculative_eagle_topk = None
        self.dsa_prefill_backend = None
        self.dsa_decode_backend = None


class TestHisparseQsaRadixGate(CustomTestCase):
    def _run(self, *, is_qsa: bool, disable_radix_cache: bool, env: str | None):
        from sglang.srt.arg_groups import hisparse_hook as hook
        from sglang.srt.configs import model_config as mc
        from sglang.srt.layers.attention.qsa import config as qsa_config

        view = _View(disable_radix_cache=disable_radix_cache)
        server_args = SimpleNamespace(enable_hisparse=True)

        orig = {
            "resolving_view": hook.resolving_view,
            "resolved_view": hook.resolved_view,
            "model_config_of": hook.model_config_of,
            "get_platform": hook.get_platform,
            "feature_combos": hook.validate_hisparse_feature_combos,
            "dsa": mc.is_deepseek_dsa,
            "v4": mc.is_deepseek_v4,
            "m3": mc.is_minimax_sparse,
            "qsa": qsa_config.is_qwen_qsa,
        }
        import os

        orig_env = os.environ.get("SGLANG_QSA_HISPARSE_RADIX")
        try:
            hook.resolving_view = lambda _sa: view
            hook.resolved_view = lambda _sa: view
            hook.model_config_of = lambda _sa: SimpleNamespace(hf_config=object())
            hook.get_platform = lambda: SimpleNamespace(is_hip=False)
            hook.validate_hisparse_feature_combos = lambda _sa: None
            # Non-QSA cases stand in for DSA, the other family that accepts
            # --enable-hisparse, so the model-class assertion stays quiet.
            mc.is_deepseek_dsa = lambda _cfg: not is_qsa
            mc.is_deepseek_v4 = lambda _cfg: False
            mc.is_minimax_sparse = lambda _cfg: False
            qsa_config.is_qwen_qsa = lambda _cfg: is_qsa
            if env is None:
                os.environ.pop("SGLANG_QSA_HISPARSE_RADIX", None)
            else:
                os.environ["SGLANG_QSA_HISPARSE_RADIX"] = env
            hook.validate_hisparse(server_args)
        finally:
            hook.resolving_view = orig["resolving_view"]
            hook.resolved_view = orig["resolved_view"]
            hook.model_config_of = orig["model_config_of"]
            hook.get_platform = orig["get_platform"]
            hook.validate_hisparse_feature_combos = orig["feature_combos"]
            mc.is_deepseek_dsa = orig["dsa"]
            mc.is_deepseek_v4 = orig["v4"]
            mc.is_minimax_sparse = orig["m3"]
            qsa_config.is_qwen_qsa = orig["qsa"]
            if orig_env is None:
                os.environ.pop("SGLANG_QSA_HISPARSE_RADIX", None)
            else:
                os.environ["SGLANG_QSA_HISPARSE_RADIX"] = orig_env

    def test_qsa_with_radix_is_rejected_by_default(self):
        with self.assertRaises(AssertionError) as ctx:
            self._run(is_qsa=True, disable_radix_cache=False, env=None)
        self.assertIn("--disable-radix-cache", str(ctx.exception))

    def test_qsa_with_radix_is_allowed_when_opted_in(self):
        self._run(is_qsa=True, disable_radix_cache=False, env="1")

    def test_qsa_still_accepts_the_documented_configuration(self):
        self._run(is_qsa=True, disable_radix_cache=True, env=None)

    def test_non_qsa_hisparse_models_still_require_disable_radix_cache(self):
        with self.assertRaises(AssertionError):
            self._run(is_qsa=False, disable_radix_cache=False, env="1")

    def test_env_switch_does_not_open_other_models(self):
        # env set, QSA off, and the plain path keeps the flag: fine.
        self._run(is_qsa=False, disable_radix_cache=True, env="1")


if __name__ == "__main__":
    unittest.main()
