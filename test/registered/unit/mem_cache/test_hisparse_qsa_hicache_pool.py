"""HiCache host pool over a HiSparse QSA cache: unit tests.

``HiSparseQSATokenToKVPool`` keeps the full-attention K/V on a pinned mirror
addressed by *logical* slot id, and its device tier only holds the static
resident window (``hisparse_resident_tokens``).  A stock device<->host HiCache
pool would therefore index rows the device buffer does not have, so the QSA
host pool (``HiSparseQSAFullKVPoolHost``) moves data host-to-host: backup copies
the mirror into its own pages, load writes the mirror back and also fills the
device rows that the attention kernels prefer for slots inside the window.

These tests pin that behaviour with fake pools, so no GPU is required.
"""

import unittest

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.mem_cache.pool_host.qsa import (  # noqa: E402
    HiSparseQSAFullKVPoolHost,
)

HEADS = 2
DIM = 3
PAGE = 4
LAYERS = 2


class _FakeFullPool:
    """The paged device tier of a QSA pool (window-sized)."""

    head_num = HEADS
    head_dim = DIM

    def __init__(self, layers: int = LAYERS, rows: int = 16):
        self.k_buffer = [
            torch.zeros(rows, HEADS, DIM, dtype=torch.float32) for _ in range(layers)
        ]
        self.v_buffer = [
            torch.zeros(rows, HEADS, DIM, dtype=torch.float32) for _ in range(layers)
        ]


class _FakeQsaPool:
    """The bits of ``HiSparseQSATokenToKVPool`` the host pool touches."""

    def __init__(
        self,
        *,
        layers: int = LAYERS,
        logical_rows: int = 24,
        resident_tokens: int = 8,
        page_size: int = PAGE,
    ):
        self.page_size = page_size
        self.hisparse_resident_tokens = resident_tokens
        self._logical_size = logical_rows
        self.full_kv_pool = _FakeFullPool(layers)
        self.host_key_buffer_pool = [
            torch.zeros(logical_rows + page_size, HEADS, DIM) for _ in range(layers)
        ]
        self.host_value_buffer_pool = [
            torch.zeros(logical_rows + page_size, HEADS, DIM) for _ in range(layers)
        ]

    def _transfer_full_attention_id(self, layer_id: int) -> int:
        return layer_id


def _make_host(qsa: _FakeQsaPool) -> HiSparseQSAFullKVPoolHost:
    host = object.__new__(HiSparseQSAFullKVPoolHost)
    host._qsa_pool = qsa
    host.page_size = qsa.page_size
    host.layer_num = len(qsa.host_key_buffer_pool)
    host.start_layer = 0
    host.end_layer = host.layer_num
    return host


class TestMirrorPages(unittest.TestCase):
    def test_mirror_page_roundtrip(self):
        qsa = _FakeQsaPool()
        host = _make_host(qsa)
        for layer in range(LAYERS):
            qsa.host_key_buffer_pool[layer][PAGE : 2 * PAGE] = float(layer + 1)
            qsa.host_value_buffer_pool[layer][PAGE : 2 * PAGE] = float(layer + 11)

        flat = host._mirror_flat_page(PAGE)
        self.assertEqual(flat.numel(), 2 * LAYERS * PAGE * HEADS * DIM)

        for layer in range(LAYERS):
            qsa.host_key_buffer_pool[layer].zero_()
            qsa.host_value_buffer_pool[layer].zero_()
        host._write_mirror_flat_page(PAGE, flat)

        for layer in range(LAYERS):
            self.assertTrue(
                torch.all(qsa.host_key_buffer_pool[layer][PAGE : 2 * PAGE] == layer + 1)
            )
            self.assertTrue(
                torch.all(
                    qsa.host_value_buffer_pool[layer][PAGE : 2 * PAGE] == layer + 11
                )
            )

    def test_page_starts_are_page_granular(self):
        host = _make_host(_FakeQsaPool())
        self.assertEqual(host._page_starts(torch.arange(2 * PAGE)), [0, PAGE])
        with self.assertRaises(ValueError):
            host._page_starts(torch.arange(PAGE + 2))


class TestResidentWindow(unittest.TestCase):
    def test_rows_inside_window(self):
        host = _make_host(_FakeQsaPool(resident_tokens=2 * PAGE))
        self.assertEqual(host._resident_window_rows(0), PAGE)
        self.assertEqual(host._resident_window_rows(PAGE), PAGE)
        self.assertEqual(host._resident_window_rows(2 * PAGE), 0)

    def test_page_straddling_the_window_edge(self):
        host = _make_host(_FakeQsaPool(resident_tokens=PAGE + 2))
        self.assertEqual(host._resident_window_rows(0), PAGE)
        self.assertEqual(host._resident_window_rows(PAGE), 2)
        self.assertEqual(host._resident_window_rows(2 * PAGE), 0)

    def test_no_window_means_no_device_rows(self):
        host = _make_host(_FakeQsaPool(resident_tokens=0))
        self.assertEqual(host._resident_window_rows(0), 0)


class TestLoadAndBackup(unittest.TestCase):
    def _host_with_page(self, qsa, page_value: float):
        host = _make_host(qsa)
        flat = torch.full(
            (2 * LAYERS * PAGE * HEADS * DIM,), page_value, dtype=torch.float32
        )
        host.get_data_page = lambda host_row: flat.clone()  # noqa: ARG005
        return host

    def test_load_writes_mirror_once_and_fills_the_device_window(self):
        qsa = _FakeQsaPool(resident_tokens=2 * PAGE)
        host = self._host_with_page(qsa, 7.0)
        host_indices = torch.tensor([0, 1, 2, 3])
        device_indices = torch.tensor([PAGE, PAGE + 1, PAGE + 2, PAGE + 3])

        host.load_to_device_per_layer_physical(
            qsa.full_kv_pool, host_indices, device_indices, 0, "kernel"
        )

        # Inside the window: the device rows the kernels read are filled.
        self.assertTrue(torch.all(qsa.full_kv_pool.k_buffer[0][PAGE : 2 * PAGE] == 7.0))
        self.assertTrue(
            torch.all(qsa.full_kv_pool.v_buffer[0][PAGE : 2 * PAGE] == 7.0)
        )
        # Every page carries all layers, so the mirror is written on layer 0.
        self.assertTrue(torch.all(qsa.host_key_buffer_pool[0][PAGE : 2 * PAGE] == 7.0))
        self.assertTrue(
            torch.all(qsa.host_key_buffer_pool[1][PAGE : 2 * PAGE] == 7.0)
        )

    def test_load_outside_the_window_leaves_the_device_tier_alone(self):
        qsa = _FakeQsaPool(resident_tokens=PAGE)
        host = self._host_with_page(qsa, 9.0)
        host_indices = torch.tensor([0, 1, 2, 3])
        device_indices = torch.tensor([2 * PAGE + i for i in range(PAGE)])

        host.load_to_device_per_layer_physical(
            qsa.full_kv_pool, host_indices, device_indices, 0, "kernel"
        )

        self.assertTrue(torch.count_nonzero(qsa.full_kv_pool.k_buffer[0]) == 0)
        self.assertTrue(
            torch.all(
                qsa.host_key_buffer_pool[0][2 * PAGE : 3 * PAGE] == 9.0
            )
        )

    def test_backup_copies_the_mirror_into_pool_pages(self):
        qsa = _FakeQsaPool()
        for layer in range(LAYERS):
            qsa.host_key_buffer_pool[layer][PAGE : 2 * PAGE] = 3.0
            qsa.host_value_buffer_pool[layer][PAGE : 2 * PAGE] = 5.0
        host = _make_host(qsa)
        written = {}
        host.set_from_flat_data_page = (  # noqa: ARG002
            lambda host_row, data: written.update({host_row: data.clone()})
        )

        host.backup_from_device_all_layer_physical(
            qsa.full_kv_pool,
            torch.tensor([2 * PAGE + i for i in range(PAGE)]),
            torch.tensor([PAGE + i for i in range(PAGE)]),
            "kernel",
        )

        self.assertEqual(list(written), [2 * PAGE])
        page = written[2 * PAGE].reshape(2, LAYERS, PAGE, HEADS, DIM)
        self.assertTrue(torch.all(page[0, 0] == 3.0))
        self.assertTrue(torch.all(page[1, 1] == 5.0))


class TestAssemblerGate(unittest.TestCase):
    """``build_kv_host_pool`` must not silently build a device<->host pool."""

    def _build(self, *, hisparse_env: bool):
        from types import SimpleNamespace
        from unittest import mock

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm

        mirror_owner = SimpleNamespace(_logical_size=8 * PAGE)
        kv_pool = SimpleNamespace(
            is_hisparse_qsa=True,
            qsa_mirror_owner=mirror_owner,
        )
        captured = {}

        class _Sentinel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.object(envs.SGLANG_QSA_HISPARSE_HICACHE, "get",
                               return_value=hisparse_env), \
             mock.patch.object(asm, "_mha_host_inputs",
                               return_value=(object, object(), {"start_layer": 0})), \
             mock.patch(
                 "sglang.srt.mem_cache.pool_host.qsa.HiSparseQSAFullKVPoolHost",
                 _Sentinel,
             ):
            try:
                asm.build_kv_host_pool(kv_pool=kv_pool, page_size=PAGE)
            except NotImplementedError:
                raise
            except Exception:
                pass  # later wiring needs a real pool; the gate already ran
        return captured

    def test_gate_is_closed_by_default(self):
        with self.assertRaises(NotImplementedError):
            self._build(hisparse_env=False)

    def test_gate_opens_when_enabled(self):
        # The wiring needs a real pool to build, but the gate itself must let the
        # run through once SGLANG_QSA_HISPARSE_HICACHE is set.
        try:
            self._build(hisparse_env=True)
        except NotImplementedError:  # pragma: no cover - the failure we guard on
            self.fail("the HiCache gate stayed closed with the env switch on")


if __name__ == "__main__":
    unittest.main()
