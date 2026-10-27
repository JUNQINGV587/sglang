from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page-row mirror of QSA compressed keys, indexed by full-KV token slots."""

    def __init__(
        self,
        device_pool: QSATokenToKVPool,
        anchor_host,
        *,
        allocator_type: str = "default",
        mtp_draft_device_pools: tuple[QSATokenToKVPool, ...] = (),
    ):
        page_size = anchor_host.page_size
        ratio = device_pool.qsa_compress_ratio
        if page_size % ratio:
            raise ValueError(
                "QSA HiCache pages must be divisible by the compress ratio"
            )
        self.device_pool = device_pool
        pools = (device_pool, *mtp_draft_device_pools)
        buffers = [b for pool in pools for b in pool.qsa_compressed_k_buffer_pool]
        if not buffers:
            raise ValueError("QSA HiCache requires at least one local indexer layer")
        item_bytes = buffers[0][0].nbytes * (page_size // ratio)
        if any(
            pool.qsa_compress_ratio != ratio
            or any(
                b[0].nbytes * (page_size // ratio) != item_bytes
                for b in pool.qsa_compressed_k_buffer_pool
            )
            for pool in pools
        ):
            raise ValueError("Packed QSA indexer pools must share the same page shape")
        # A full-KV page and its compressed groups share ownership. Packing the
        # groups into a byte row lets the existing whole-page kernels move both
        # coordinate spaces using the same full-KV page number.
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=[
                b.view(torch.uint8).reshape(-1, item_bytes) for b in buffers
            ],
            item_bytes=item_bytes,
            num_host_pages=anchor_host.page_num,
            slot_page_size=page_size,
            layout=anchor_host.layout,
            device=anchor_host.device,
            pin_memory=anchor_host.pin_memory,
            allocator_type=allocator_type,
            page_aligned_only=True,
        )
        self.size_per_token = self.get_size_per_token()

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()


class HiSparseQSAFullKVPoolHost(MHATokenToKVPoolHost):
    """HiCache L2 host pool for HiSparse QSA full-attention K/V.

    ``HiSparseQSATokenToKVPool`` already keeps a pinned, write-through mirror of
    every logical slot (``host_key_buffer_pool`` / ``host_value_buffer_pool``,
    row == logical slot id), and the attention kernels read that mirror directly
    for every slot outside the static resident window.  So the device tier has
    no copy of most of the K/V: a plain device<->host HiCache pool cannot work
    here -- ``host_transfer_translate`` would have to map logical ids that do
    not exist in the device buffer.

    This pool therefore performs **host-to-host** transfers only:

    * backup (D->H) copies the logical rows of the QSA mirror into this pool;
    * load   (H->D) copies them back into the QSA mirror, which is exactly the
      copy the kernels read.  The device pool is never touched.

    Everything above this layer (page API, L3 storage backends, eviction) is the
    stock MHA host pool logic.
    """

    #: L2 transfers must not be forwarded to ``device_pool``.
    host_only_transfers = True

    def __init__(self, *args, qsa_pool, **kwargs):
        self._qsa_pool = qsa_pool
        super().__init__(*args, **kwargs)
        layer_num = len(qsa_pool.host_key_buffer_pool)
        if layer_num != self.layer_num:
            raise ValueError(
                "HiSparse QSA host pool geometry mismatch: the QSA mirror has "
                f"{layer_num} layers, the host pool expects {self.layer_num}."
            )

    def _mirror_flat_page(self, start: int) -> torch.Tensor:
        """Flat storage page holding ``page_size`` logical rows of the mirror."""
        stop = start + self.page_size
        k = torch.stack([buf[start:stop] for buf in self._qsa_pool.host_key_buffer_pool])
        v = torch.stack(
            [buf[start:stop] for buf in self._qsa_pool.host_value_buffer_pool]
        )
        return torch.stack((k, v)).flatten()

    def _write_mirror_flat_page(self, start: int, data_page: torch.Tensor) -> None:
        stop = start + self.page_size
        page = data_page.reshape(
            2,
            self.layer_num,
            self.page_size,
            self._qsa_pool.full_kv_pool.head_num,
            self._qsa_pool.full_kv_pool.head_dim,
        )
        for slot in range(self.layer_num):
            self._qsa_pool.host_key_buffer_pool[slot][start:stop] = page[0, slot]
            self._qsa_pool.host_value_buffer_pool[slot][start:stop] = page[1, slot]

    def _page_starts(self, indices: torch.Tensor):
        """First token id of each page-aligned block in a transfer."""
        ids = indices.tolist()
        step = self.page_size
        if len(ids) % step:
            raise ValueError(
                "HiSparse QSA host transfers are page granular; got "
                f"{len(ids)} tokens with page_size={step}."
            )
        return [int(ids[off]) for off in range(0, len(ids), step)]

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        # The mirror always holds the newest K/V (QSA writes through), so a
        # backup only has to move it into this pool's pages.
        for host_row, logical_row in zip(
            self._page_starts(host_indices), self._page_starts(device_indices)
        ):
            self.set_from_flat_data_page(
                host_row, self._mirror_flat_page(logical_row)
            )

    def _reshape_page(self, flat_page, like):
        """A flat (2, L, P, H, D) page view in this pool's K/V layout."""
        return flat_page.reshape(
            2, self.layer_num, self.page_size, like.shape[1], like.shape[2]
        )

    def _resident_window_rows(self, logical_row: int) -> int:
        """Rows of the page at ``logical_row`` that the device tier also mirrors.

        The QSA host tier keeps a static prefix window: logical slots below
        ``hisparse_resident_tokens`` are also readable from the device rows of
        the same index, and the attention kernels prefer them.  A page loaded
        back from the host therefore has to land in both places.
        """
        resident = int(getattr(self._qsa_pool, "hisparse_resident_tokens", 0))
        if resident <= 0:
            return 0
        return max(0, min(self.page_size, resident - logical_row))

    def _local_layer_index(self, layer_id: int) -> int:
        if 0 <= layer_id < self.layer_num:
            return int(layer_id)
        return int(self._qsa_pool._transfer_full_attention_id(layer_id))

    def _write_resident_device_rows(self, layer_id, host_starts, logical_starts):
        local = self._local_layer_index(layer_id)
        full_kv_pool = self._qsa_pool.full_kv_pool
        k_buf = full_kv_pool.k_buffer[local]
        v_buf = full_kv_pool.v_buffer[local]
        page = None
        page_host_row = -1
        for host_row, logical_row in zip(host_starts, logical_starts):
            rows = self._resident_window_rows(logical_row)
            if rows <= 0:
                continue
            if page is None or page_host_row != host_row:
                page = self._reshape_page(self.get_data_page(host_row), k_buf)
                page_host_row = host_row
            k_buf[logical_row : logical_row + rows] = page[0, local, :rows]
            v_buf[logical_row : logical_row + rows] = page[1, local, :rows]

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        # Every page carries all layers, so the mirror is written once (on the
        # first layer) and the remaining layers only fill the device window.
        host_starts = self._page_starts(host_indices)
        logical_starts = self._page_starts(device_indices)
        if layer_id == self.start_layer:
            for host_row, logical_row in zip(host_starts, logical_starts):
                self._write_mirror_flat_page(logical_row, self.get_data_page(host_row))
        # The pinned mirror is authoritative for every logical slot, but the
        # attention kernels read the *device* rows for slots inside the static
        # resident window, and a cold start allocates from slot 0, i.e. entirely
        # inside that window.
        self._write_resident_device_rows(layer_id, host_starts, logical_starts)
