"""Temporary CUDA-graph capture diagnostics for the QSA HiSparse host tier.

Enabled only when ``SGLANG_QSA_CFX_DEBUG`` is set to a non-empty, non-zero
value, so the default path is byte-for-byte the old one (no prints, no
allocs).  Added 2026-10-23 to localise ``Xid 31 FAULT_PDE`` seen during decode
graph capture; safe to delete once the capture path is proven.
"""

import os
import sys

_ON = os.environ.get("SGLANG_QSA_CFX_DEBUG", "") not in ("", "0")
_printed = set()


def enabled() -> bool:
    return _ON


def flag(name: str) -> bool:
    """Independent experiment switch, e.g. SGLANG_QSA_CFX_NO_HOST=1."""
    return os.environ.get(name, "") not in ("", "0")


def mark(where: str, extra: str = "") -> None:
    if not _ON:
        return
    print(f"[CFX] {where}{(' ' + extra) if extra else ''}", file=sys.stderr, flush=True)


def once(key: str) -> bool:
    """True exactly once per process for ``key``."""
    if key in _printed:
        return False
    _printed.add(key)
    return True


def dump_pool(pool, where: str) -> None:
    """Log the host-tier geometry the capture path is about to hand to kernels."""
    if not _ON or not once("pool"):
        return
    try:
        host_k = getattr(pool, "host_key_buffer_pool", None)
        host_v = getattr(pool, "host_value_buffer_pool", None)
        lines = [
            f"[CFX] pool {where} type={type(pool).__name__}",
            f"[CFX]   size={getattr(pool, 'size', None)}"
            f" logical={getattr(pool, '_logical_size', None)}"
            f" page={getattr(pool, 'page_size', None)}"
            f" resident={getattr(pool, 'hisparse_resident_tokens', None)}"
            f" ratio={getattr(pool, 'host_to_device_ratio', None)}"
            f" post_capture={getattr(pool, 'post_capture_active', None)}",
        ]
        full = getattr(pool, "full_kv_pool", None)
        if full is not None:
            kb = getattr(full, "k_buffer", None)
            rows = None
            try:
                rows = kb[0].shape[0] if kb is not None else None
            except Exception:  # noqa: BLE001
                pass
            lines.append(
                f"[CFX]   full_pool={type(full).__name__} device_rows={rows}"
                f" head_num={getattr(full, 'head_num', None)}"
                f" head_dim={getattr(full, 'head_dim', None)}"
                f" dtype={getattr(full, 'dtype', None)}"
            )
        for tag, bufs in (("host_k", host_k), ("host_v", host_v)):
            if not bufs:
                continue
            t = bufs[0]
            lo = t.data_ptr()
            hi = lo + t.numel() * t.element_size()
            lines.append(
                f"[CFX]   {tag} n_layers={len(bufs)} rows={t.shape[0]}"
                f" heads={t.shape[1]} dim={t.shape[2]} elem={t.element_size()}"
                f" pinned={t.is_pinned()} va=[0x{lo:x},0x{hi:x})"
                f" total={len(bufs) * t.numel() * t.element_size() / 2**20:.1f}MiB"
            )
        lines.append(f"[CFX]   torch_alloc={_cuda_alloc_summary()}")
        print("\n".join(lines), file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[CFX] dump_pool failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _cuda_alloc_summary() -> str:
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return (
            f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB"
            f" reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB"
        )
    except Exception as exc:  # noqa: BLE001
        return f"<{exc}>"


def dump_batch(forward_batch, where: str) -> None:
    """Log the capture-time index buffers feeding the KV write/read path."""
    if not _ON or not once(f"batch:{where}"):
        return
    try:
        parts = [f"[CFX] batch {where} bs={forward_batch.batch_size}"]
        for name in ("out_cache_loc", "req_pool_indices", "positions", "seq_lens"):
            t = getattr(forward_batch, name, None)
            if t is None or not hasattr(t, "numel"):
                continue
            try:
                mn = int(t.min())
                mx = int(t.max())
                parts.append(f"{name}[{t.numel()}] min={mn} max={mx}")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"{name}<{exc}>")
        meta = getattr(forward_batch, "attn_metadata_capture", None)
        del meta
        print(" ".join(parts), file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[CFX] dump_batch failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
