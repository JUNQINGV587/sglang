"""Compile the prefill-shaped Triton specializations during engine init.

Triton compiles a kernel per specialization and loads that cubin on its first
launch (``CompiledKernel._init_handles`` -> ``cuModuleLoadData``). The load needs
free device memory *outside* the torch caching allocator, and a pooled engine
leaves well under a gigabyte once the weights and the KV/mamba pools are
resident. Kernels that only a long prompt reaches -- the chunked linear-attention
scan, the conv1d stage, the sparse prefill attention and the QSA KV packing --
therefore first load at serving time, which is where a real deployment died:

    Triton kernel 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'
    device-loaded after serving started (free device mem: 0.58 GiB)

Every warmup request the engine runs on its own is short, so it only compiles the
short-prompt specializations. Running one throwaway prefill at
``--chunked-prefill-size`` (plus one two-chunk prefill, which is the only shape
that takes the packed-attention path) compiles the rest while the pools are still
empty.

``SGLANG_PREFILL_WARMUP_TOKENS`` selects the token count; 0 (the default) keeps
the previous lazy behaviour. A failure here is logged and swallowed: the pass is
an optimization, and a model that cannot run a prefill-shaped dummy must still
serve.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import empty_context, log_info_on_rank0

logger = logging.getLogger(__name__)

# The tail chunk of a long prompt is page-aligned and short; only its *presence*
# matters here because the packed path is keyed on "some request has a prefix".
_TAIL_CHUNK_TOKENS = 256


def warmup_prefill_kernels(runner) -> None:
    """Run the dummy prefills described in the module docstring."""
    mr = runner.model_runner
    tokens = envs.SGLANG_PREFILL_WARMUP_TOKENS.get()
    if tokens <= 0 or mr.device != "cuda" or mr.is_draft_worker:
        return

    chunk = min(tokens, get_schedule().max_prefill_tokens or tokens)
    page = max(1, get_schedule().page_size)
    chunk = (chunk // page) * page
    if chunk < page:
        return

    canary_run_ctx = (
        c.with_active_single_forward_manager(0)
        if (c := mr.canary_manager) is not None
        else empty_context()
    )

    log_info_on_rank0(
        logger,
        f"Prefill kernel warmup: one {chunk}-token prefill, plus a two-chunk "
        f"prefill (packed attention path).",
    )
    _run_dummy_prefill(runner, chunk, 0, canary_run_ctx)
    if chunk > _TAIL_CHUNK_TOKENS:
        _run_dummy_prefill(
            runner, _TAIL_CHUNK_TOKENS, chunk - _TAIL_CHUNK_TOKENS, canary_run_ctx
        )


def _run_dummy_prefill(runner, extend_len: int, prefix_len: int, run_ctx) -> None:
    mr = runner.model_runner
    device = mr.device
    total = prefix_len + extend_len
    buffers = None
    row = 0
    saved_row = None
    try:
        buffers = runner._alloc_dummy_decode_buffers(
            1, num_tokens_per_req=extend_len, allocate_logits_buffer=False
        )
        buffers.req_pool_indices[0] = row
        # The dummy request has to look like a real one to the attention backends:
        # the new tokens write scratch slots right after the prefix, and the
        # pool row maps every position of the sequence to a real slot, because
        # the packed path reads the whole row back out of the KV pool.
        buffers.out_cache_loc[:extend_len] = torch.arange(
            prefix_len, total, dtype=buffers.out_cache_loc.dtype, device=device
        )
        buffers.positions[:extend_len] = torch.arange(
            prefix_len, total, dtype=buffers.positions.dtype, device=device
        )

        pool = getattr(mr, "req_to_token_pool", None)
        if prefix_len > 0 and pool is not None:
            saved_row = pool.req_to_token[row, :total].clone()
            pool.req_to_token[row, :total] = torch.arange(
                total, dtype=pool.req_to_token.dtype, device=pool.req_to_token.device
            )

        runner._dummy_run(
            batch_size=1,
            buffers=buffers,
            run_ctx=run_ctx,
            forward_mode_override=ForwardMode.EXTEND,
            extend_num_tokens_per_req=extend_len,
            extend_prefix_len=prefix_len,
        )
        torch.get_device_module(device).synchronize()
    except Exception:
        logger.warning(
            "Prefill kernel warmup at %d tokens (prefix %d) failed; those kernels "
            "will compile lazily at serving time instead.",
            extend_len,
            prefix_len,
            exc_info=True,
        )
    finally:
        if saved_row is not None:
            pool.req_to_token[row, :total] = saved_row
        del buffers
        torch.cuda.empty_cache()
