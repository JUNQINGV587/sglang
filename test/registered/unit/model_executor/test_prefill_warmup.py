"""Unit tests for the engine-init prefill kernel warmup.

Two pieces are covered:

* ``sglang.srt.model_executor.runner.prefill_warmup`` -- the dummy prefills that
  compile the dense and the packed-attention prefill specializations while the
  pools are still empty.
* ``sglang.srt.entrypoints.http_server`` -- the request-driven long prefill that
  runs during server warmup, and the prompt it builds for it.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.model_executor.runner import prefill_warmup as pw  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeRunner:
    """Records the dummy forwards instead of running them."""

    def __init__(self, device="cuda", is_draft_worker=False, fail=False, pool=None):
        self.model_runner = SimpleNamespace(
            device=device,
            is_draft_worker=is_draft_worker,
            canary_manager=None,
            req_to_token_pool=pool,
        )
        self.calls = []
        self._fail = fail

    def _alloc_dummy_decode_buffers(
        self, batch_size, *, num_tokens_per_req=1, allocate_logits_buffer=True
    ):
        n = num_tokens_per_req
        return SimpleNamespace(
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=torch.zeros(n, dtype=torch.int64),
            positions=torch.zeros(n, dtype=torch.int64),
        )

    def _dummy_run(
        self,
        batch_size,
        buffers=None,
        run_ctx=None,
        forward_mode_override=None,
        extend_num_tokens_per_req=None,
        extend_prefix_len=0,
    ):
        self.calls.append((extend_num_tokens_per_req, extend_prefix_len))
        if self._fail:
            raise RuntimeError("dummy forward failed")


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(pw, "log_info_on_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )


def _schedule(monkeypatch, *, max_prefill_tokens=12288, page_size=64):
    monkeypatch.setattr(
        pw,
        "get_schedule",
        lambda: SimpleNamespace(
            max_prefill_tokens=max_prefill_tokens, page_size=page_size
        ),
    )


def _record(monkeypatch):
    calls = []

    def fake_run(runner, extend_len, prefix_len, run_ctx):
        calls.append((extend_len, prefix_len))

    monkeypatch.setattr(pw, "_run_dummy_prefill", fake_run)
    return calls


def test_disabled_by_default(monkeypatch, quiet):
    monkeypatch.delenv("SGLANG_PREFILL_WARMUP_TOKENS", raising=False)
    _schedule(monkeypatch)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    assert calls == []


def test_skips_non_cuda_and_draft_workers(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "12288")
    _schedule(monkeypatch)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner(device="cpu"))
    pw.warmup_prefill_kernels(_FakeRunner(is_draft_worker=True))

    assert calls == []


def test_runs_dense_then_packed_prefill(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "12288")
    _schedule(monkeypatch)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    # A full chunk with no prefix (dense path), then a short tail on top of a
    # prefix (the only shape that takes the packed path).
    assert calls == [(12288, 0), (256, 12032)]


def test_chunk_is_clamped_to_max_prefill_tokens(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "32768")
    _schedule(monkeypatch, max_prefill_tokens=12288)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    assert calls == [(12288, 0), (256, 12032)]


def test_chunk_is_page_aligned(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "9999")
    _schedule(monkeypatch, max_prefill_tokens=16384, page_size=64)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    assert calls == [(9984, 0), (256, 9728)]


def test_small_chunk_skips_the_packed_pass(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "128")
    _schedule(monkeypatch)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    # The tail pass would leave no prefix to pack, so it must not run.
    assert calls == [(128, 0)]


def test_no_pass_when_chunk_is_below_a_page(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "32")
    _schedule(monkeypatch, page_size=64)
    calls = _record(monkeypatch)

    pw.warmup_prefill_kernels(_FakeRunner())

    assert calls == []


def test_dummy_prefill_failure_is_swallowed(monkeypatch, quiet):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "12288")
    _schedule(monkeypatch)

    # A model that cannot run a prefill-shaped dummy must still serve.
    pw.warmup_prefill_kernels(_FakeRunner(fail=True))


def test_dummy_prefill_restores_the_pool_row(monkeypatch, quiet):
    pool = SimpleNamespace(
        req_to_token=torch.full((2, 2048), -1, dtype=torch.int64)
    )
    pool.req_to_token[0, :1024] = torch.arange(1024, dtype=torch.int64)
    expected = pool.req_to_token[0, :1024].clone()
    # cpu device: the test has no accelerator, and the buffers only need a
    # device to build their index tensors on.
    runner = _FakeRunner(device="cpu", pool=pool)

    pw._run_dummy_prefill(runner, 256, 768, None)

    assert runner.calls == [(256, 768)]
    assert torch.equal(pool.req_to_token[0, :1024], expected)
    assert torch.equal(pool.req_to_token[1], torch.full((2048,), -1, dtype=torch.int64))


http_server = pytest.importorskip("sglang.srt.entrypoints.http_server")


def test_long_warmup_prompt_length():
    prompt = http_server._long_warmup_prompt(16384)

    words = prompt.split(" ")
    # Ceiling division by the tokens-per-repeat of the word run.
    assert len(words) == -(-16384 // http_server._LONG_PREFILL_TOKENS_PER_REPEAT) * 9


def test_long_warmup_prompt_covers_short_targets():
    assert http_server._long_warmup_prompt(1).split(" ")[0] == (
        http_server._LONG_PREFILL_WORDS.split(" ")[0]
    )


class _Post:
    def __init__(self, calls, prompt_tokens=18210, status=200):
        self.calls = calls
        self.prompt_tokens = prompt_tokens
        self.status_code = status

    def raise_for_status(self):
        assert self.status_code == 200

    def json(self):
        return {"meta_info": {"prompt_tokens": self.prompt_tokens}}


@pytest.fixture
def warmup_env(monkeypatch):
    calls = []
    post = _Post(calls)
    monkeypatch.setattr(http_server, "requests", SimpleNamespace(post=lambda *a, **k: (calls.append((a, k)), post)[1]))
    monkeypatch.setattr(
        http_server,
        "get_serving",
        lambda: SimpleNamespace(skip_tokenizer_init=False, api_key=None),
    )
    monkeypatch.setattr(
        http_server,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="null", language_only=False),
    )
    return calls


def test_long_warmup_is_disabled_by_default(monkeypatch, warmup_env):
    monkeypatch.delenv("SGLANG_PREFILL_WARMUP_TOKENS", raising=False)

    http_server._warmup_long_prefill("http://x", {}, True, {"is_generation": True})

    assert warmup_env == []


def test_long_warmup_posts_one_generate_request(monkeypatch, warmup_env):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "16384")

    http_server._warmup_long_prefill("http://x", {}, True, {"is_generation": True})

    assert len(warmup_env) == 1
    args, kwargs = warmup_env[0]
    assert args[0] == "http://x/generate"
    payload = kwargs["json"]
    assert payload["sampling_params"]["max_new_tokens"] == 1
    assert len(payload["text"].split(" ")) == (
        -(-16384 // http_server._LONG_PREFILL_TOKENS_PER_REPEAT) * 9
    )


@pytest.mark.parametrize(
    "model_info,skip_tokenizer_init",
    [
        ({"is_generation": False}, False),
        ({"is_generation": True}, True),
    ],
)
def test_long_warmup_skips_ineligible_deployments(
    monkeypatch, warmup_env, model_info, skip_tokenizer_init
):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "16384")
    monkeypatch.setattr(
        http_server,
        "get_serving",
        lambda: SimpleNamespace(
            skip_tokenizer_init=skip_tokenizer_init, api_key=None
        ),
    )

    http_server._warmup_long_prefill("http://x", {}, True, model_info)

    assert warmup_env == []


def test_long_warmup_skips_pd_disaggregation(monkeypatch, warmup_env):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "16384")
    monkeypatch.setattr(
        http_server,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="prefill", language_only=False),
    )

    http_server._warmup_long_prefill("http://x", {}, True, {"is_generation": True})

    assert warmup_env == []


def test_long_warmup_failure_does_not_raise(monkeypatch, warmup_env):
    monkeypatch.setenv("SGLANG_PREFILL_WARMUP_TOKENS", "16384")

    def boom(*args, **kwargs):
        raise RuntimeError("warmup endpoint down")

    monkeypatch.setattr(http_server, "requests", SimpleNamespace(post=boom))

    http_server._warmup_long_prefill("http://x", {}, True, {"is_generation": True})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
