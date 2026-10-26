"""A chunked admission must be registered as an in-flight chunk.

``PrefillAdder._commit_prefill_admission`` is the single place where the regular
prefill path (``add_one_req`` -> ``_select_prefill_admission``) commits a request.
When the request is longer than ``chunked_prefill_size`` the admission is
truncated and flagged ``is_chunked``; the scheduler then relies on
``new_chunked_reqs`` to (a) schedule the continuation in the next pass and
(b) mark the request as mid-prefill so the sampled token of the truncated chunk
is not committed as the final answer.

4097d169 turned the single ``chunked_req`` slot into the ``new_chunked_reqs``
list and updated the legacy chunked branch, but left this commit path setting
only ``new_chunked_req``.  The registration was therefore silently missing for
every chunked request: prompts longer than ``chunked_prefill_size`` were answered
from their first chunk alone -- empty content, or the tools JSON echoed back when
the truncation landed inside the tool definitions.
"""

import pytest

from sglang.srt.managers.schedule_policy import PrefillAdder, _PrefillAdmission
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeTreeCache:
    def lock(self, node):
        return ("locked", node)


class _FakeAllocator:
    def request_slot_reserve(self, *, has_req_pool_slot, prefill_tokens):
        return 0


class _Adder(PrefillAdder):
    """``PrefillAdder`` without the collaborators the commit path does not need."""

    def __init__(self, page_size: int = 64):
        # Deliberately not calling PrefillAdder.__init__: the regression under
        # test is the registration bookkeeping, not budget arithmetic.
        self.page_size = page_size
        self.can_run_list = []
        self.new_chunked_req = None
        self.new_chunked_reqs = []
        self.tree_cache = _FakeTreeCache()
        self.token_to_kv_pool_allocator = _FakeAllocator()
        self._mamba_slot_cost = 0
        self.budget_updates = []
        self.cache_admissions = []

    def _update_prefill_budget(self, prefix_len, extend_len, max_new_tokens, *_, **__):
        self.budget_updates.append((prefix_len, extend_len, max_new_tokens))

    def _request_slot_reserve_for_req(self, req, extend_input_len=None):
        return 0

    def _mamba_gap_budget_for_req(self, req, *, include_host_hit=False):
        return 0

    def _account_prefill_cache_admission(self, req, prefix_len):
        self.cache_admissions.append((req.rid, prefix_len))


class _Req:
    def __init__(self, rid: str, prefix_len: int = 0):
        self.rid = rid
        self.prefix_len = prefix_len
        self.extend_end = None
        self.last_node = ("node", rid)
        self.retracted_stain = False


def _admission(prefix_len: int, extend_len: int, is_chunked: bool) -> _PrefillAdmission:
    return _PrefillAdmission(
        prefix_len=prefix_len,
        extend_len=extend_len,
        max_new_tokens=0 if is_chunked else 32,
        is_chunked=is_chunked,
        replay_len=0,
    )


def test_chunked_admission_is_registered_as_inflight():
    adder = _Adder()
    req = _Req("req-chunked")

    adder._commit_prefill_admission(req, _admission(0, 12288, True), 0)

    assert req.extend_end == 12288
    assert adder.new_chunked_req is req
    assert adder.new_chunked_reqs == [req], (
        "a truncated prefill must be registered so the scheduler schedules the "
        "continuation and keeps the middle-chunk token out of the answer"
    )
    assert adder.can_run_list == [req]
    assert adder.budget_updates == [(0, 12288, 0)]
    assert adder.cache_admissions == [("req-chunked", 0)]


def test_whole_prompt_admission_is_not_registered():
    adder = _Adder()
    req = _Req("req-whole")

    adder._commit_prefill_admission(req, _admission(0, 12019, False), 0)

    assert req.extend_end == 12019
    assert adder.new_chunked_req is None
    assert adder.new_chunked_reqs == []


def test_continuation_admission_is_registered():
    """The last chunk of a long prompt is still committed through this path."""
    adder = _Adder()
    req = _Req("req-tail", prefix_len=12288)

    adder._commit_prefill_admission(req, _admission(12288, 331, False), 0)

    assert req.extend_end == 12619
    assert adder.new_chunked_reqs == []


def test_registration_appends_every_admission():
    adder = _Adder()
    first, second = _Req("req-a"), _Req("req-b")

    adder._commit_prefill_admission(first, _admission(0, 12288, True), 0)
    adder._commit_prefill_admission(second, _admission(0, 12288, True), 0)

    assert adder.new_chunked_reqs == [first, second]
    assert adder.new_chunked_req is second


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
