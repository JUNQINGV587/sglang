"""``process_mm_data`` must survive an out-of-memory on the fast image processor.

The fast image processor runs on the serving device but owns no reserved budget:
``--mem-fraction-static`` is sized so the weights and the KV/mamba pools take
almost everything, and a request carrying several images then asks the device to
grow past what is left.  Observed in production on a 2x L20 box: 122 MiB free,
``torch.cat`` of the per-image ``pixel_values`` asked for 124 MiB, the whole
request failed with a 500 -- six times in a row for the same conversation.

Preprocessing is an optimization, not a correctness requirement, so the request
should be retried on the CPU instead of being failed.  Two cases must keep
raising: there is no device to fall back *from*, and the features have to stay on
the device (``cuda_ipc``/``cuda_vmm`` transport), where a CPU result is useless.
"""

import contextlib

import pytest
import torch

from sglang.srt.multimodal.processors import base_processor as bp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _DummyImageProcessor:
    """Stands in for transformers' BaseImageProcessor in the isinstance check."""


class _FakeProcessor:
    def __init__(self, raise_always=False):
        self.image_processor = _DummyImageProcessor()
        self.calls = []
        self.raise_always = raise_always

    def __call__(self, text=None, padding=True, return_tensors="pt", **kwargs):
        device = kwargs.get("device", "<unset>")
        self.calls.append(device)
        if self.raise_always or device not in ("<unset>", "cpu"):
            raise torch.OutOfMemoryError("simulated device oom")
        return {
            "pixel_values": torch.zeros(2, 3),
            "image_grid_thw": torch.zeros(1, 3),
        }


class _Subject(bp.BaseMultimodalProcessor):
    def __init__(self, proc, device, keep_on_device=False):
        # Bypass __init__: it needs server args, a tokenizer and a model config.
        self._proc = proc
        self._device = device
        self.mm_feature_transport = "cuda_ipc" if keep_on_device else "cpu"
        self.FEATURE_NAMES = ["pixel_values"]
        self.image_config = {}
        self.video_config = {}
        self.audio_config = {}
        self.video_preprocessing_device = None
        self._tokenizer_auto_adds_specials = False
        self.precompute_hash_before_cpu_transfer = False
        self.disable_fast_image_processor = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError

    def _resolve_processor(self, processor=None):
        return self._proc, object()

    def _fast_image_processor_device(self, processor):
        return self._device

    @contextlib.contextmanager
    def _temporary_fast_processor_cuda_pool(self, device):
        yield


@pytest.fixture(autouse=True)
def _dummy_image_processor(monkeypatch):
    monkeypatch.setattr(bp, "BaseImageProcessor", _DummyImageProcessor)


def test_device_oom_is_retried_on_cpu():
    proc = _FakeProcessor()
    subject = _Subject(proc, "cuda:0")

    result = subject.process_mm_data("describe these", images=["img0", "img1"])

    assert proc.calls == ["cuda:0", "cpu"], proc.calls
    assert result["pixel_values"].device.type == "cpu"
    assert set(result) == {"pixel_values", "image_grid_thw"}


def test_no_fallback_without_a_device():
    proc = _FakeProcessor(raise_always=True)
    subject = _Subject(proc, None)

    with pytest.raises(torch.OutOfMemoryError):
        subject.process_mm_data("x", images=["img0"])


def test_no_fallback_when_features_must_stay_on_device():
    proc = _FakeProcessor()
    subject = _Subject(proc, "cuda:0", keep_on_device=True)

    with pytest.raises(torch.OutOfMemoryError):
        subject.process_mm_data("x", images=["img0"])
    assert proc.calls == ["cuda:0"]


def test_cpu_path_is_unchanged():
    proc = _FakeProcessor()
    subject = _Subject(proc, "cpu")

    result = subject.process_mm_data("x", images=["img0"])

    assert proc.calls == ["cpu"]
    assert result["pixel_values"].device.type == "cpu"
