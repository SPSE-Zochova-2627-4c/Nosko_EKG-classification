import math

import pytest
import torch

from ekg import gpu_benchmark


@pytest.mark.parametrize("model_name", gpu_benchmark.MODELS)
def test_bench_cpu_runs(model_name):
    r = gpu_benchmark.bench(model_name, torch.device("cpu"), steps=1, warmup=1, batch_size=2)
    assert r.model == model_name
    assert r.device == "cpu"
    assert r.ms_per_step > 0
    assert r.samples_per_s > 0
    assert math.isfinite(r.final_loss)


def test_amp_only_on_gpu():
    with pytest.raises(ValueError):
        gpu_benchmark.bench("classifier", torch.device("cpu"), steps=1, warmup=0, amp=True)


def test_unknown_model():
    with pytest.raises(ValueError):
        gpu_benchmark.bench("lstm", torch.device("cpu"), steps=1, warmup=0, batch_size=2)


def test_environment_info_has_versions():
    info = gpu_benchmark.environment_info()
    for key in ("python", "os", "torch", "hip (ROCm)", "cuda", "zariadenie"):
        assert key in info
    assert info["torch"] == torch.__version__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nie je dostupné")
@pytest.mark.parametrize("model_name", gpu_benchmark.MODELS)
@pytest.mark.parametrize("amp", [False, True])
def test_bench_gpu_runs(model_name, amp):
    # na GPU musí prejsť aj BatchNorm v tréningovom režime (vo WSL bez libc6-dev padal)
    r = gpu_benchmark.bench(
        model_name, torch.device("cuda"), steps=1, warmup=1, batch_size=2, amp=amp
    )
    assert r.device == ("cuda bf16" if amp else "cuda")
    assert math.isfinite(r.final_loss)
