import eujeno.config as cfg
from eujeno.config import auto_device, resolve_device, default_dtype, DEVICE


def test_library_default_stays_cpu():
    # the library default must remain CPU for deterministic golden tests;
    # GPU auto-selection is a CLI-only behaviour.
    assert DEVICE == "cpu"


def test_auto_device_returns_valid():
    assert auto_device() in {"cpu", "cuda", "mps"}


def test_resolve_explicit_is_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_none_and_auto_detect(monkeypatch):
    monkeypatch.setattr(cfg, "auto_device", lambda: "mps")
    assert resolve_device(None) == "mps"
    assert resolve_device("auto") == "mps"
    assert resolve_device("AUTO") == "mps"


def test_default_dtype_per_device():
    assert default_dtype("cpu") == "float32"      # determinism on CPU
    assert default_dtype("mps") == "bfloat16"     # halve memory on GPU
    assert default_dtype("cuda") == "bfloat16"


def test_auto_device_prefers_gpu_then_falls_back(monkeypatch):
    import torch
    # GPU present -> mps wins
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert auto_device() == "mps"
    # no GPU -> cpu
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert auto_device() == "cpu"
