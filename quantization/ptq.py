from __future__ import annotations

import platform
from copy import deepcopy

import torch

from utils.io import get_serialized_model_size_kb


def _pick_engine(preferred: str | None = None) -> str | None:
    """
    Return a valid quantization engine name for the current PyTorch build,
    or None if quantization isn't supported.

    Preference order when `preferred` isn't available:
      * qnnpack on Apple Silicon / ARM
      * fbgemm on x86_64
    """
    supported = list(getattr(torch.backends.quantized, "supported_engines", []) or [])
    if not supported or supported == ["none"]:
        return None
    if preferred and preferred in supported:
        return preferred
    arch = platform.machine().lower()
    if arch in ("arm64", "aarch64") and "qnnpack" in supported:
        return "qnnpack"
    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    # Pick whatever's left — usually "none"; treat as "no engine".
    real = [e for e in supported if e != "none"]
    return real[0] if real else None


def quantize_model_dynamic(model: torch.nn.Module, backend: str | None = None) -> torch.nn.Module:
    engine = _pick_engine(backend)
    if engine is None:
        raise RuntimeError(
            "No quantization engine is registered in this PyTorch build. "
            "Quantization experiments cannot run on this environment — skip pillar 3 "
            "or reinstall PyTorch with qnnpack / fbgemm support."
        )
    torch.backends.quantized.engine = engine
    model = deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_model_static(model: torch.nn.Module, calibration_loader, device: torch.device, backend: str = "fbgemm"):
    engine = _pick_engine(backend)
    if engine is None:
        # Fall back to dynamic quantization; if that also fails it raises loudly.
        return quantize_model_dynamic(model, backend=backend)
    try:
        torch.backends.quantized.engine = engine
        quantized = deepcopy(model).cpu().eval()
        quantized.qconfig = torch.ao.quantization.get_default_qconfig(engine)
        torch.ao.quantization.prepare(quantized, inplace=True)
        with torch.no_grad():
            for images, _ in calibration_loader:
                quantized(images.to(device="cpu"))
        torch.ao.quantization.convert(quantized, inplace=True)
        return quantized
    except Exception:
        return quantize_model_dynamic(model, backend=engine)


def get_quantized_model_size(model: torch.nn.Module) -> float:
    return get_serialized_model_size_kb(model)
