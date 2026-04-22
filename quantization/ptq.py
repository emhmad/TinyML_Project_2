from __future__ import annotations

from copy import deepcopy

import torch

from utils.io import get_serialized_model_size_kb


def quantize_model_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    model = deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_model_static(model: torch.nn.Module, calibration_loader, device: torch.device, backend: str = "fbgemm"):
    try:
        torch.backends.quantized.engine = backend
        quantized = deepcopy(model).cpu().eval()
        quantized.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(quantized, inplace=True)
        with torch.no_grad():
            for images, _ in calibration_loader:
                quantized(images.to(device="cpu"))
        torch.ao.quantization.convert(quantized, inplace=True)
        return quantized
    except Exception:
        return quantize_model_dynamic(model)


def get_quantized_model_size(model: torch.nn.Module) -> float:
    return get_serialized_model_size_kb(model)
