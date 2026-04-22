from __future__ import annotations

import torch
from torch import Tensor


def magnitude_score(weight: Tensor) -> Tensor:
    return weight.abs()


def wanda_score(weight: Tensor, activation_norm: Tensor) -> Tensor:
    return weight.abs() * activation_norm.to(weight.device).unsqueeze(0)


def taylor_score(weight: Tensor, gradient: Tensor) -> Tensor:
    return (weight * gradient).abs()


def random_score(weight: Tensor, seed: int = 42) -> Tensor:
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(seed)
    return torch.rand(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)
