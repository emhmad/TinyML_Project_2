from __future__ import annotations

import statistics
import time

import numpy as np
import torch


def measure_latency(model, input_shape=(1, 3, 224, 224), device="cpu", warmup=10, timed_runs=100):
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    sample = torch.randn(*input_shape, device=torch_device)
    times_ms: list[float] = []

    with torch.inference_mode():
        if torch_device.type == "cuda":
            for _ in range(warmup):
                _ = model(sample)
            torch.cuda.synchronize(torch_device)

            for _ in range(timed_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(sample)
                end.record()
                torch.cuda.synchronize(torch_device)
                times_ms.append(float(start.elapsed_time(end)))
        else:
            for _ in range(warmup):
                _ = model(sample)
            for _ in range(timed_runs):
                start = time.perf_counter()
                _ = model(sample)
                end = time.perf_counter()
                times_ms.append((end - start) * 1000.0)

    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "all_times_ms": times_ms,
    }
