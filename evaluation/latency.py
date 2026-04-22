"""
Latency benchmarking for compressed models (W9).

Three runtime targets:
  - Native PyTorch on CPU or CUDA, deterministic scheduling
  - ONNX Runtime (CPU provider by default; CoreML / CUDA providers if
    installed on the target machine)
  - TensorFlow Lite (built from the exported ONNX via
    `onnx_tf` / `tf` if available; guarded with a clear error otherwise)

Every target returns the same dict so the CSV schema is uniform:
  {mean_ms, median_ms, p95_ms, p99_ms, std_ms, n, runtime, provider}

The benchmarking loop stabilises thermal state by running a
`warmup_seconds` soft warmup in addition to the fixed `warmup` runs;
this matters on Raspberry Pi / phone hardware where the first burst is
DVFS-throttled. On CUDA we use `torch.cuda.Event` for wallclock-safe
timing; elsewhere `time.perf_counter`.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass
class LatencyResult:
    runtime: str
    provider: str
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float
    n: int
    all_times_ms: list[float]

    def as_row(self) -> dict:
        return {
            "runtime": self.runtime,
            "provider": self.provider,
            "mean_ms": float(self.mean_ms),
            "median_ms": float(self.median_ms),
            "std_ms": float(self.std_ms),
            "p95_ms": float(self.p95_ms),
            "p99_ms": float(self.p99_ms),
            "n": int(self.n),
        }


def _summarise(times_ms: Sequence[float], runtime: str, provider: str) -> LatencyResult:
    if not times_ms:
        return LatencyResult(
            runtime=runtime, provider=provider,
            mean_ms=float("nan"), median_ms=float("nan"),
            std_ms=float("nan"), p95_ms=float("nan"), p99_ms=float("nan"),
            n=0, all_times_ms=[],
        )
    arr = np.asarray(times_ms, dtype=np.float64)
    return LatencyResult(
        runtime=runtime,
        provider=provider,
        mean_ms=float(arr.mean()),
        median_ms=float(statistics.median(arr.tolist())),
        std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        n=int(arr.size),
        all_times_ms=arr.tolist(),
    )


def _pin_cpu_threads(num_threads: int | None) -> None:
    """
    On edge targets (RPi, phone), latency variance is dominated by
    scheduler noise. Pin PyTorch / ORT to a fixed thread count so the
    reported numbers are interpretable.
    """
    if num_threads is None or num_threads <= 0:
        return
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(int(num_threads))
    except RuntimeError:
        # interop threads can only be set before forward dispatch; ignore if already set.
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(int(num_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(int(num_threads)))


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str | torch.device = "cpu",
    warmup: int = 10,
    timed_runs: int = 100,
    warmup_seconds: float = 0.0,
    num_threads: int | None = None,
) -> dict:
    """
    PyTorch-native latency. Returns the legacy dict shape
    {median_ms, mean_ms, std_ms, p95_ms, all_times_ms} plus p99_ms and
    a runtime/provider tag, so existing callers keep working.
    """
    torch_device = torch.device(device) if not isinstance(device, torch.device) else device
    _pin_cpu_threads(num_threads if torch_device.type == "cpu" else None)
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
            if warmup_seconds > 0:
                deadline = time.perf_counter() + warmup_seconds
                while time.perf_counter() < deadline:
                    _ = model(sample)
            for _ in range(timed_runs):
                start = time.perf_counter()
                _ = model(sample)
                end = time.perf_counter()
                times_ms.append((end - start) * 1000.0)

    summary = _summarise(times_ms, runtime="torch", provider=torch_device.type)
    return {
        "median_ms": summary.median_ms,
        "mean_ms": summary.mean_ms,
        "std_ms": summary.std_ms,
        "p95_ms": summary.p95_ms,
        "p99_ms": summary.p99_ms,
        "n": summary.n,
        "runtime": summary.runtime,
        "provider": summary.provider,
        "all_times_ms": summary.all_times_ms,
    }


def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """
    Export a PyTorch model to ONNX for edge-runtime benchmarking.
    Uses dynamic batch axis by default so the same ONNX graph can be
    re-benchmarked across batch sizes without re-exporting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*input_shape)
    model_cpu = model.eval().to("cpu")
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model_cpu,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return output_path


def measure_latency_onnx(
    onnx_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    providers: Iterable[str] | None = None,
    warmup: int = 10,
    timed_runs: int = 100,
    warmup_seconds: float = 0.0,
    num_threads: int | None = None,
) -> LatencyResult:
    """
    ONNX Runtime benchmark. `providers` is the execution-provider list
    passed to `InferenceSession`. Default: CPUExecutionProvider — the
    one edge target we can count on. Callers running on a jetson or
    phone can pass ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    or the TensorRT provider and the function will fall through.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed. `pip install onnxruntime` (CPU) "
            "or `onnxruntime-gpu` on the target machine."
        ) from exc

    providers = list(providers or ["CPUExecutionProvider"])
    session_options = ort.SessionOptions()
    if num_threads is not None:
        session_options.intra_op_num_threads = int(num_threads)
        session_options.inter_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
    input_name = session.get_inputs()[0].name
    sample = np.random.rand(*input_shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: sample})
    if warmup_seconds > 0:
        deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < deadline:
            session.run(None, {input_name: sample})
    times_ms: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    provider_name = session.get_providers()[0] if session.get_providers() else "unknown"
    return _summarise(times_ms, runtime="onnxruntime", provider=provider_name)


def measure_latency_tflite(
    tflite_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup: int = 10,
    timed_runs: int = 100,
    warmup_seconds: float = 0.0,
    num_threads: int | None = None,
) -> LatencyResult:
    """
    TensorFlow Lite benchmark. Requires `tflite-runtime` (the minimal
    on-device package) or the full `tensorflow` install. The caller is
    responsible for producing the `.tflite` file from the exported
    ONNX — see `scripts/export_tflite.py` for the one-shot converter.
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "tflite_runtime or tensorflow is required for TFLite benchmarking. "
                "Install one of them on the target device."
            ) from exc

    interpreter = Interpreter(model_path=str(tflite_path), num_threads=num_threads or 1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != tuple(input_shape):
        interpreter.resize_tensor_input(input_detail["index"], list(input_shape))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
    sample = np.random.rand(*input_shape).astype(np.float32)

    for _ in range(warmup):
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
    if warmup_seconds > 0:
        deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < deadline:
            interpreter.set_tensor(input_detail["index"], sample)
            interpreter.invoke()
    times_ms: list[float] = []
    for _ in range(timed_runs):
        interpreter.set_tensor(input_detail["index"], sample)
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        interpreter.get_tensor(output_detail["index"])
        times_ms.append((end - start) * 1000.0)

    return _summarise(times_ms, runtime="tflite", provider="xnnpack")
