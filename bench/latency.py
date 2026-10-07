import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n_timed: int
    n_warmup: int
    fps: float
    wall_s: float

    def as_dict(self):
        return asdict(self)


def summarize(samples_ms, n_warmup, wall_s=None):
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no timed samples")
    mean = float(arr.mean())
    # fps from the mean, not p50: frames finished per second of the loop is a throughput
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=mean,
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        n_timed=int(arr.size),
        n_warmup=n_warmup,
        fps=1000.0 / mean,
        wall_s=float(wall_s if wall_s is not None else arr.sum() / 1000),
    )


def measure(step, n_warmup=50, n_iters=500):
    """`step` must block until the output is on the host, or this times the launch."""
    for _ in range(n_warmup):
        step()
    samples = np.empty(n_iters)
    t_start = time.perf_counter()
    for i in range(n_iters):
        t0 = time.perf_counter()
        step()
        samples[i] = (time.perf_counter() - t0) * 1000
    return summarize(samples, n_warmup, time.perf_counter() - t_start)
