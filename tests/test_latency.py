from bench.latency import measure, summarize


def test_summarize_percentiles_and_fps():
    s = summarize([10.0] * 9 + [20.0], n_warmup=5)
    assert s.p50_ms == 10.0 and s.p95_ms > 10.0 and s.n_timed == 10 and s.n_warmup == 5
    assert abs(s.mean_ms - 11.0) < 1e-9 and abs(s.fps - 1000 / 11.0) < 1e-9


def test_measure_discards_warmup():
    calls = []
    s = measure(lambda: calls.append(1), n_warmup=3, n_iters=7)
    assert len(calls) == 10 and s.n_timed == 7 and s.n_warmup == 3 and s.wall_s >= 0


def test_measure_collects_execute_only_series():
    ticks = iter(range(1, 100))
    s = measure(lambda: None, n_warmup=1, n_iters=4, gpu_ms=lambda: float(next(ticks)))
    assert (s.gpu_p50_ms, s.gpu_mean_ms) == (2.5, 2.5) and s.gpu_p95_ms > 3.5
    assert summarize([1.0], 0).gpu_p50_ms is None
