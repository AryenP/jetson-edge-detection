from bench.latency import measure, summarize


def test_summarize_percentiles_and_fps():
    s = summarize([10.0] * 9 + [20.0], n_warmup=5)
    assert s.p50_ms == 10.0 and s.p95_ms > 10.0 and s.n_timed == 10 and s.n_warmup == 5
    assert abs(s.mean_ms - 11.0) < 1e-9 and abs(s.fps - 1000 / 11.0) < 1e-9


def test_measure_discards_warmup():
    calls = []
    s = measure(lambda: calls.append(1), n_warmup=3, n_iters=7)
    assert len(calls) == 10 and s.n_timed == 7 and s.n_warmup == 3 and s.wall_s >= 0
