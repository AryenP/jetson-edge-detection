import copy
import json

import pytest

from bench import results

GOOD = {
    "run_id": "20261006T120000Z_yolov8n_640_fp16",
    "timestamp": "2026-10-06T12:00:00+00:00",
    "board": "NVIDIA Jetson Orin Nano Developer Kit",
    "jetpack": "6.2",
    "tensorrt": "10.3.0",
    "model": "yolov8n_640",
    "input_res": 640,
    "precision": "fp16",
    "calib_imgs": None,
    "calib_batch_size": None,
    "calib_scheme": None,
    "nvpmodel_mode": "15W (mode 0)",
    "jetson_clocks": True,
    "n_warmup_discarded": 50,
    "latency_ms": {"p50": 8.1, "p95": 8.9},
    "fps": 121.0,
    "map_50_95": 0.371,
    "power_w": {"mean": 7.2, "source": "tegrastats"},
}


def test_good_row_validates():
    results.validate(GOOD)
    results.validate({**GOOD, "meta": {"anything": 1}})


@pytest.mark.parametrize(
    "patch",
    [
        {"precision": "fp32"},
        {"latency_ms": {"p50": 9.0, "p95": 8.0}},
        {"map_50_95": 1.5},
        {"power_w": {"mean": 1.0, "source": "guess"}},
        {"jetson_clocks": "yes"},
        {"nvpmodel_mode": ""},
        {"precision": "int8"},  # int8 without calibration info
        {"bogus": 1},
    ],
)
def test_bad_rows_rejected(patch):
    with pytest.raises(results.SchemaError):
        results.validate({**copy.deepcopy(GOOD), **patch})


def test_missing_field_rejected():
    row = copy.deepcopy(GOOD)
    del row["fps"]
    with pytest.raises(results.SchemaError, match="fps"):
        results.validate(row)


def test_append_and_report(tmp_path):
    path = tmp_path / "results.json"
    results.append(path, GOOD)
    int8 = {**copy.deepcopy(GOOD), "run_id": "r2", "precision": "int8", "calib_imgs": 1000, "calib_batch_size": 8, "calib_scheme": "mixed"}
    results.append(path, int8)
    data = json.loads(path.read_text())
    assert [r["run_id"] for r in data["runs"]] == [GOOD["run_id"], "r2"] and data["schema"] == results.SCHEMA
    with pytest.raises(results.SchemaError, match="already in"):
        results.append(path, GOOD)
    table = results.report(results.load(path))
    assert table.count("\n") == 3 and "1000 img, bs 8, mixed" in table and "| fp16 |" in table
    assert results.report({"runs": []}) == "_no runs yet_"
