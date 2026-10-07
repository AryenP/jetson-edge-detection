"""The board-only paths, run against tests/fake_trt."""
import json
import sys

import numpy as np
import pytest
from PIL import Image

from bench import backends
from scripts.build_engine import build, calib_paths, pin_blocks
from scripts.calibrate import entropy_calibrator
from scripts.prepare_calib import write_manifest


def make_calib(tmp_path, n=5, split="val2017"):
    paths = calib_paths(split, "yolov8n_640")
    paths = {k: tmp_path / v for k, v in paths.items()}
    paths["images"].mkdir(parents=True)
    names = []
    for i in range(n):
        name = f"{i:012d}.jpg"
        Image.new("RGB", (64 + 8 * i, 48), (10 * i, 20, 30)).save(paths["images"] / name)
        names.append(name)
    write_manifest(paths["manifest"], names, "synthetic", 0, 640)
    return {**paths, "split": split, "batch": 2}


def test_backend_runs_the_copy_chain(fake_trt, tmp_path):
    engine = tmp_path / "x.engine"
    engine.write_bytes(b"fake-engine")
    be = backends.load(engine)
    assert be.input_shape == (1, 3, 640, 640)
    x = np.full((1, 3, 640, 640), 0.25, np.float32)
    y = be.infer(x)
    assert y.shape == (1, 84, 8400) and y.dtype == np.float32 and np.allclose(y, 0.25)
    assert be.gpu_ms == 1.5
    be.close()
    assert all(m.freed for m in fake_trt.Mem.registry.values())


def test_backend_rejects_foreign_engine(fake_trt, tmp_path):
    engine = tmp_path / "x.engine"
    engine.write_bytes(b"built elsewhere")
    with pytest.raises(RuntimeError, match="deserialize"):
        backends.load(engine)


def test_calibrator_batches_drop_the_remainder_and_cache_round_trips(fake_trt, tmp_path):
    c = make_calib(tmp_path)
    cal = entropy_calibrator(c["manifest"], c["images"], c["cache"], 2, 640)
    assert cal.get_batch_size() == 2 and cal.read_calibration_cache() is None
    ptrs = [cal.get_batch(["images"]) for _ in range(3)]
    assert ptrs[0] == [int(cal.dev)] and ptrs[1] == ptrs[0] and ptrs[2] is None
    assert cal.host.shape == (2, 3, 640, 640) and cal.host.max() <= 1.0
    cal.write_calibration_cache(b"abc")
    assert cal.read_calibration_cache() == b"abc"
    cal.free()
    assert fake_trt.Mem.registry[int(cal.dev)].freed


def test_pin_blocks_skips_int_layers_and_other_blocks(fake_trt):
    net = fake_trt.Network()
    pinned = pin_blocks(net, {"model.22"}, 1)
    assert pinned == ["/model.22/cv2.0/cv2.0.0/conv/Conv", "/model.22/cv3.0/cv3.0.0/conv/Conv", "/model.22/Concat_3"]
    assert net.layers[0].precision is None and net.layers[4].precision is None


def test_int8_build_calibrates_once_then_reuses_cache(fake_trt, tmp_path):
    c = make_calib(tmp_path)
    onnx = tmp_path / "yolov8n_640.onnx"
    onnx.write_bytes(b"onnx")
    out = tmp_path / "engines" / "e.engine"
    side = build(onnx, out, "int8", 256, 640, c, ("model.22",), 1, quiet=True)
    assert out.read_bytes() == b"fake-engine" and c["cache"].read_bytes() == b"fake-cache"
    assert side["calib_imgs"] == 4 and side["calib_split"] == "val2017" and side["pin_fp16"] == ["model.22"] and len(side["pinned_layers"]) == 3
    assert json.loads(out.with_suffix(".engine.json").read_text())["engine_sha256"] == side["engine_sha256"]
    b = fake_trt.Builder.built[-1]
    assert b["flags"] == {"fp16", "int8", "obey"} and b["batches"] == 2 and len(b["pinned"]) == 3
    assert all(m.freed for m in fake_trt.Mem.registry.values())
    # second build must not touch the images at all
    for p in c["images"].iterdir():
        p.unlink()
    build(onnx, out, "int8", 256, 640, c, quiet=True)
    assert fake_trt.Builder.built[-1]["batches"] == 0 and fake_trt.Builder.built[-1]["flags"] == {"fp16", "int8"}


def test_fp16_build_has_no_calibration(fake_trt, tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"onnx")
    side = build(onnx, tmp_path / "m_fp16.engine", "fp16", 256, 640, quiet=True)
    assert side["calib_imgs"] is None and side["pin_fp16"] is None and fake_trt.Builder.built[-1]["flags"] == {"fp16"}


def test_bad_onnx_fails_the_parse(fake_trt, tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="parse failed"):
        build(onnx, tmp_path / "m_fp16.engine", "fp16", 256, 640, quiet=True)


def test_bench_run_records_execute_only_latency(fake_trt, tmp_path, monkeypatch, capsys):
    from bench import run

    engine = tmp_path / "yolov8n_640_fp16.engine"
    engine.write_bytes(b"fake-engine")
    engine.with_suffix(".engine.json").write_text(json.dumps({"precision": "fp16", "imgsz": 640, "onnx": "models/yolov8n_640.onnx", "engine_sha256": "e", "onnx_sha256": "o"}))
    monkeypatch.setattr(sys, "argv", ["run", "--engine", str(engine), "--no-power", "--no-accuracy", "--dry-run", "--nvpmodel", "15W (mode 0)", "--jetson-clocks", "yes", "--warmup", "1", "--iters", "4"])
    run.main()
    out = capsys.readouterr().out
    assert "execute only: p50 1.50 ms" in out
    row = json.loads(out[out.index("{") :])
    assert row["model"] == "yolov8n_640" and row["precision"] == "fp16" and row["jetson_clocks"] is True
    assert row["meta"]["latency"]["gpu_p50_ms"] == 1.5 and row["meta"]["backend"] == "tensorrt" and row["n_warmup_discarded"] == 1


def test_bench_run_refuses_unknown_power_mode(fake_trt, tmp_path, monkeypatch):
    from bench import run

    engine = tmp_path / "e_fp16.engine"
    engine.write_bytes(b"fake-engine")
    monkeypatch.setattr(sys, "argv", ["run", "--engine", str(engine), "--precision", "fp16", "--no-power", "--no-accuracy", "--dry-run"])
    with pytest.raises(SystemExit, match="nvpmodel"):
        run.main()


def test_sensitivity_sweep_is_resumable(fake_trt, tmp_path, monkeypatch, capsys):
    pytest.importorskip("pycocotools")
    from scripts import sensitivity
    from tests.test_accuracy import COCO80_TO_91

    c = make_calib(tmp_path)
    onnx = tmp_path / "yolov8n_640.onnx"
    onnx.write_bytes(b"onnx")
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": [{"id": 1, "file_name": "000000000000.jpg", "width": 64, "height": 48}],
                               "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}],
                               "categories": [{"id": i, "name": str(i)} for i in COCO80_TO_91]}))
    out = tmp_path / "sens.json"
    argv = ["sens", "--onnx", str(onnx), "--blocks", "model.22,model.0", "--coco-ann", str(ann), "--coco-images", str(c["images"]), "--out", str(out), "--engine-dir", str(tmp_path / "eng")]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    sensitivity.main()
    rows = json.loads(out.read_text())
    assert set(rows) == {"fp16", "int8", "int8_fp16-model.22", "int8_fp16-model.0"}
    assert rows["int8_fp16-model.22"]["n_pinned"] == 3 and rows["int8"]["map_50_95"] == 0.0
    assert len(fake_trt.Builder.built) == 4
    table = capsys.readouterr().out
    assert "| model.22 | 3 |" in table
    sensitivity.main()
    assert len(fake_trt.Builder.built) == 4  # every row already present, nothing rebuilt
