"""The explicit (laptop) sweep loop, with quantize replaced by a copy of the fp32 export."""
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")
pytest.importorskip("onnx")
from PIL import Image

from scripts.prepare_calib import write_manifest

ONNX = Path(__file__).resolve().parents[1] / "models" / "yolov8n_640.onnx"
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="no exported onnx")


def test_explicit_sweep_ranks_blocks_and_resumes(tmp_path, monkeypatch, capsys):
    from scripts import sensitivity
    from tests.test_accuracy import COCO80_TO_91

    calls = []

    def fake_quantize(onnx, out, x, method, nodes_to_exclude=None):
        calls.append((nodes_to_exclude, x.shape[0], method))
        shutil.copy(onnx, out)
        return "fake"

    monkeypatch.setattr(sensitivity, "quantize", fake_quantize)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    shutil.copy(ONNX, tmp_path / "models" / "yolov8n_640.onnx")
    img_dir = tmp_path / "calib" / "val2017" / "images"
    img_dir.mkdir(parents=True)
    names = []
    for i in range(3):
        names.append(f"{i:012d}.jpg")
        Image.new("RGB", (64, 48), (120, 120, 120)).save(img_dir / names[-1])
    write_manifest(tmp_path / "calib" / "val2017" / "manifest.json", names, "synthetic", 0, 640)
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": [{"id": 1, "file_name": names[0], "width": 64, "height": 48}],
                               "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}],
                               "categories": [{"id": c, "name": str(c)} for c in COCO80_TO_91]}))
    argv = ["sens", "--explicit", "--blocks", "model.22,model.0", "--calib-n", "2", "--coco-ann", str(ann), "--coco-images", str(img_dir), "--engine-dir", str(tmp_path / "work")]
    monkeypatch.setattr(sys, "argv", argv)
    sensitivity.main()

    rows = json.loads((tmp_path / "runs" / "sensitivity_explicit.json").read_text())
    assert set(rows) == {"fp32", "int8qdq", "int8qdq_excl-model.22", "int8qdq_excl-model.0"}
    assert calls == [(None, 2, "entropy"), ([r"/model.22/.*"], 2, "entropy"), ([r"/model.0/.*"], 2, "entropy")]
    assert "| block kept float | mAP50-95 | vs int8qdq |" in capsys.readouterr().out
    sensitivity.main()
    assert len(calls) == 3  # every row present, nothing re-quantized
