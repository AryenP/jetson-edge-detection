from pathlib import Path

import pytest

from scripts.build_engine import block_of, calib_paths

ONNX = Path(__file__).resolve().parents[1] / "models" / "yolov8n_640.onnx"


def test_block_of_depths():
    assert block_of("/model.22/cv2.0/cv2.0.0/conv/Conv") == "model.22"
    assert block_of("/model.22/cv2.0/cv2.0.0/conv/Conv", depth=2) == "model.22/cv2.0"
    assert block_of("/model.0/act/Sigmoid") == "model.0"
    assert block_of("(Unnamed Layer* 3) [Shuffle]") is None


def test_calib_paths_are_per_split():
    p = calib_paths("train2017", "yolov8n_640")
    assert p["manifest"] == Path("calib/train2017/manifest.json")
    assert p["cache"] == Path("calib/train2017/yolov8n_640_entropy2.cache")


@pytest.mark.skipif(not ONNX.exists(), reason="no exported onnx")
def test_every_exported_node_maps_to_a_block():
    onnx = pytest.importorskip("onnx")
    names = [n.name for n in onnx.load(str(ONNX)).graph.node]
    blocks = {block_of(n) for n in names}
    assert None not in blocks
    assert blocks == {f"model.{i}" for i in range(23)}
