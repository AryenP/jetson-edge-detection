"""Real-model smoke test: runs only when an exported ONNX file is present."""
from pathlib import Path

import numpy as np
import pytest

ONNX = Path(__file__).resolve().parents[1] / "models" / "yolov8n_640.onnx"
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="models/yolov8n_640.onnx not exported")


def test_onnx_output_layout_and_decode():
    from bench import backends
    from bench.postprocess import decode

    be = backends.load(ONNX)
    assert be.input_shape == (1, 3, 640, 640)
    y = be.infer(np.random.rand(1, 3, 640, 640).astype(np.float32))
    assert y.shape == (1, 84, 8400)
    assert 0.0 <= y[0, 4:].max() <= 1.0  # class scores are sigmoided
    assert len(decode(y, conf_thres=0.25)) <= 300
