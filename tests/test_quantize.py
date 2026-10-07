import json

import numpy as np
import pytest
from PIL import Image

from scripts.quantize_onnx import calib_array


def test_calib_array_follows_manifest_order_and_letterbox(tmp_path):
    names = []
    for i in range(3):
        name = f"{i:012d}.jpg"
        Image.new("RGB", (40 + 10 * i, 30), (255, 0, 0)).save(tmp_path / name)
        names.append(name)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": names}))
    x = calib_array(tmp_path / "manifest.json", tmp_path, 2, 64)
    assert x.shape == (2, 3, 64, 64) and x.dtype == np.float32
    assert x[0, 0].max() > 0.98 and x[0, 1].max() < 0.5  # red image (jpeg-lossy), grey padding


def test_calib_array_refuses_missing_images(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": ["nope.jpg"]}))
    with pytest.raises(FileNotFoundError, match="nope.jpg"):
        calib_array(tmp_path / "manifest.json", tmp_path, 1, 64)
