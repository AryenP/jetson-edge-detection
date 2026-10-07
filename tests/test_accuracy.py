"""End-to-end mAP on a synthetic 2-image COCO set with a fake backend that returns perfect boxes."""
import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")
from PIL import Image

from bench.accuracy import COCO80_TO_91, evaluate
from bench.preprocess import letterbox


class PerfectBackend:
    """Emits the ground-truth boxes (in letterboxed coords) for whichever image is being run."""

    name = "fake"
    input_shape = (1, 3, 640, 640)

    def __init__(self, gts):
        self.gts = gts  # list of (meta, [(x, y, w, h, cls)]) in call order
        self.calls = 0

    def infer(self, x):
        meta, boxes = self.gts[self.calls]
        self.calls += 1
        out = np.zeros((1, 84, 8400), np.float32)
        for i, (x0, y0, w, h, c) in enumerate(boxes):
            cx, cy = (x0 + w / 2) * meta.scale + meta.pad_x, (y0 + h / 2) * meta.scale + meta.pad_y
            out[0, :4, i] = (cx, cy, w * meta.scale, h * meta.scale)
            out[0, 4 + c, i] = 0.9
        return out

    def close(self):
        pass


def test_category_map_is_coco_80_to_91():
    assert len(set(COCO80_TO_91)) == 80 and COCO80_TO_91[0] == 1 and COCO80_TO_91[79] == 90 and 12 not in COCO80_TO_91


def test_perfect_detector_scores_map_one(tmp_path):
    sizes = [(400, 300), (640, 480)]
    boxes = [[(10, 20, 100, 80, 0), (200, 100, 50, 60, 2)], [(50, 50, 300, 200, 7)]]
    images, anns, gts, aid = [], [], [], 1
    for i, ((w, h), bxs) in enumerate(zip(sizes, boxes), start=1):
        name = f"{i:012d}.jpg"
        Image.new("RGB", (w, h), (90, 120, 150)).save(tmp_path / name)
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for x, y, bw, bh, c in bxs:
            anns.append({"id": aid, "image_id": i, "category_id": COCO80_TO_91[c], "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
        gts.append((letterbox(np.zeros((h, w, 3), np.uint8), 640)[1], bxs))
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in COCO80_TO_91]}))

    res = evaluate(PerfectBackend(gts), ann_file, tmp_path, 640, dets_out=tmp_path / "det.json")
    assert res.n_images == 2 and res.n_dets == 3
    assert res.map_50_95 > 0.99 and res.map_50 > 0.99
    assert (tmp_path / "det.json").exists()
