import json
import time
from dataclasses import dataclass
from pathlib import Path

from .postprocess import MAX_DET, VAL_CONF, VAL_IOU, decode
from .preprocess import preprocess, scale_boxes

# model class index -> COCO category id; the 91-id space has gaps (no 12, 26, 29, ...)
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


@dataclass
class Accuracy:
    map_50_95: float
    map_50: float
    n_images: int
    n_dets: int
    conf_thres: float
    iou_thres: float
    max_det: int
    seconds: float


def eval_images(ann_file, images_dir, limit=None):
    """(image_id, path) for annotated images present on disk, so a partial download still runs."""
    with open(ann_file) as f:
        ann = json.load(f)
    images_dir = Path(images_dir)
    found = []
    for im in sorted(ann["images"], key=lambda i: i["id"]):
        p = images_dir / im["file_name"]
        if p.exists():
            found.append((int(im["id"]), p))
    return found[:limit] if limit else found


def detect_all(backend, images, imgsz, conf_thres, iou_thres, max_det):
    dets = []
    for image_id, path in images:
        x, meta = preprocess(path, imgsz)
        d = decode(backend.infer(x), conf_thres, iou_thres, max_det)
        for box, score, cls in zip(scale_boxes(d.boxes, meta), d.scores, d.classes):
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append({"image_id": image_id, "category_id": COCO80_TO_91[int(cls)], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return dets


def coco_map(ann_file, dets, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dets:
        return 0.0, 0.0
    gt = COCO(str(ann_file))
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def evaluate(backend, ann_file, images_dir, imgsz=640, limit=None, conf_thres=VAL_CONF, iou_thres=VAL_IOU, max_det=MAX_DET, dets_out=None):
    images = eval_images(ann_file, images_dir, limit)
    if not images:
        raise FileNotFoundError(f"no images from {ann_file} under {images_dir}")
    t0 = time.perf_counter()
    dets = detect_all(backend, images, imgsz, conf_thres, iou_thres, max_det)
    if dets_out:
        Path(dets_out).parent.mkdir(parents=True, exist_ok=True)
        Path(dets_out).write_text(json.dumps(dets))
    m, m50 = coco_map(ann_file, dets, [i for i, _ in images])
    return Accuracy(m, m50, len(images), len(dets), conf_thres, iou_thres, max_det, time.perf_counter() - t0)
