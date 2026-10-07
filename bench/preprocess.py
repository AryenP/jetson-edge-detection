from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

PAD = 114


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


def load_rgb(path):
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize(img, w, h):
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def letterbox(img, size=640):
    """Scale to fit a size x size square, pad the rest. Matches ultralytics' centred letterbox."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) != (w, h):
        img = resize(img, nw, nh)
    # ultralytics rounds the half-padding this way; off-by-one here shifts every box
    left = int(round((size - nw) / 2 - 0.1))
    top = int(round((size - nh) / 2 - 0.1))
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = img
    return out, LetterboxMeta(scale, left, top, w, h)


def to_tensor(img):
    chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None])


def preprocess(path, size=640):
    img, meta = letterbox(load_rgb(path), size)
    return to_tensor(img), meta


def scale_boxes(xyxy, meta):
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] -= meta.pad_x
    boxes[:, [1, 3]] -= meta.pad_y
    boxes /= meta.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta.orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta.orig_h)
    return boxes
