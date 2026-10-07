from dataclasses import dataclass

import numpy as np

# yolo val defaults; the same three numbers at every precision or the mAP delta is noise
VAL_CONF = 0.001
VAL_IOU = 0.7
MAX_DET = 300


@dataclass
class Detections:
    boxes: np.ndarray  # xyxy, letterboxed-input pixels
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self):
        return int(self.boxes.shape[0])


def xywh2xyxy(xywh):
    out = np.empty_like(xywh)
    hw, hh = xywh[:, 2] / 2, xywh[:, 3] / 2
    out[:, 0] = xywh[:, 0] - hw
    out[:, 1] = xywh[:, 1] - hh
    out[:, 2] = xywh[:, 0] + hw
    out[:, 3] = xywh[:, 1] + hh
    return out


def box_iou(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes, scores, iou_thres):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest]) <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def decode(output, conf_thres=VAL_CONF, iou_thres=VAL_IOU, max_det=MAX_DET):
    """(1, 84, 8400) head output -> per-class NMS'd detections.

    Rows 0-3 are cx, cy, w, h in input pixels; the other 80 are sigmoided class
    scores. There is no objectness row in the v8 head.
    """
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T
    cls_scores = pred[:, 4:]
    classes = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(classes)), classes]
    keep = scores > conf_thres
    boxes, scores, classes = xywh2xyxy(pred[keep, :4]), scores[keep], classes[keep]
    if not len(boxes):
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    # offset each class into its own region so one NMS pass is per-class
    offset = classes.astype(np.float32)[:, None] * 7680.0
    idx = nms(boxes + offset, scores, iou_thres)[:max_det]
    return Detections(boxes[idx], scores[idx], classes[idx].astype(np.int64))
