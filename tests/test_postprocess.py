import numpy as np

from bench.postprocess import box_iou, decode, nms, xywh2xyxy


def _raw(rows):
    """rows: list of (cx, cy, w, h, cls, score) -> (1, 84, N) head output."""
    out = np.zeros((1, 84, max(len(rows), 1)), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


def test_xywh2xyxy():
    assert xywh2xyxy(np.array([[10.0, 20.0, 4.0, 6.0]])).tolist() == [[8.0, 17.0, 12.0, 23.0]]


def test_box_iou_identical_and_disjoint():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    ious = box_iou(a, np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [5.0, 0.0, 15.0, 10.0]]))
    assert np.allclose(ious, [1.0, 0.0, 1 / 3])


def test_nms_suppresses_overlaps_keeps_best_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_decode_filters_conf_and_runs_per_class_nms():
    raw = _raw(
        [
            (100, 100, 50, 50, 0, 0.9),  # person
            (102, 102, 50, 50, 0, 0.8),  # same person, suppressed
            (102, 102, 50, 50, 2, 0.6),  # car at same spot: different class survives
            (300, 300, 20, 20, 5, 0.0005),  # below conf
        ]
    )
    d = decode(raw, conf_thres=0.001, iou_thres=0.7)
    assert len(d) == 2
    assert np.allclose(d.scores, [0.9, 0.6]) and d.classes.tolist() == [0, 2]
    assert np.allclose(d.boxes[0], [75, 75, 125, 125])


def test_decode_respects_max_det_and_empty():
    raw = _raw([(10 + 100 * i, 10, 5, 5, 1, 0.5) for i in range(20)])
    assert len(decode(raw, max_det=7)) == 7
    empty = decode(np.zeros((1, 84, 8400), np.float32))
    assert len(empty) == 0 and empty.boxes.shape == (0, 4)
