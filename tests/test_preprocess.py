import numpy as np

from bench.preprocess import letterbox, scale_boxes, to_tensor


def test_letterbox_landscape_pads_top_and_bottom():
    img = np.zeros((480, 640, 3), np.uint8)
    out, meta = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert meta.scale == 1.0 and meta.pad_x == 0 and meta.pad_y == 80
    assert (out[:80] == 114).all() and (out[560:] == 114).all()
    assert (out[80:560] == 0).all()


def test_letterbox_scales_large_portrait():
    img = np.full((1000, 500, 3), 200, np.uint8)
    out, meta = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert abs(meta.scale - 0.64) < 1e-9
    assert meta.pad_y == 0 and meta.pad_x == 160
    assert (out[:, :160] == 114).all() and (out[:, 160:480] == 200).all()


def test_to_tensor_layout_and_range():
    img = np.zeros((640, 640, 3), np.uint8)
    img[..., 0] = 255
    t = to_tensor(img)
    assert t.shape == (1, 3, 640, 640) and t.dtype == np.float32 and t.flags.c_contiguous
    assert t[0, 0].min() == 1.0 and t[0, 1].max() == 0.0


def test_scale_boxes_round_trip():
    img = np.zeros((300, 400, 3), np.uint8)
    _, meta = letterbox(img, 640)
    orig = np.array([[40.0, 30.0, 200.0, 150.0]])
    in_lb = orig * meta.scale + np.array([meta.pad_x, meta.pad_y, meta.pad_x, meta.pad_y])
    back = scale_boxes(in_lb, meta)
    assert np.allclose(back, orig, atol=1e-3)


def test_scale_boxes_clips_to_image():
    _, meta = letterbox(np.zeros((300, 400, 3), np.uint8), 640)
    back = scale_boxes(np.array([[-50.0, -50.0, 700.0, 700.0]]), meta)
    assert back[0].tolist() == [0.0, 0.0, 400.0, 300.0]
