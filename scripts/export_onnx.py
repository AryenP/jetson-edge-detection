import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(weights, out, imgsz, opset):
    import torch
    import ultralytics
    from ultralytics import YOLO

    # fixed batch 1 and no dynamic axes: the engine runner allocates buffers once from the engine shape
    produced = Path(YOLO(weights).export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, batch=1, half=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out.resolve():
        shutil.move(produced, out)
    sidecar = {
        "weights": Path(weights).name,
        "imgsz": imgsz,
        "opset": opset,
        "batch": 1,
        "sha256": sha256(out),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".onnx.json").write_text(json.dumps(sidecar, indent=2) + "\n")


def check(onnx, imgsz):
    import numpy as np

    from bench.backends import OnnxRuntime

    be = OnnxRuntime(onnx)
    y = be.infer(np.random.rand(1, 3, imgsz, imgsz).astype(np.float32))
    anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    assert y.shape == (1, 84, anchors), f"unexpected output shape {y.shape}"
    print(f"ok: {onnx} {be.input_shape} -> {y.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", help="default: models/<weights stem>_<imgsz>.onnx")
    args = ap.parse_args()
    out = Path(args.out or f"models/{Path(args.weights).stem}_{args.imgsz}.onnx")
    export(args.weights, out, args.imgsz, args.opset)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")
    check(out, args.imgsz)
