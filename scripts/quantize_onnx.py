"""Explicit INT8 quantization: ONNX -> ONNX with Q/DQ nodes, via NVIDIA ModelOpt.

    python -m scripts.quantize_onnx --calib-split val2017

Runs on a laptop CPU in minutes; the Jetson only builds the engine from the
result. Unlike the TensorRT calibrator path this is not deprecated, and the
scales live in the graph where they can be inspected per tensor.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench.preprocess import preprocess

from .build_engine import calib_paths
from .export_onnx import sha256


def calib_array(manifest, images_dir, n, imgsz):
    """(N, 3, imgsz, imgsz) float32 from the first n manifest images, same letterbox as everything else."""
    files = json.loads(Path(manifest).read_text())["files"][:n]
    missing = [f for f in files if not (Path(images_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} calibration images missing under {images_dir}, e.g. {missing[0]}")
    return np.concatenate([preprocess(Path(images_dir) / f, imgsz)[0] for f in files]).astype(np.float32)


def quantize(onnx, out, x, method):
    import modelopt
    import modelopt.onnx.quantization as moq

    moq.quantize(onnx_path=str(onnx), calibration_data={"images": x}, calibration_method=method, quantize_mode="int8", output_path=str(out))
    return modelopt.__version__


def count_qdq(path):
    import onnx

    ops = [n.op_type for n in onnx.load(str(path)).graph.node]
    return ops.count("QuantizeLinear"), len(ops)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/yolov8n_640.onnx")
    ap.add_argument("--calib-split", choices=("val2017", "train2017"), default="val2017")
    ap.add_argument("--n", type=int, help="images from the manifest; default all")
    ap.add_argument("--method", choices=("entropy", "max"), default="entropy")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", help="default: models/<stem>_int8qdq_<split>.onnx")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    paths = calib_paths(args.calib_split, onnx.stem)
    manifest = json.loads(paths["manifest"].read_text())
    n = args.n or manifest["n"]
    out = Path(args.out or f"models/{onnx.stem}_int8qdq_{args.calib_split}.onnx")
    x = calib_array(paths["manifest"], paths["images"], n, args.imgsz)
    print(f"calibrating on {x.shape[0]} {args.calib_split} images, {args.method}")
    version = quantize(onnx, out, x, args.method)
    n_q, n_nodes = count_qdq(out)
    sidecar = {
        "qdq": True,
        "onnx": str(onnx),
        "onnx_sha256": sha256(onnx),
        "sha256": sha256(out),
        "imgsz": args.imgsz,
        "calib_imgs": x.shape[0],
        "calib_batch_size": x.shape[0],
        "calib_split": args.calib_split,
        "calib_method": f"modelopt-{args.method}",
        # Q/DQ from ModelOpt: per-channel weight scales, per-tensor activation scales
        "calib_scheme": "mixed",
        "calib_manifest_sha256": manifest["files_sha256"],
        "quantized_nodes": n_q,
        "nodes": n_nodes,
        "modelopt": version,
        "quantized_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".onnx.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {n_q} quantized tensors in {n_nodes} nodes")
