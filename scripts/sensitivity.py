"""Per-block int8 sensitivity: take one block out of int8 at a time, measure mAP, rank.

    python -m scripts.sensitivity --limit 500             # on the board: tensorrt calibrator, fp16 pins
    python -m scripts.sensitivity --explicit --limit 500  # on a laptop: modelopt q/dq, block excluded

Builds one int8 variant per block (23 for yolov8n at depth 1) plus the
all-int8 and float references, all evaluated on the same image subset. The
subset ranks blocks; it is not a reportable mAP. The final mixed engine is
then built with build_engine --pin-fp16 (or quantize_onnx with the same
exclusions) and benchmarked on the full split like any other row.
"""
import argparse
import json
import time
from pathlib import Path

from bench import backends
from bench.accuracy import evaluate

from .build_engine import add_calib_args, block_of, build, calib_from_args, parse
from .quantize_onnx import calib_array, quantize


def list_blocks(onnx, depth, explicit=False):
    if explicit:
        import onnx as onnx_lib

        names = {block_of(n.name, depth) for n in onnx_lib.load(str(onnx)).graph.node}
    else:
        import tensorrt as trt

        _, network = parse(onnx, trt.Logger(trt.Logger.ERROR))
        names = {block_of(network.get_layer(i).name, depth) for i in range(network.num_layers)}
    names.discard(None)
    return sorted(names, key=lambda b: [int(t) if t.isdigit() else t for t in b.replace("model.", "").replace("/", ".").split(".")])


def eval_model(path, ann, images, imgsz, limit):
    be = backends.load(path)
    try:
        return evaluate(be, ann, images, imgsz, limit).map_50_95
    finally:
        be.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="models/yolov8n_640.onnx")
    ap.add_argument("--explicit", action="store_true", help="modelopt q/dq on cpu instead of tensorrt engines")
    ap.add_argument("--blocks", default="", help="comma-separated subset; default is every block")
    ap.add_argument("--pin-depth", type=int, default=1, help="2 splits the head into model.22/cv2.0 etc.")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--calib-n", type=int, default=200, help="explicit mode: manifest images per variant")
    ap.add_argument("--method", choices=("entropy", "max"), default="entropy")
    ap.add_argument("--coco-ann", default="datasets/coco/annotations/instances_val2017.json")
    ap.add_argument("--coco-images", default="datasets/coco/val2017")
    ap.add_argument("--out", default=None, help="default runs/sensitivity.json, or runs/sensitivity_explicit.json")
    ap.add_argument("--engine-dir", default="engines/sensitivity")
    add_calib_args(ap)
    args = ap.parse_args()

    onnx = Path(args.onnx)
    calib = calib_from_args(args, onnx)
    out = Path(args.out or ("runs/sensitivity_explicit.json" if args.explicit else "runs/sensitivity.json"))
    # resumable: a 25-variant sweep is hours either way and a crash mid-way shouldn't restart it
    rows = json.loads(out.read_text()) if out.exists() else {}
    work_dir = Path(args.engine_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def record(key, t0, **fields):
        rows[key] = {**fields, "seconds": round(time.perf_counter() - t0)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{key:28s} mAP50-95 {rows[key]['map_50_95']}  ({rows[key]['seconds']}s)")

    def run_trt(key, precision, pins=()):
        if key in rows:
            return
        engine = work_dir / f"{onnx.stem}_{key}.engine"
        t0 = time.perf_counter()
        try:
            sidecar = build(onnx, engine, precision, args.workspace_mb, args.imgsz, calib, pins, args.pin_depth, quiet=True)
            record(key, t0, map_50_95=eval_model(engine, args.coco_ann, args.coco_images, args.imgsz, args.limit), n_pinned=len(sidecar["pinned_layers"] or []))
        except RuntimeError as e:
            record(key, t0, map_50_95=None, error=str(e))

    def run_explicit(key, exclude=None):
        if key in rows:
            return
        t0 = time.perf_counter()
        if exclude is None and key == "fp32":
            record(key, t0, map_50_95=eval_model(onnx, args.coco_ann, args.coco_images, args.imgsz, args.limit))
            return
        path = work_dir / f"{onnx.stem}_{key}.onnx"
        quantize(onnx, path, x, args.method, [rf"/{exclude}/.*"] if exclude else None)
        record(key, t0, map_50_95=eval_model(path, args.coco_ann, args.coco_images, args.imgsz, args.limit))

    blocks = [b for b in args.blocks.split(",") if b] or list_blocks(onnx, args.pin_depth, args.explicit)
    if args.explicit:
        x = calib_array(calib["manifest"], calib["images"], args.calib_n, args.imgsz)
        print(f"{len(blocks)} blocks, {args.limit} eval images, {x.shape[0]} {calib['split']} calibration images, {args.method}")
        run_explicit("fp32")
        run_explicit("int8qdq")
        for b in blocks:
            run_explicit(f"int8qdq_excl-{b.replace('/', '.')}", b)
        ref, base, prefix, label = "fp32", "int8qdq", "int8qdq_excl-", "block kept float"
    else:
        print(f"{len(blocks)} blocks, {args.limit} images, calib {calib['split']}")
        run_trt("fp16", "fp16")
        run_trt("int8", "int8")
        for b in blocks:
            run_trt(f"int8_fp16-{b.replace('/', '.')}", "int8", (b,))
        ref, base, prefix, label = "fp16", "int8", "int8_fp16-", "block pinned to fp16"

    b0 = rows[base]["map_50_95"]
    ranked = sorted(((k, r) for k, r in rows.items() if k.startswith(prefix) and r["map_50_95"] is not None), key=lambda kr: -kr[1]["map_50_95"])
    print(f"\n{ref} {rows[ref]['map_50_95']:.4f}   {base} {b0:.4f}   gap {rows[ref]['map_50_95'] - b0:+.4f}\n")
    print(f"| {label} | mAP50-95 | vs {base} |")
    print("|---|---|---|")
    for k, r in ranked:
        print(f"| {k[len(prefix):]} | {r['map_50_95']:.4f} | {r['map_50_95'] - b0:+.4f} |")


if __name__ == "__main__":
    main()
