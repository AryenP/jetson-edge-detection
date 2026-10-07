"""Per-block int8 sensitivity: pin one block to fp16 at a time, measure mAP, rank.

    python -m scripts.sensitivity --limit 500

Builds one int8 engine per block (23 for yolov8n at depth 1) plus the
unpinned int8 and full fp16 references, all evaluated on the same image
subset. The subset ranks blocks; it is not a reportable mAP. The final mixed
engine is then built with build_engine --pin-fp16 and benchmarked on the
full split like any other row.
"""
import argparse
import json
import time
from pathlib import Path

from bench import backends
from bench.accuracy import evaluate

from .build_engine import add_calib_args, block_of, build, calib_from_args, parse


def list_blocks(onnx, depth):
    import tensorrt as trt

    _, network = parse(onnx, trt.Logger(trt.Logger.ERROR))
    names = {block_of(network.get_layer(i).name, depth) for i in range(network.num_layers)}
    names.discard(None)
    return sorted(names, key=lambda b: [int(t) if t.isdigit() else t for t in b.replace("model.", "").replace("/", ".").split(".")])


def eval_engine(engine, ann, images, imgsz, limit):
    be = backends.TensorRT(engine)
    try:
        return evaluate(be, ann, images, imgsz, limit).map_50_95
    finally:
        be.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="models/yolov8n_640.onnx")
    ap.add_argument("--blocks", default="", help="comma-separated subset; default is every block")
    ap.add_argument("--pin-depth", type=int, default=1, help="2 splits the head into model.22/cv2.0 etc.")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--coco-ann", default="datasets/coco/annotations/instances_val2017.json")
    ap.add_argument("--coco-images", default="datasets/coco/val2017")
    ap.add_argument("--out", default="runs/sensitivity.json")
    ap.add_argument("--engine-dir", default="engines/sensitivity")
    add_calib_args(ap)
    args = ap.parse_args()

    onnx = Path(args.onnx)
    calib = calib_from_args(args, onnx)
    out = Path(args.out)
    # resumable: a 25-engine sweep is hours on an Orin Nano and a crash mid-way shouldn't restart it
    rows = json.loads(out.read_text()) if out.exists() else {}
    engine_dir = Path(args.engine_dir)

    def run(key, precision, pins=()):
        if key in rows:
            return
        engine = engine_dir / f"{onnx.stem}_{key}.engine"
        t0 = time.perf_counter()
        try:
            sidecar = build(onnx, engine, precision, args.workspace_mb, args.imgsz, calib, pins, args.pin_depth, log=lambda *_: None)
            rows[key] = {"map_50_95": eval_engine(engine, args.coco_ann, args.coco_images, args.imgsz, args.limit), "n_pinned": len(sidecar["pinned_layers"] or []), "seconds": round(time.perf_counter() - t0)}
        except RuntimeError as e:
            rows[key] = {"map_50_95": None, "error": str(e), "seconds": round(time.perf_counter() - t0)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{key:28s} mAP50-95 {rows[key]['map_50_95']}  ({rows[key]['seconds']}s)")

    blocks = [b for b in args.blocks.split(",") if b] or list_blocks(onnx, args.pin_depth)
    print(f"{len(blocks)} blocks, {args.limit} images, calib {calib['split']}")
    run("fp16", "fp16")
    run("int8", "int8")
    for b in blocks:
        run(f"int8_fp16-{b.replace('/', '.')}", "int8", (b,))

    base = rows["int8"]["map_50_95"]
    ranked = sorted(((k, r) for k, r in rows.items() if k.startswith("int8_fp16-") and r["map_50_95"] is not None), key=lambda kr: -kr[1]["map_50_95"])
    print(f"\nfp16 {rows['fp16']['map_50_95']:.4f}   int8 {base:.4f}   gap {rows['fp16']['map_50_95'] - base:+.4f}\n")
    print("| block pinned to fp16 | layers | mAP50-95 | vs int8 |")
    print("|---|---|---|---|")
    for k, r in ranked:
        print(f"| {k[len('int8_fp16-'):]} | {r['n_pinned']} | {r['map_50_95']:.4f} | {r['map_50_95'] - base:+.4f} |")


if __name__ == "__main__":
    main()
