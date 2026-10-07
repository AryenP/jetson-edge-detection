"""Benchmark one engine: latency, fps, power, mAP -> one validated row in results.json.

    python -m bench.run --engine engines/yolov8n_640_fp16.engine
    python -m bench.run --engine engines/yolov8n_640_int8.engine --external-meter-w 9.8
    python -m bench.run --engine models/yolov8n_640.onnx --no-power --dry-run
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import backends, results
from .accuracy import evaluate
from .env import collect
from .latency import measure
from .power import Tegrastats


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sidecar_for(engine):
    p = engine.with_suffix(engine.suffix + ".json")
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help=".engine, or .onnx for an onnxruntime dry run")
    ap.add_argument("--precision", choices=results.PRECISIONS, help="default: from the engine sidecar")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--coco-ann", default="datasets/coco/annotations/instances_val2017.json")
    ap.add_argument("--coco-images", default="datasets/coco/val2017")
    ap.add_argument("--accuracy-limit", type=int, help="first N images only; smoke tests, never a reported number")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--power-rail", help="tegrastats rail; default is the module input rail")
    ap.add_argument("--external-meter-w", type=float, help="mean W read off a USB meter over the timed loop; becomes the reference")
    ap.add_argument("--nvpmodel", help="override the detected mode string")
    ap.add_argument("--jetson-clocks", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--run-id")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--dry-run", action="store_true", help="print the row instead of appending")
    args = ap.parse_args()

    engine = Path(args.engine)
    sidecar = sidecar_for(engine)
    precision = args.precision or sidecar.get("precision")
    if precision is None:
        if engine.suffix != ".onnx":
            raise SystemExit(f"no sidecar for {engine}; pass --precision")
        precision = "fp16"  # onnxruntime is fp32; label is for the schema, the row is never recorded
    imgsz = int(sidecar.get("imgsz", 640))
    model = Path(sidecar.get("onnx", engine.name)).stem

    env = collect()
    if args.jetson_clocks == "auto":
        clocks = env["jetson_clocks"]
        if clocks is None:
            print("warning: jetson_clocks state not detectable; recording False")
            clocks = False
    else:
        clocks = args.jetson_clocks == "yes"
    nvp = args.nvpmodel or env["nvpmodel_mode"]
    if nvp == "unknown" and engine.suffix == ".engine":
        raise SystemExit("nvpmodel mode unknown; a row without it is not comparable. pass --nvpmodel")

    be = backends.load(engine)
    print(f"{be.name}: {engine} input {be.input_shape}")
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    def step():
        be.infer(x)

    gpu_ms = (lambda: be.gpu_ms) if be.name == "tensorrt" else None

    print(f"latency: {args.warmup} warm-up + {args.iters} timed")
    sample_power = not args.no_power and Tegrastats.available()
    if not args.no_power and not sample_power:
        print("warning: tegrastats not found; power recorded as 0")
    power_meta = {}
    tegra_w = None
    if sample_power:
        with Tegrastats() as ts:
            lat = measure(step, args.warmup, args.iters, gpu_ms)
        try:
            ps = ts.summary(args.power_rail)
            power_meta = {"tegrastats": ps.__dict__}
            tegra_w = ps.mean_w
            print(f"  power {ps.mean_w:.2f} W mean of {ps.n_samples} samples on {ps.rail}")
        except ValueError as e:
            print(f"warning: {e}")
    else:
        lat = measure(step, args.warmup, args.iters, gpu_ms)
    print(f"  p50 {lat.p50_ms:.2f} ms  p95 {lat.p95_ms:.2f} ms  mean {lat.mean_ms:.2f} ms  {lat.fps:.1f} fps")
    if lat.gpu_p50_ms is not None:
        print(f"  execute only: p50 {lat.gpu_p50_ms:.2f} ms  p95 {lat.gpu_p95_ms:.2f} ms")

    if args.external_meter_w is not None:
        power = {"mean": args.external_meter_w, "source": "external_meter"}
    else:
        power = {"mean": tegra_w or 0.0, "source": "tegrastats"}

    ts_now = datetime.now(timezone.utc)
    run_id = args.run_id or f"{ts_now.strftime('%Y%m%dT%H%M%SZ')}_{model}_{precision}"
    acc = None
    if not args.no_accuracy:
        acc = evaluate(be, args.coco_ann, args.coco_images, imgsz, args.accuracy_limit, dets_out=f"runs/{run_id}_dets.json")
        print(f"  mAP50-95 {acc.map_50_95:.4f}  mAP50 {acc.map_50:.4f}  on {acc.n_images} images")
    be.close()

    row = {
        "run_id": run_id,
        "timestamp": ts_now.isoformat(timespec="seconds"),
        "board": env["board"],
        "jetpack": env["jetpack"],
        "tensorrt": env["tensorrt"],
        "model": model,
        "input_res": imgsz,
        "precision": precision,
        "calib_imgs": sidecar.get("calib_imgs"),
        "calib_batch_size": sidecar.get("calib_batch_size"),
        "calib_scheme": sidecar.get("calib_scheme"),
        "nvpmodel_mode": nvp,
        "jetson_clocks": bool(clocks),
        "n_warmup_discarded": args.warmup,
        "latency_ms": {"p50": round(lat.p50_ms, 3), "p95": round(lat.p95_ms, 3)},
        "fps": round(lat.fps, 2),
        "map_50_95": round(acc.map_50_95, 4) if acc else 0.0,
        "power_w": power,
        "meta": {
            "backend": be.name,
            "engine": str(engine),
            "engine_sha256": sidecar.get("engine_sha256"),
            "onnx_sha256": sidecar.get("onnx_sha256"),
            "latency": lat.as_dict(),
            "calib_split": sidecar.get("calib_split"),
            "pin_fp16": sidecar.get("pin_fp16"),
            "accuracy": acc.__dict__ if acc else None,
            "l4t": env["l4t"],
            "cuda": env["cuda"],
            "git": git_rev(),
            **power_meta,
        },
    }
    results.validate(row)
    if args.dry_run:
        print(json.dumps(row, indent=2))
        return
    results.append(args.out, row)
    print(f"appended {run_id} to {args.out}")


if __name__ == "__main__":
    main()
