import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.env import collect

from .export_onnx import sha256

SPLITS = ("val2017", "train2017")


def block_of(name, depth=1):
    """'/model.22/cv2.0/cv2.0.0/conv/Conv' -> 'model.22' (depth 1) or 'model.22/cv2.0' (depth 2)."""
    m = re.match(r"/(model\.\d+)((?:/[^/]+){0,%d})" % (depth - 1), name)
    return (m.group(1) + m.group(2)) if m else None


def calib_paths(split, onnx_stem):
    d = Path("calib") / split
    return {"manifest": d / "manifest.json", "images": d / "images", "cache": d / f"{onnx_stem}_entropy2.cache"}


def parse(onnx, logger):
    import tensorrt as trt

    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TensorRT 10; the old flag is ignored
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError("onnx parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    return builder, network


def pin_blocks(network, blocks, depth):
    """Force every float layer in `blocks` to compute in fp16. Returns the layer names pinned."""
    import tensorrt as trt

    pinned = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if block_of(layer.name, depth) not in blocks:
            continue
        # shape/constant layers carry int32; a half precision constraint on them fails the build
        if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE) or any(layer.get_output(j).dtype != trt.float32 for j in range(layer.num_outputs)):
            continue
        # precision only, not output type: the output may still be re-quantized to int8 for the
        # next layer, so the pin isolates this block's own arithmetic rather than its neighbours'
        layer.precision = trt.DataType.HALF
        pinned.append(layer.name)
    return pinned


def build(onnx, out, precision, workspace_mb, imgsz, calib=None, pin_fp16=(), pin_depth=1, log=print):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder, network = parse(onnx, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    calib_info = {"calib_imgs": None, "calib_batch_size": None, "calib_scheme": None}
    pinned = []
    if precision == "int8":
        from .calibrate import entropy_calibrator, load_manifest

        # FP16 stays on so layers TensorRT won't run in int8 fall back to fp16 rather than fp32
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = entropy_calibrator(calib["manifest"], calib["images"], calib["cache"], calib["batch"], imgsz)
        m = load_manifest(calib["manifest"])
        calib_info = {
            "calib_imgs": (m["n"] // calib["batch"]) * calib["batch"],
            "calib_batch_size": calib["batch"],
            # entropy cache = per-tensor activation scales; the builder quantizes weights per-channel
            "calib_scheme": "mixed",
            "calib_split": calib["split"],
            "calib_manifest_sha256": m["files_sha256"],
            "calib_seed": m["seed"],
            "calib_cache": str(calib["cache"]),
        }
        if pin_fp16:
            # OBEY, not PREFER: a pin the builder silently drops would make the sensitivity number a lie
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            pinned = pin_blocks(network, set(pin_fp16), pin_depth)
            log(f"pinned {len(pinned)} layers in {sorted(pin_fp16)} to fp16")

    log(f"building {precision} from {onnx}")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build failed, see log above")
    build_s = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(serialized))

    env = collect()
    sidecar = {
        "engine": str(out),
        "engine_sha256": sha256(out),
        "onnx": str(onnx),
        "onnx_sha256": sha256(onnx),
        "precision": precision,
        "imgsz": imgsz,
        "workspace_mb": workspace_mb,
        "build_seconds": round(build_s),
        "tensorrt": trt.__version__,
        "board": env["board"],
        "jetpack": env["jetpack"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pin_fp16": sorted(pin_fp16) or None,
        "pinned_layers": pinned or None,
        **calib_info,
    }
    out.with_suffix(".engine.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {build_s:.0f}s")
    if precision == "int8" and not pin_fp16:
        log(f"trtexec equivalent: trtexec --onnx={onnx} --int8 --fp16 --calib={calib['cache']} --saveEngine={out}")
    return sidecar


def add_calib_args(ap):
    ap.add_argument("--calib-split", choices=SPLITS, default="val2017")
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--imgsz", type=int, default=640)


def calib_from_args(args, onnx):
    paths = calib_paths(args.calib_split, onnx.stem)
    return {**paths, "split": args.calib_split, "batch": args.calib_batch}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/yolov8n_640.onnx")
    ap.add_argument("--precision", choices=["fp16", "int8"], required=True)
    ap.add_argument("--out", help="default: engines/<stem>_fp16.engine or engines/<stem>_int8_<split>.engine")
    ap.add_argument("--pin-fp16", default="", help="int8 only: comma-separated blocks to keep in fp16, e.g. model.22,model.21")
    ap.add_argument("--pin-depth", type=int, default=1)
    add_calib_args(ap)
    args = ap.parse_args()
    onnx = Path(args.onnx)
    pins = tuple(b for b in args.pin_fp16.split(",") if b)
    if args.precision == "fp16":
        out = Path(args.out or f"engines/{onnx.stem}_fp16.engine")
        build(onnx, out, "fp16", args.workspace_mb, args.imgsz)
    else:
        suffix = f"_fp16-{'-'.join(pins)}" if pins else ""
        out = Path(args.out or f"engines/{onnx.stem}_int8_{args.calib_split}{suffix}.engine")
        build(onnx, out, "int8", args.workspace_mb, args.imgsz, calib_from_args(args, onnx), pins, args.pin_depth)
