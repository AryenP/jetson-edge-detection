import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.env import collect

from .export_onnx import sha256


def build(onnx, out, precision, workspace_mb, imgsz, calib=None):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError("onnx parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    calib_info = {"calib_imgs": None, "calib_batch_size": None, "calib_scheme": None}
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
            "calib_manifest_sha256": m["files_sha256"],
            "calib_seed": m["seed"],
            "calib_cache": str(calib["cache"]),
        }

    print(f"building {precision} from {onnx}")
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
        **calib_info,
    }
    out.with_suffix(".engine.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {build_s:.0f}s")
    if precision == "int8":
        print(f"trtexec equivalent: trtexec --onnx={onnx} --int8 --fp16 --calib={calib['cache']} --saveEngine={out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/yolov8n_640.onnx")
    ap.add_argument("--precision", choices=["fp16", "int8"], required=True)
    ap.add_argument("--out", help="default: engines/<onnx stem>_<precision>.engine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--calib-manifest", default="calib/manifest.json")
    ap.add_argument("--calib-images", default="calib/images")
    ap.add_argument("--calib-cache", help="default: calib/<onnx stem>_entropy2.cache")
    ap.add_argument("--calib-batch", type=int, default=8)
    args = ap.parse_args()
    onnx = Path(args.onnx)
    out = Path(args.out or f"engines/{onnx.stem}_{args.precision}.engine")
    calib = None
    if args.precision == "int8":
        calib = {
            "manifest": Path(args.calib_manifest),
            "images": Path(args.calib_images),
            "cache": Path(args.calib_cache or f"calib/{onnx.stem}_entropy2.cache"),
            "batch": args.calib_batch,
        }
    build(onnx, out, args.precision, args.workspace_mb, args.imgsz, calib)
