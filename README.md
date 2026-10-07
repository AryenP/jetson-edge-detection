# jetson-edge-detection

YOLOv8n on a Jetson Orin Nano, quantised to INT8 with TensorRT and measured
for latency, throughput, COCO mAP and power under a stated power mode.
Edge inference is latency under a power budget; most benchmarks leave the
watts out. This one treats the power mode and the power source as part of
every number.

## Status

| stage | state |
|---|---|
| ONNX export, pre/post-processing, COCO eval, results schema | done, checked against ultralytics on CPU |
| TensorRT FP16 / INT8 build, entropy calibration, engine runner | written, not yet run (no board) |
| FP16 baseline numbers | not measured |
| INT8 numbers and per-layer sensitivity | not measured |
| Power at each nvpmodel mode | not measured |

Hardware has not arrived. There are no performance numbers in this repo yet.

## Pipeline

```
YOLOv8n (PyTorch) ─► ONNX, opset 17, 1x3x640x640
                       ├─► TensorRT FP16 engine
                       └─► TensorRT INT8 engine ◄── IInt8EntropyCalibrator2 cache, 1000 COCO val2017 images
                                  │
                    latency p50/p95, fps ─┐
                    tegrastats @ 100 ms ──┼─► results.json (schema'd, one row per run)
                    COCO mAP50-95 ────────┘
```

## Results

Rendered from `results.json` by `./init.sh report`. Every row carries board,
JetPack and TensorRT version, model, input resolution, precision, calibration
set, nvpmodel mode, jetson_clocks state, warm-up count and power source.

_no runs yet_

## Run it

```
./init.sh test                 # unit tests, any machine
./init.sh export yolov8n.pt    # -> models/yolov8n_640.onnx + sidecar
./init.sh bench-cpu            # pipeline dry run on onnxruntime, nothing recorded

# on the board (docs/jetson-setup.md)
./init.sh env                  # JetPack, TensorRT, CUDA, nvpmodel mode
./init.sh coco all             # COCO val2017 + annotations
./init.sh calib                # seeded 1000-image calibration subset -> calib/manifest.json
./init.sh engine fp16          # or int8; writes engines/*.engine.json
./init.sh bench fp16           # or int8; appends one validated row to results.json
./init.sh report
```

## Measurement rules

These are enforced by `bench/results.py`, not just intended: a row missing
its nvpmodel mode, an INT8 row without calibration details, or p95 below p50
is rejected before it is written.

- Latency is per frame at batch 1 and spans host-to-device copy, execute,
  device-to-host copy and a stream sync. Pre- and post-processing are outside
  it. 50 warm-up iterations are discarded, 500 are timed. fps is 1000 / mean
  latency, derived, not measured separately.
- Power is the mean of tegrastats samples at 100 ms over the timed loop only,
  on the module input rail (`VDD_IN` on Orin Nano). tegrastats is approximate
  and excludes the carrier board. Where a USB meter is in line both are
  recorded and the meter is the reference (`power_w.source`).
- INT8 calibration uses 1000 COCO val2017 images picked with a fixed seed;
  the file list and its hash sit in `calib/manifest.json` and in the engine
  sidecar. The cache comes from TensorRT's `IInt8EntropyCalibrator2` and is
  the file `trtexec --calib` reads.
- mAP is COCO val2017, all 5000 images, same letterbox and NMS at every
  precision (conf 0.001, IoU 0.7, max 300 detections, the `yolo val` defaults).

## Layout

```
bench/        preprocess, postprocess, backends, latency, power, accuracy, env, results, run
scripts/      export_onnx, build_engine, calibrate, prepare_calib, fetch_coco
tests/        CPU-only unit tests, plus an ONNX smoke test when an export is present
docs/         jetson-setup.md
experiments/  dead ends, one line each on why
DECISIONS.md  choices, rejected alternatives, reversals
```
