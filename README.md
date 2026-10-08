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
| TensorRT FP16 / INT8 build, entropy calibration, fp16 pins, engine runner | written, tested against an in-memory fake of tensorrt and pycuda, not yet run on hardware |
| Explicit INT8 via ModelOpt Q/DQ | done, checked on CPU against fp32 on coco128 (see DECISIONS) |
| FP16 baseline numbers | not measured |
| INT8 numbers, val2017 and train2017 calibration | not measured |
| Per-block fp16-pin sensitivity, mixed engine | not measured |
| Power at each nvpmodel mode | not measured |

Hardware has not arrived. There are no performance numbers in this repo yet.

## Pipeline

```
YOLOv8n (PyTorch) ─► ONNX, opset 17, 1x3x640x640
                       ├─► TensorRT FP16 engine
                       ├─► TensorRT INT8 engine ◄── IInt8EntropyCalibrator2 cache, 1000 COCO images
                       └─► ModelOpt Q/DQ ONNX ─► TensorRT INT8 engine (explicit, same 1000 images)
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
./init.sh calib                # seeded 1000-image subset -> calib/val2017/manifest.json
./init.sh coco 1000 train2017  # and the same from train2017
./init.sh calib train2017
./init.sh quantize             # on the laptop: explicit int8 Q/DQ onnx via ModelOpt, ~minutes on CPU
./init.sh engine fp16          # writes engines/*.engine + .engine.json sidecar
./init.sh engine int8          # calibrates on val2017; `int8 train2017` for the other
./init.sh engine int8qdq       # builds from the Q/DQ onnx, no calibrator
./init.sh bench fp16           # appends one validated row to results.json
./init.sh bench int8
./init.sh bench int8qdq
./init.sh sens --limit 500     # per-block sensitivity ranking -> runs/sensitivity.json
./init.sh engine int8 val2017 --pin-fp16 model.22   # then bench it
./init.sh sweep "0 1" "yolov8n yolov8s yolov8m yolov8l"  # every model in every power mode
./init.sh report               # rows, then fp16 -> int8 speedup / mAP cost / power delta
./init.sh plots                # docs/plots/latency_power.png, map_latency.png
```

## Measurement rules

These are enforced by `bench/results.py`, not just intended: a row missing
its nvpmodel mode, an INT8 row without calibration details, or p95 below p50
is rejected before it is written.

- Latency is per frame at batch 1 and spans host-to-device copy, execute,
  device-to-host copy and a stream sync. Pre- and post-processing are outside
  it. Execute-only time from CUDA events, the figure trtexec reports, is
  recorded alongside. 50 warm-up iterations are discarded, 500 are timed. fps
  is 1000 / mean latency, derived, not measured separately.
- Power is the mean of tegrastats samples at 100 ms over the timed loop only,
  on the module input rail (`VDD_IN` on Orin Nano). tegrastats is approximate
  and excludes the carrier board. Where a USB meter is in line both are
  recorded and the meter is the reference (`power_w.source`).
- INT8 calibration uses 1000 images picked with a fixed seed, once from
  val2017 and once from train2017; each file list and its hash sit in
  `calib/<split>/manifest.json` and in the engine sidecar. The cache comes
  from TensorRT's `IInt8EntropyCalibrator2` and is the file `trtexec --calib`
  reads. INT8 engines keep the FP16 flag, so layers TensorRT will not run in
  int8 fall back to fp16.
- Per-block sensitivity (`./init.sh sens`) pins one `model.N` block to fp16
  at a time and ranks blocks by mAP recovered on a 500-image subset. Only
  the final mixed engine, benchmarked on the full split, is a reported number.
- mAP is COCO val2017, all 5000 images, same letterbox and NMS at every
  precision (conf 0.001, IoU 0.7, max 300 detections, the `yolo val` defaults).

## Layout

```
bench/        preprocess, postprocess, backends, latency, power, accuracy, env, results, run
scripts/      export_onnx, build_engine, calibrate, sensitivity, prepare_calib, fetch_coco, plot
tests/        CPU-only unit tests; fake_trt.py stands in for tensorrt and pycuda so the board paths run too
docs/         jetson-setup.md
experiments/  dead ends, one line each on why
DECISIONS.md  choices, rejected alternatives, reversals
```
