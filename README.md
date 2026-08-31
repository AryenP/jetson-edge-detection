# jetson-edge-detection

YOLOv8n on a Jetson Orin Nano: ONNX export, TensorRT engines at FP16 and INT8,
benchmarked for latency, throughput, accuracy, and power.

Work in progress. Hardware not yet in hand.

## Pipeline

```
YOLOv8n (PyTorch) → ONNX → TensorRT engine
                             ├─ FP16 baseline
                             └─ INT8 + calibration cache
→ latency, FPS, mAP, power → compare
```

## Running it

```
./init.sh env              # JetPack, TensorRT, CUDA, nvpmodel mode
./init.sh export           # YOLOv8n → ONNX
./init.sh engine fp16      # or: engine int8
./init.sh bench            # → results.json
```

## Measurement notes

- Every row in `results.json` carries board, JetPack and TensorRT version, model,
  input resolution, precision, `nvpmodel` mode, and whether `jetson_clocks` was
  applied. Power modes change the numbers; omitting the mode makes them useless
- Power is the mean over the benchmark window, sampled from `tegrastats` at
  100 ms. tegrastats is approximate — where an external USB meter is used, both
  are reported and the meter is the reference
- Latency is reported as p50 and p95. FPS is derived from latency, not measured
  separately
- INT8 calibration uses ~1000 COCO val2017 images. `trtexec --calib` consumes a
  calibration *cache*; the cache is produced by a TensorRT `IInt8EntropyCalibrator2`
  over that image set. Calibration set size, batch size, and preprocessing are
  recorded alongside each engine
- Accuracy is measured on the same COCO val split at every precision, with
  identical preprocessing and NMS thresholds
