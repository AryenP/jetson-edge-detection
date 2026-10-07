# Decisions

Each entry: what was chosen, what was rejected, why. Entries marked *open*
are waiting on data from the board. Reversals go at the bottom and are never
deleted.

## Calibration method: IInt8EntropyCalibrator2

Chosen over `IInt8MinMaxCalibrator` because min-max clips to the observed
range and a single outlier activation blows the scale for the whole tensor;
entropy picks the threshold that minimises KL divergence to the fp32
histogram. Chosen over the legacy percentile calibrator because it is
deprecated in TensorRT 10. *Open:* min-max on the detection head only, if
entropy clips the box regression range.

## Calibration set: 1000 COCO val2017 images, seed 0

The same split the mAP is measured on. That leaks nothing (calibration sees
no labels) but it does mean the calibration distribution matches the test
distribution exactly, which flatters INT8. *Open:* repeat with 1000 train2017
images and report both.

## Quantisation scheme recorded as `mixed`

TensorRT implicit quantisation gives per-tensor activation scales from the
cache and per-channel symmetric weight scales from the builder. There is no
knob to make activations per-channel. "Which layers lose the most to
quantisation" therefore has to be answered by per-layer precision
sensitivity (leave layer N in fp16, measure mAP), not by switching scale
granularity. *Open:* whether that sweep uses `ILayer.precision` pins or
explicit Q/DQ via ModelOpt.

## INT8 engines keep the FP16 flag

Layers TensorRT refuses to run in int8 fall back to fp16 instead of fp32.
Matches how DeepStream-Yolo and trtexec `--int8 --fp16` build. Cost: the
"INT8" engine is really int8-with-fp16-fallback and the README has to say so.

## Latency window: H2D + execute + D2H + sync

Pure GPU compute time (CUDA events around execute only) is what trtexec
reports and what the published numbers use. The copies are included here
because an application pays them, and on a unified-memory board they are
small enough to leave in. *Open:* report trtexec's GPU-compute-only figure
alongside, so the two are comparable.

## fps = 1000 / mean latency

Not 1000 / p50. Frames finished per second of the loop is a throughput and
belongs to the mean. The row carries both percentiles so either can be
recomputed.

## Power: tegrastats VDD_IN, instantaneous samples, 100 ms

tegrastats' own `/avg` column is a running mean since the process started,
not since the timed loop, so it is ignored and the mean is taken over the
samples inside the window. VDD_IN is module input, not wall draw; the carrier
board and fan are outside it. A USB meter at the barrel jack is the reference
when present.

## Engines built with the Python builder, not trtexec

The calibrator is a Python class, so the cache has to come from a Python
build. Once the cache exists trtexec can rebuild from it, and the builder
prints the equivalent trtexec line for that reason.

## Reversals

None yet. The first measured number that contradicts a choice above goes here.
