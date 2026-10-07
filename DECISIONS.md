# Decisions

Each entry: what was chosen, what was rejected, why. Entries marked *open*
are waiting on data from the board. Reversals go at the bottom and are never
deleted.

## Calibration method: IInt8EntropyCalibrator2

Chosen over `IInt8MinMaxCalibrator` because min-max clips to the observed
range and a single outlier activation blows the scale for the whole tensor;
entropy picks the threshold that minimises KL divergence to the fp32
histogram. Known cost: TensorRT 10.1 deprecated the whole implicit
quantisation path, calibrators included, in favour of explicit Q/DQ
quantisation. It still ships and works in the 10.3 that JetPack 6.2 carries,
with a deprecation warning, and it is what every published Jetson YOLO INT8
number was produced with, so it stays as the baseline method. That
deprecation is the strongest argument for the ModelOpt follow-up below.
*Open:* min-max on the detection head only, if entropy clips the box
regression range.

## Calibration set: 1000 images, seed 0, from val2017 and from train2017

Calibrating on val2017 leaks no labels, but the calibration distribution then
matches the test distribution exactly, which flatters INT8. So both: one
engine calibrated on 1000 val2017 images, one on 1000 train2017 images,
same seed, both benchmarked on the full val2017 split. The gap between them
is the honest estimate of how much the val2017 calibration flatters.
Rejected: val2017 only (cheaper, but the flattering is unmeasured).

## Quantisation scheme recorded as `mixed`

TensorRT implicit quantisation gives per-tensor activation scales from the
cache and per-channel symmetric weight scales from the builder. There is no
knob to make activations per-channel. "Which layers lose the most to
quantisation" is therefore answered by per-block precision sensitivity, not
by switching scale granularity.

## Sensitivity sweep: `ILayer.precision` pins, one block at a time

`scripts/sensitivity.py` builds one int8 engine per `model.N` block with that
block's layers forced to compute in fp16 (`OBEY_PRECISION_CONSTRAINTS`, so a
pin the builder cannot honour fails loudly instead of silently running
int8), evaluates each on the same 500-image subset, and ranks blocks by mAP
recovered. Only `ILayer.precision` is set, not the output type: the block's
outputs may still be re-quantised to int8 for the next block, so each
measurement isolates one block's arithmetic rather than also changing what
its neighbours receive. The
blocks that recover the most go into the final mixed engine via
`build_engine --pin-fp16`, which is then benchmarked on the full split.
Rejected for now: explicit quantisation with Q/DQ nodes through NVIDIA
ModelOpt. It gives per-layer control over both scales and precision and is
the stronger answer, but it is a second toolchain and a week of its own.
Named as future work, not skipped silently.

## INT8 engines keep the FP16 flag

Layers TensorRT refuses to run in int8 fall back to fp16 instead of fp32.
Matches how DeepStream-Yolo and trtexec `--int8 --fp16` build. Cost: the
"INT8" engine is really int8-with-fp16-fallback and the README has to say so.

## Latency: both the copy-inclusive span and execute-only

`latency_ms` in the row spans host-to-device copy, execute, device-to-host
copy and a sync, because that is what an application pays per frame. CUDA
events around the execute call alone give the figure trtexec reports and
the published numbers use; it lives in `meta.latency.gpu_*` and the report
table shows its p50 next to the inclusive one. Rejected: picking one. The
inclusive number is the honest one, the execute-only number is the
comparable one, and the difference is itself a measurement of copy cost on
a unified-memory board.

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
