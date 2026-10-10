# Progress

Kept current by hand. Dates are when the state changed, not when it was written up.

## Done

- 2026-10-07: export, letterbox, decode and NMS, COCO mAP eval, results schema, tradeoff report, plots. Checked against ultralytics on CPU: boxes within 2 px, confidences within 0.01.
- 2026-10-07: TensorRT FP16 and INT8 builders, entropy calibrator on seeded val2017 and train2017 subsets, fp16 block pins, engine runner with CUDA-event timing, tegrastats sampler with temperatures. Tested against an in-memory fake of tensorrt and pycuda; not yet run on a board.
- 2026-10-07: env probe doubles as preflight (JetPack, TensorRT, nvpmodel, jetson_clocks, pycuda, tegrastats). CI runs the tests on every push.
- 2026-10-08: explicit INT8 through ModelOpt Q/DQ as a third engine. Remainder kept in fp32 after the fp16 remainder collapsed coco128 mAP (see DECISIONS, Reversals).
- 2026-10-08: sensitivity sweep runs on a laptop through ModelOpt exclusions; first pass on coco128 recorded. Head in float recovers the gap.
- 2026-10-08: calibration set manifest pinned (1000 images, seed 0).

## Next

- Run `init.sh env` on the board and paste the output here before anything else.
- FP16 baseline: latency, fps, power, val2017 mAP at the default nvpmodel mode. Record before any INT8 work.
- INT8 entropy engines from the val2017 and train2017 calibration sets, same bench.
- INT8 Q/DQ engine, same bench. Three INT8 rows side by side.
- Per-block fp16-pin sweep on the board, then the mixed engine the sweep picks.
- Power at each nvpmodel mode, with and without jetson_clocks.
- Fill the README results table from results.json only.

## Blocked

- No board. Loan requests out to UCSD labs since 2026-10-07; Triton AI said not before 2026-11-15. The alternative platforms (Pi 5 with Hailo-8, original Nano) are open decisions in DECISIONS.md.
- Everything under Next except the first item waits on the board. The CPU-side pipeline has no remaining work that does not need one.

## Why

Aryen's own words, one line per decision that mattered. Left empty on purpose until he writes them.

-
