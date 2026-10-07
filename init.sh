#!/usr/bin/env bash
# Entry point for every step of the pipeline. Each step is a thin wrapper over
# a Python module so the flags stay discoverable with --help.
set -euo pipefail
cd "$(dirname "$0")"

ONNX="${ONNX:-models/yolov8n_640.onnx}"

case "${1:-help}" in
  env)        # what the board is running; every results row carries this
    python3 -m bench.env
    ;;
  test)       # unit tests; run on any machine, no GPU needed
    python3 -m pytest -q "${@:2}"
    ;;
  coco)       # annotations only by default; ./init.sh coco all for the 5000-image val2017 set
    python3 -m scripts.fetch_coco --images "${2:-0}"
    ;;
  export)     # YOLOv8n -> ONNX (needs ultralytics + torch; laptop is fine)
    python3 -m scripts.export_onnx --weights "${2:-yolov8n.pt}" --out "$ONNX"
    ;;
  calib)      # seeded 1000-image calibration subset + manifest
    python3 -m scripts.prepare_calib --n "${2:-1000}" --seed "${3:-0}"
    ;;
  engine)     # ./init.sh engine fp16|int8  (on the Jetson)
    prec="${2:?usage: ./init.sh engine fp16|int8}"
    python3 -m scripts.build_engine --onnx "$ONNX" --precision "$prec" "${@:3}"
    ;;
  bench)      # ./init.sh bench fp16|int8 [extra bench.run flags] -> appends to results.json
    prec="${2:?usage: ./init.sh bench fp16|int8 [flags]}"
    python3 -m bench.run --engine "engines/$(basename "${ONNX%.onnx}")_${prec}.engine" --out results.json "${@:3}"
    ;;
  bench-cpu)  # pipeline dry run with onnxruntime on this machine; nothing is recorded
    python3 -m bench.run --engine "$ONNX" --no-power --iters 20 --warmup 5 --dry-run "${@:2}"
    ;;
  report)     # markdown table of results.json
    python3 -m bench.results
    ;;
  power)      # ad-hoc tegrastats log outside the benchmark (bench.run samples on its own)
    tegrastats --interval 100 --logfile /tmp/tegrastats.log &
    echo "sampling to /tmp/tegrastats.log; pkill tegrastats to stop"
    ;;
  *)
    sed -n '/^case/,/^esac/p' "$0" | grep -E '^\s+[a-z|-]+\)' | sed -E 's/^\s+([a-z|-]+)\)\s*#?\s*/  \1\t/'
    [ "${1:-help}" = help ] || exit 1
    ;;
esac
