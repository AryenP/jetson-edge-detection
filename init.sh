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
  coco)       # ./init.sh coco [all|N] [val2017|train2017]; annotations only by default
    python3 -m scripts.fetch_coco --images "${2:-0}" --split "${3:-val2017}"
    ;;
  export)     # YOLOv8n -> ONNX (needs ultralytics + torch; laptop is fine)
    python3 -m scripts.export_onnx --weights "${2:-yolov8n.pt}" --out "$ONNX"
    ;;
  calib)      # ./init.sh calib [val2017|train2017]: seeded 1000-image subset + manifest
    python3 -m scripts.prepare_calib --split "${2:-val2017}"
    ;;
  quantize)   # ./init.sh quantize [val2017|train2017]: explicit int8 via modelopt -> models/<stem>_int8qdq_<split>.onnx (laptop)
    python3 -m scripts.quantize_onnx --onnx "$ONNX" --calib-split "${2:-val2017}" "${@:3}"
    ;;
  engine)     # ./init.sh engine fp16 | int8 [split] [--pin-fp16 model.22,...] | int8qdq [split]
    prec="${2:?usage: ./init.sh engine fp16|int8|int8qdq [split] [flags]}"
    case "$prec" in
      int8)    python3 -m scripts.build_engine --onnx "$ONNX" --precision int8 --calib-split "${3:-val2017}" "${@:4}" ;;
      int8qdq) python3 -m scripts.build_engine --onnx "${ONNX%.onnx}_int8qdq_${3:-val2017}.onnx" --precision int8 "${@:4}" ;;
      *)       python3 -m scripts.build_engine --onnx "$ONNX" --precision fp16 "${@:3}" ;;
    esac
    ;;
  bench)      # ./init.sh bench fp16 | int8 [split] | int8qdq [split] [bench.run flags] -> results.json
    prec="${2:?usage: ./init.sh bench fp16|int8|int8qdq [split] [flags]}"
    stem="engines/$(basename "${ONNX%.onnx}")"
    if [ "$prec" = int8 ] || [ "$prec" = int8qdq ]; then
      python3 -m bench.run --engine "${stem}_${prec}_${3:-val2017}.engine" --out results.json "${@:4}"
    else
      python3 -m bench.run --engine "${stem}_fp16.engine" --out results.json "${@:3}"
    fi
    ;;
  sens)       # per-block fp16-pin sensitivity sweep; ranks blocks, reportable numbers come from bench
    python3 -m scripts.sensitivity --onnx "$ONNX" "${@:2}"
    ;;
  sweep)      # ./init.sh sweep "<mode ids>" "<model stems>": bench fp16 + int8 of each model in each mode
    modes="${2:?usage: ./init.sh sweep \"0 1\" \"yolov8n yolov8s\"}"
    for mode in $modes; do
      sudo nvpmodel -m "$mode" && sudo jetson_clocks
      sleep 10   # let clocks and temperature settle before the first timed loop
      for m in ${3:-yolov8n}; do
        ONNX="models/${m}_640.onnx" "$0" bench fp16
        ONNX="models/${m}_640.onnx" "$0" bench int8
      done
    done
    ;;
  plots)      # results.json -> docs/plots/*.png
    python3 -m scripts.plot
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
