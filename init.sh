#!/usr/bin/env bash
set -euo pipefail

case "${1:-env}" in
  env)
    dpkg-query --show nvidia-jetpack 2>/dev/null || echo "nvidia-jetpack: not found"
    /usr/src/tensorrt/bin/trtexec --version 2>/dev/null | head -1 || echo "trtexec: not found"
    nvcc --version 2>/dev/null | tail -2 || echo "nvcc: not found"
    sudo nvpmodel -q
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || echo "torch: not installed"
    ;;
  export)
    python3 -m scripts.export_onnx
    ;;
  engine)
    prec="${2:?usage: ./init.sh engine fp16|int8}"
    python3 -m scripts.build_engine --precision "$prec"
    ;;
  bench)
    python3 -m bench.run --out results.json
    ;;
  power)
    tegrastats --interval 100 --logfile /tmp/tegrastats.log &
    echo "sampling to /tmp/tegrastats.log; pkill tegrastats to stop"
    ;;
  *) echo "usage: ./init.sh [env|export|engine fp16|int8|bench|power]"; exit 1 ;;
esac
