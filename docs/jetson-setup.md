# Jetson Orin Nano setup

Everything the benchmark depends on, in the order it has to happen. Written
before the board arrived so the first session on hardware is a checklist, not
a research project.

## 1. Flash JetPack 6.x

Target JetPack 6.2 (L4T R36.4.x, CUDA 12.6, TensorRT 10.3). Use NVIDIA SDK
Manager from an x86 Ubuntu host, or the pre-built SD image for the developer
kit. After first boot:

```
sudo apt update && sudo apt install -y nvidia-jetpack python3-pip python3-venv
./init.sh env          # should show jetpack, trtexec, nvcc, torch (if installed)
```

Record what `./init.sh env` prints; `bench.run` reads the same probes and
stores them in every results row.

## 2. Power mode and clocks

Numbers are meaningless without these two settings, which is why both are
schema fields.

| nvpmodel mode | Orin Nano 8GB | note |
|---|---|---|
| 0 | 15W | default on JetPack 6.0/6.1 |
| 1 | 7W | |
| 2 | 25W (MAXN SUPER) | JetPack 6.2 "Super" mode, higher GPU/memory clocks |

```
sudo nvpmodel -q            # current mode
sudo nvpmodel -m 0          # set 15W
sudo jetson_clocks          # pin clocks to max for the current mode (until reboot)
sudo jetson_clocks --show   # verify
```

`bench.run` detects `jetson_clocks` by checking whether the GPU devfreq
min_freq equals max_freq; override with `--jetson-clocks yes|no` if the probe
disagrees with what you did. Run the FP16 and INT8 benchmarks back to back
under the same mode, and repeat the whole set per mode you care about.

## 3. Python dependencies

```
export PATH=/usr/local/cuda/bin:$PATH      # pycuda build needs nvcc
pip3 install -r requirements-jetson.txt
python3 -c "import tensorrt, pycuda.autoinit; print(tensorrt.__version__)"
```

`tensorrt` comes from JetPack's `python3-libnvinfer` package. Do not `pip
install tensorrt`; it will pull an x86 build.

## 4. Model

The ONNX export is device-independent, so do it wherever torch is easiest
(laptop) and copy `models/yolov8n_640.onnx` plus its `.onnx.json` sidecar to
the board. Exporting on the board works too but needs NVIDIA's torch wheel for
JetPack, which is a separate download.

## 5. COCO val2017

```
./init.sh coco all     # ~780 MB zip + annotations into datasets/coco/
./init.sh calib        # seeded 1000-image subset -> calib/images + calib/manifest.json
```

The mAP run reads the full 5000-image set. `--accuracy-limit N` exists for
smoke tests only; a reported mAP must be on the full split.

## 6. Engines

```
./init.sh engine fp16
./init.sh engine int8          # runs calibration on first build, caches to calib/*.cache
```

Engines are tied to the GPU and the TensorRT version; rebuild after any
JetPack upgrade. Each build writes `engines/<name>.engine.json` with the ONNX
hash, TensorRT version, build time and (for INT8) the calibration manifest hash.
Commit those sidecars; the engines themselves are gitignored.

## 7. Benchmark

```
./init.sh bench fp16
./init.sh bench int8
./init.sh report
```

`bench.run` warms up, times 500 iterations with tegrastats sampling at 100 ms
in the background, then runs the mAP evaluation and appends one validated row
to `results.json`. With an external USB power meter in line, read its mean
over the timed loop and pass `--external-meter-w <W>`; the row then records
the meter as the reference and keeps the tegrastats figure in `meta`.

## Known traps

- `tegrastats` rail names differ by module (`VDD_IN` on Orin Nano/NX,
  `VIN_SYS_5V0` on AGX Orin). The sampler picks whichever is present; pass
  `--power-rail` to force one.
- `VDD_IN` is the module's draw, not the wall draw. Carrier board, fan and USB
  peripherals are not in it. State which one a number is.
- INT8 with no calibration cache makes TensorRT silently use a fixed dynamic
  range; the engine builder here refuses to build INT8 without a manifest.
- Thermal throttling shows up as a rising p95. If p95/p50 > 1.3, check
  `tegrastats` temperatures and let the board cool before trusting the row.
