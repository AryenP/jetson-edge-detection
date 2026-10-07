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
stores them in every results row. `pycuda` and `tegrastats` must both read
true before a benchmark means anything.

## 2. Power mode and clocks

Numbers are meaningless without these two settings, which is why both are
schema fields.

JetPack 6.2 on the Orin Nano 8GB offers 15W, 25W and MAXN SUPER (uncapped;
throttles when the module exceeds its thermal budget). The 7W mode belongs
to the pre-6.2 configs. Mode ids are not fixed across JetPack versions, so
read them off the board rather than a table:

```
sudo nvpmodel -q                      # current mode name and id
grep -E "POWER_MODEL|NAME" /etc/nvpmodel.conf
sudo nvpmodel -m <id>
sudo jetson_clocks                    # pin clocks to max for the current mode (until reboot)
sudo jetson_clocks --show             # verify
```

MAXN SUPER under sustained load is a thermal experiment, not a power mode.
Report 15W and 25W as the primary rows and MAXN SUPER with its p95/p50 ratio
and the temperature from tegrastats.

`bench.run` records the mode string `nvpmodel -q` prints, name and id. It
detects `jetson_clocks` by checking whether the GPU devfreq
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
./init.sh coco all             # ~780 MB zip + annotations into datasets/coco/
./init.sh calib                # symlinks the committed calib/val2017/manifest.json set
./init.sh coco 1000 train2017  # 1000 seeded train2017 files, not the 18 GB zip
./init.sh calib train2017      # -> calib/train2017/
```

The mAP run reads the full 5000-image set. `--accuracy-limit N` exists for
smoke tests only; a reported mAP must be on the full split.

## 6. Engines

```
./init.sh engine fp16
./init.sh engine int8              # calibrates on val2017 on first build, caches to calib/val2017/
./init.sh engine int8 train2017
```

Engines are tied to the GPU and the TensorRT version; rebuild after any
JetPack upgrade. Each build writes `engines/<name>.engine.json` with the ONNX
hash, TensorRT version, build time and (for INT8) the calibration manifest hash.
Commit those sidecars; the engines themselves are gitignored.

## 7. Benchmark

```
./init.sh bench fp16
./init.sh bench int8
./init.sh bench int8 train2017
./init.sh report
```

`bench.run` warms up, times 500 iterations with tegrastats sampling at 100 ms
in the background, then runs the mAP evaluation and appends one validated row
to `results.json`. With an external USB power meter in line, read its mean
over the timed loop and pass `--external-meter-w <W>`; the row then records
the meter as the reference and keeps the tegrastats figure in `meta`.

## 8. Model and power-mode sweep

```
for m in yolov8n yolov8s yolov8m yolov8l; do ONNX=models/${m}_640.onnx ./init.sh engine fp16; done
./init.sh sweep "0 1" "yolov8n yolov8s yolov8m yolov8l"   # mode ids from nvpmodel -q
```

`sweep` sets each mode, re-applies `jetson_clocks`, and benches every listed
model's fp16 and int8 engines. Engines are built once per model; TensorRT
engines do not depend on the power mode.

## 9. Sensitivity sweep

```
./init.sh sens --limit 500                    # ~25 engine builds; resumable, hours on an Orin Nano
./init.sh engine int8 val2017 --pin-fp16 model.22,model.21
./init.sh bench int8 val2017 --engine engines/yolov8n_640_int8_val2017_fp16-model.22-model.21.engine
```

The ranking in `runs/sensitivity.json` comes from a 500-image subset and is
not a reported number. Only the final mixed engine on the full split is.

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
