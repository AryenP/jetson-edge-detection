import glob
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

UNKNOWN = "unknown"


def run(cmd):
    if shutil.which(cmd[0]) is None:
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ((res.stdout or "") + (res.stderr or "")).strip() or None


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return None


def dpkg_version(pkg):
    out = run(["dpkg-query", "--show", "--showformat=${Version}", pkg])
    if not out or out.startswith("dpkg-query"):
        return UNKNOWN
    return out.split("+")[0]


def parse_nv_tegra_release(text):
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", text)
    return f"R{m.group(1)}.{m.group(2)}" if m else None


def tensorrt_version():
    try:
        import tensorrt

        return tensorrt.__version__
    except ImportError:
        return dpkg_version("libnvinfer10")


def cuda_version():
    m = re.search(r"release\s+([\d.]+)", run(["nvcc", "--version"]) or "")
    return m.group(1) if m else UNKNOWN


def parse_nvpmodel(text):
    name = re.search(r"NV Power Mode:\s*(.+)", text)
    if not name:
        return None
    mode = re.search(r"^\s*(\d+)\s*$", text, re.MULTILINE)
    label = name.group(1).strip()
    return f"{label} (mode {mode.group(1)})" if mode else label


def nvpmodel_mode():
    # unprivileged -q works on JetPack 6; sudo -n so a password prompt can't hang the run
    for cmd in (["nvpmodel", "-q"], ["sudo", "-n", "nvpmodel", "-q"]):
        parsed = parse_nvpmodel(run(cmd) or "")
        if parsed:
            return parsed
    return UNKNOWN


def jetson_clocks_on():
    """jetson_clocks pins the GPU devfreq governor, so min_freq == max_freq. None off a Jetson."""
    for node in glob.glob("/sys/class/devfreq/*"):
        name = read(os.path.join(node, "name")) or node
        if not any(k in name for k in ("gpu", "ga10b", "gv11b")):
            continue
        lo, hi = read(os.path.join(node, "min_freq")), read(os.path.join(node, "max_freq"))
        if lo and hi and lo.isdigit() and hi.isdigit():
            return int(lo) == int(hi)
    return None


def importable(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def collect():
    return {
        "board": read("/proc/device-tree/model") or UNKNOWN,
        "jetpack": dpkg_version("nvidia-jetpack"),
        "l4t": parse_nv_tegra_release(read("/etc/nv_tegra_release") or "") or UNKNOWN,
        "tensorrt": tensorrt_version(),
        "cuda": cuda_version(),
        "nvpmodel_mode": nvpmodel_mode(),
        "jetson_clocks": jetson_clocks_on(),
        "pycuda": importable("pycuda.driver"),
        "tegrastats": shutil.which("tegrastats") is not None,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "kernel": platform.release(),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
