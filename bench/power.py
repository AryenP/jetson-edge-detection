import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

# "<RAIL> <now>mW/<avg>mW". The avg is since tegrastats started, not our window; ignore it.
RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW/(\d+)mW")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
TEMP_RE = re.compile(r"\b(cpu|gpu|tj|soc\d)@([\d.]+)C")

# module input rail per board: Orin Nano/NX, AGX Orin, Xavier NX. None of these is wall draw.
TOTAL_RAILS = ("VDD_IN", "VIN_SYS_5V0", "POM_5V_IN")


def parse_line(line):
    out = {m.group(1): float(m.group(2)) for m in RAIL_RE.finditer(line)}
    gr3d = GR3D_RE.search(line)
    if gr3d:
        out["GR3D_FREQ_PCT"] = float(gr3d.group(1))
    for m in TEMP_RE.finditer(line):
        out[f"temp_{m.group(1)}_c"] = float(m.group(2))
    return out


def total_rail(samples):
    seen = set().union(*samples) if samples else set()
    return next((r for r in TOTAL_RAILS if r in seen), None)


@dataclass
class PowerSummary:
    rail: str
    mean_w: float
    n_samples: int
    duration_s: float
    min_w: float
    max_w: float
    gpu_load_pct: float | None = None
    tj_max_c: float | None = None


def summarize(samples, times, rail):
    mw = [s[rail] for s in samples if rail in s]
    if not mw:
        seen = sorted(set().union(*samples)) if samples else []
        raise ValueError(f"rail {rail!r} not in tegrastats output; saw {seen}")
    gpu = [s["GR3D_FREQ_PCT"] for s in samples if "GR3D_FREQ_PCT" in s]
    tj = [s["temp_tj_c"] for s in samples if "temp_tj_c" in s]
    duration = times[-1] - times[0] if times and len(times) > 1 else 0.0
    return PowerSummary(rail, sum(mw) / len(mw) / 1000, len(mw), duration, min(mw) / 1000, max(mw) / 1000, sum(gpu) / len(gpu) if gpu else None, max(tj) if tj else None)


@dataclass
class Tegrastats:
    """Runs tegrastats for the length of a `with` block and keeps every sample."""

    interval_ms: int = 100
    samples: list = field(default_factory=list)
    times: list = field(default_factory=list)
    proc: subprocess.Popen | None = None
    thread: threading.Thread | None = None

    @staticmethod
    def available():
        return shutil.which("tegrastats") is not None

    def reader(self):
        for raw in self.proc.stdout:
            parsed = parse_line(raw.decode(errors="replace"))
            if parsed:
                self.samples.append(parsed)
                self.times.append(time.monotonic())

    def __enter__(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self.reader, daemon=True)
        self.thread.start()
        # first sample lands ~1 interval after start; wait so the window is covered from its start
        time.sleep(self.interval_ms / 1000 * 2)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.thread.join(timeout=2)

    def summary(self, rail=None):
        rail = rail or total_rail(self.samples) or TOTAL_RAILS[0]
        return summarize(self.samples, self.times, rail)
