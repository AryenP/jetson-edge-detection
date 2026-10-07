import argparse
import json
from pathlib import Path

PRECISIONS = ("fp16", "int8")
CALIB_SCHEMES = ("per-tensor", "per-channel", "mixed")
POWER_SOURCES = ("tegrastats", "external_meter")

SCHEMA = {
    "run_id": "string",
    "timestamp": "iso8601",
    "board": "string",
    "jetpack": "string",
    "tensorrt": "string",
    "model": "string",
    "input_res": "int",
    "precision": "fp16 | int8",
    "calib_imgs": "int | null",
    "calib_batch_size": "int | null",
    "calib_scheme": "per-tensor | per-channel | mixed | null",
    "nvpmodel_mode": "string",
    "jetson_clocks": "bool",
    "n_warmup_discarded": "int",
    "latency_ms": {"p50": "float", "p95": "float"},
    "fps": "float",
    "map_50_95": "float",
    "power_w": {"mean": "float", "source": "tegrastats | external_meter"},
}


class SchemaError(ValueError):
    pass


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(row):
    missing = [k for k in SCHEMA if k not in row]
    if missing:
        raise SchemaError(f"missing fields: {missing}")
    unknown = [k for k in row if k not in SCHEMA and k != "meta"]
    if unknown:
        raise SchemaError(f"unknown fields: {unknown}; extras go under 'meta'")

    def need(cond, msg):
        if not cond:
            raise SchemaError(msg)

    for key in ("run_id", "timestamp", "board", "jetpack", "tensorrt", "model", "nvpmodel_mode"):
        need(isinstance(row[key], str) and row[key], f"{key} must be a non-empty string")
    need(isinstance(row["input_res"], int) and row["input_res"] > 0, "input_res must be a positive int")
    need(row["precision"] in PRECISIONS, f"precision must be one of {PRECISIONS}")
    for key in ("calib_imgs", "calib_batch_size"):
        need(row[key] is None or (isinstance(row[key], int) and row[key] > 0), f"{key} must be a positive int or null")
    need(row["calib_scheme"] is None or row["calib_scheme"] in CALIB_SCHEMES, f"calib_scheme must be one of {CALIB_SCHEMES} or null")
    if row["precision"] == "int8":
        need(row["calib_imgs"] is not None and row["calib_scheme"] is not None, "int8 rows must record calibration")
    need(isinstance(row["jetson_clocks"], bool), "jetson_clocks must be a bool")
    need(isinstance(row["n_warmup_discarded"], int) and row["n_warmup_discarded"] >= 0, "n_warmup_discarded must be an int >= 0")
    lat = row["latency_ms"]
    need(isinstance(lat, dict) and is_num(lat.get("p50")) and is_num(lat.get("p95")), "latency_ms needs numeric p50 and p95")
    need(lat["p95"] >= lat["p50"] > 0, "latency p95 must be >= p50 > 0")
    need(is_num(row["fps"]) and row["fps"] > 0, "fps must be positive")
    need(is_num(row["map_50_95"]) and 0.0 <= row["map_50_95"] <= 1.0, "map_50_95 must be in [0, 1]")
    pw = row["power_w"]
    need(isinstance(pw, dict) and is_num(pw.get("mean")) and pw.get("source") in POWER_SOURCES, f"power_w needs numeric mean and source in {POWER_SOURCES}")
    if "meta" in row:
        need(isinstance(row["meta"], dict), "meta must be an object")


def load(path):
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data.setdefault("runs", [])
    data["schema"] = SCHEMA
    return data


def append(path, row):
    validate(row)
    data = load(path)
    if any(r.get("run_id") == row["run_id"] for r in data["runs"]):
        raise SchemaError(f"run_id {row['run_id']!r} already in {path}")
    data["runs"].append(row)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def report(data):
    # sort so fp16 and int8 of the same model/mode sit on adjacent rows
    runs = sorted(data.get("runs", []), key=lambda r: (r["model"], r["input_res"], r["nvpmodel_mode"], r["precision"], r["timestamp"]))
    if not runs:
        return "_no runs yet_"
    lines = [
        "| run | board | JetPack / TRT | model | res | precision | calib | nvpmodel | clocks | p50 ms | p95 ms | exec p50 ms | FPS | mAP50-95 | power W |",
        "|" + "---|" * 15,
    ]
    for r in runs:
        meta = r.get("meta", {})
        calib = "-"
        if r["precision"] == "int8":
            calib = f"{r['calib_imgs']} {meta.get('calib_split', '')} img, bs {r['calib_batch_size']}, {r['calib_scheme']}".replace("  ", " ")
            if meta.get("pin_fp16"):
                calib += f", fp16: {','.join(meta['pin_fp16'])}"
        gpu = meta.get("latency", {}).get("gpu_p50_ms")
        gpu = f"{gpu:.2f}" if gpu is not None else "-"
        lines.append(
            f"| {r['run_id']} | {r['board']} | {r['jetpack']} / {r['tensorrt']} | {r['model']} | {r['input_res']} | {r['precision']} | {calib} "
            f"| {r['nvpmodel_mode']} | {'on' if r['jetson_clocks'] else 'off'} | {r['latency_ms']['p50']:.2f} | {r['latency_ms']['p95']:.2f} | {gpu} "
            f"| {r['fps']:.1f} | {r['map_50_95']:.3f} | {r['power_w']['mean']:.2f} ({r['power_w']['source']}) |"
        )
    return "\n".join(lines)


def pairs(data):
    """(fp16 row, int8 row) for every model / res / mode that has both, latest run of each."""
    latest = {}
    for r in data.get("runs", []):
        key = (r["model"], r["input_res"], r["nvpmodel_mode"], r["precision"], r.get("meta", {}).get("calib_split"), tuple(r.get("meta", {}).get("pin_fp16") or ()))
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    out = []
    for key, int8 in latest.items():
        if int8["precision"] != "int8":
            continue
        fp16 = latest.get((key[0], key[1], key[2], "fp16", None, ()))
        if fp16:
            out.append((fp16, int8))
    return out


def tradeoffs(data):
    """What INT8 buys and costs against the FP16 row from the same model and power mode."""
    rows = pairs(data)
    if not rows:
        return "_no fp16/int8 pair yet_"
    lines = [
        "| model | res | nvpmodel | int8 calib | speedup (p50) | speedup (exec) | mAP50-95 fp16 -> int8 | power W fp16 -> int8 |",
        "|" + "---|" * 8,
    ]
    for fp16, int8 in rows:
        m = int8.get("meta", {})
        calib = f"{m.get('calib_split', '?')}" + (f", fp16: {','.join(m['pin_fp16'])}" if m.get("pin_fp16") else "")
        g16, g8 = fp16.get("meta", {}).get("latency", {}).get("gpu_p50_ms"), m.get("latency", {}).get("gpu_p50_ms")
        exec_speedup = f"{g16 / g8:.2f}x" if g16 and g8 else "-"
        dmap = int8["map_50_95"] - fp16["map_50_95"]
        dpw = int8["power_w"]["mean"] - fp16["power_w"]["mean"]
        pct = f" ({dpw / fp16['power_w']['mean'] * 100:+.1f}%)" if fp16["power_w"]["mean"] else ""
        lines.append(
            f"| {int8['model']} | {int8['input_res']} | {int8['nvpmodel_mode']} | {calib} | {fp16['latency_ms']['p50'] / int8['latency_ms']['p50']:.2f}x | {exec_speedup} "
            f"| {fp16['map_50_95']:.4f} -> {int8['map_50_95']:.4f} ({dmap:+.4f}) | {fp16['power_w']['mean']:.2f} -> {int8['power_w']['mean']:.2f}{pct} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results.json")
    data = load(ap.parse_args().path)
    print(report(data))
    print()
    print(tradeoffs(data))
