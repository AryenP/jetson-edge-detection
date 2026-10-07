"""results.json -> two figures: latency vs power, and mAP vs latency. One point per run."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.results import load

COLOR = {"fp16": "#2a78d6", "int8": "#eb6834"}
MARKERS = "osD^v<>"


def label(r):
    m = r.get("meta", {})
    tag = r["model"].replace("_640", "")
    if r["precision"] == "int8" and m.get("calib_split"):
        tag += f" {m['calib_split'][:-4]}"
    if m.get("pin_fp16"):
        tag += " mixed"
    return tag


def scatter(ax, runs, xkey, ykey):
    modes = sorted({r["nvpmodel_mode"] for r in runs})
    for r in runs:
        ax.scatter(xkey(r), ykey(r), s=36, color=COLOR[r["precision"]], marker=MARKERS[modes.index(r["nvpmodel_mode"]) % len(MARKERS)], edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(label(r), (xkey(r), ykey(r)), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#52514e")
    ax.grid(color="#e8e7e3", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # legend carries precision by colour and power mode by marker; the labels carry the model
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=p) for p, c in COLOR.items()]
    handles += [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], linestyle="", color="#52514e", label=m) for i, m in enumerate(modes)]
    ax.legend(handles=handles, fontsize=7, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results.json")
    ap.add_argument("--out", default="docs/plots")
    args = ap.parse_args()
    runs = load(args.path)["runs"]
    if not runs:
        print("no runs")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    board = runs[0]["board"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    scatter(ax, runs, lambda r: r["latency_ms"]["p50"], lambda r: r["power_w"]["mean"])
    ax.set_xlabel("latency p50 (ms, batch 1, copies included)")
    ax.set_ylabel("power (W, mean over timed loop)")
    ax.set_title(f"latency vs power, {board}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "latency_power.png")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    scatter(ax, runs, lambda r: r["latency_ms"]["p50"], lambda r: r["map_50_95"])
    ax.set_xlabel("latency p50 (ms, batch 1, copies included)")
    ax.set_ylabel("COCO mAP50-95")
    ax.set_title(f"accuracy vs latency, {board}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "map_latency.png")
    print(f"wrote {out}/latency_power.png and map_latency.png from {len(runs)} runs")


if __name__ == "__main__":
    main()
