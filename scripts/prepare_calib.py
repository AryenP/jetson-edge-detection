import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path


def sample_files(names, n, seed):
    names = sorted(names)
    if n >= len(names):
        return names
    return sorted(random.Random(seed).sample(names, n))


def write_manifest(path, files, source, seed, imgsz):
    manifest = {
        "n": len(files),
        "seed": seed,
        "imgsz": imgsz,
        "source": source,
        "files_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val2017", "train2017"), default="val2017")
    ap.add_argument("--coco", default="datasets/coco")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    ann = Path(args.coco) / "annotations" / f"instances_{args.split}.json"
    src = Path(args.coco) / args.split
    # sample from the annotation file, not from whatever is on disk, so the manifest is a
    # property of the dataset and seed and comes out identical on every machine
    names = [im["file_name"] for im in json.loads(ann.read_text())["images"]]
    chosen = sample_files(names, args.n, args.seed)
    out_dir = Path("calib") / args.split / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in chosen:
        dst = out_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if (src / name).exists():
            os.symlink((src / name).resolve(), dst)
        else:
            missing += 1
    manifest = out_dir.parent / "manifest.json"
    m = write_manifest(manifest, chosen, f"instances_{args.split}.json", args.seed, args.imgsz)
    print(f"{m['n']} images -> {out_dir}; manifest {manifest} ({m['files_sha256'][:12]})")
    if missing:
        print(f"warning: {missing} of {m['n']} images not under {src}; fetch them before calibrating")
