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
    ap.add_argument("--ann", default="datasets/coco/annotations/instances_val2017.json")
    ap.add_argument("--images", default="datasets/coco/val2017")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default="calib/images")
    ap.add_argument("--manifest", default="calib/manifest.json")
    args = ap.parse_args()

    names = [im["file_name"] for im in json.loads(Path(args.ann).read_text())["images"]]
    src = Path(args.images)
    present = [n for n in names if (src / n).exists()]
    if len(present) < args.n:
        print(f"warning: only {len(present)} of {len(names)} val2017 images under {src}")
    chosen = sample_files(present, args.n, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        dst = out_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink((src / name).resolve(), dst)
    m = write_manifest(Path(args.manifest), chosen, str(src), args.seed, args.imgsz)
    print(f"{m['n']} images -> {out_dir}; manifest {args.manifest} ({m['files_sha256'][:12]})")
