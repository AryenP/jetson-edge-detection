import argparse
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prepare_calib import sample_files

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    part.replace(dst)


def annotations(root):
    ann = root / "annotations" / "instances_val2017.json"
    if ann.exists():
        return ann
    z = root / "annotations_trainval2017.zip"
    if not z.exists():
        print(f"downloading {ANN_URL}")
        download(ANN_URL, z)
    with zipfile.ZipFile(z) as zf:
        zf.extract("annotations/instances_val2017.json", root)
    return ann


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/coco")
    ap.add_argument("--images", default="0", help="'all' for the val2017 zip, or N seeded files for a smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    ann = annotations(root)
    img_dir = root / "val2017"
    if args.images == "all":
        z = root / "val2017.zip"
        if not z.exists():
            print(f"downloading {VAL_URL}")
            download(VAL_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        print(f"extracted {img_dir}")
    elif int(args.images) > 0:
        names = [im["file_name"] for im in json.loads(ann.read_text())["images"]]
        todo = [n for n in sample_files(names, int(args.images), args.seed) if not (img_dir / n).exists()]
        print(f"fetching {len(todo)} images into {img_dir}")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda n: download(IMG_URL.format(n), img_dir / n), todo))
    else:
        print(f"annotations at {ann}")
