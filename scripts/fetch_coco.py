import argparse
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prepare_calib import sample_files

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_URL = "http://images.cocodataset.org/zips/{}.zip"
IMG_URL = "http://images.cocodataset.org/{}/{}"


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    part.replace(dst)


def annotations(root, split):
    ann = root / "annotations" / f"instances_{split}.json"
    if ann.exists():
        return ann
    z = root / "annotations_trainval2017.zip"
    if not z.exists():
        print(f"downloading {ANN_URL}")
        download(ANN_URL, z)
    with zipfile.ZipFile(z) as zf:
        zf.extract(f"annotations/instances_{split}.json", root)
    return ann


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/coco")
    ap.add_argument("--split", choices=("val2017", "train2017"), default="val2017")
    ap.add_argument("--images", default="0", help="'all' for the split's zip (val 780 MB, train 18 GB), or N seeded files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    ann = annotations(root, args.split)
    img_dir = root / args.split
    if args.images == "all":
        z = root / f"{args.split}.zip"
        if not z.exists():
            print(f"downloading {ZIP_URL.format(args.split)}")
            download(ZIP_URL.format(args.split), z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        print(f"extracted {img_dir}")
    elif int(args.images) > 0:
        names = [im["file_name"] for im in json.loads(ann.read_text())["images"]]
        todo = [n for n in sample_files(names, int(args.images), args.seed) if not (img_dir / n).exists()]
        print(f"fetching {len(todo)} images into {img_dir}")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda n: download(IMG_URL.format(args.split, n), img_dir / n), todo))
    else:
        print(f"annotations at {ann}")
