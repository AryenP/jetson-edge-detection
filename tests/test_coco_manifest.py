"""Checks against the official instances_val2017.json; skipped until it is fetched."""
import hashlib
import json
from pathlib import Path

import pytest

from bench.accuracy import COCO80_TO_91
from scripts.prepare_calib import sample_files

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"
MANIFEST = ROOT / "calib" / "val2017" / "manifest.json"
pytestmark = pytest.mark.skipif(not ANN.exists(), reason="run ./init.sh coco first")


def test_category_map_matches_official_annotations():
    ids = sorted(c["id"] for c in json.loads(ANN.read_text())["categories"])
    assert ids == COCO80_TO_91


def test_committed_manifest_is_reproducible_from_annotations():
    m = json.loads(MANIFEST.read_text())
    names = [im["file_name"] for im in json.loads(ANN.read_text())["images"]]
    assert sample_files(names, m["n"], m["seed"]) == m["files"]
    assert hashlib.sha256("\n".join(m["files"]).encode()).hexdigest() == m["files_sha256"]
