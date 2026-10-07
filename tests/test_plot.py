import copy
import json
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")
from tests.test_results import GOOD


def test_plot_writes_both_figures(tmp_path):
    int8 = {**copy.deepcopy(GOOD), "run_id": "i8", "precision": "int8", "calib_imgs": 1000, "calib_batch_size": 8, "calib_scheme": "mixed",
            "nvpmodel_mode": "25W (mode 1)", "meta": {"calib_split": "val2017", "pin_fp16": ["model.22"]}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"runs": [GOOD, int8]}))
    res = subprocess.run([sys.executable, "-m", "scripts.plot", "--path", str(path), "--out", str(tmp_path / "plots")], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "plots" / "latency_power.png").stat().st_size > 1000
    assert (tmp_path / "plots" / "map_latency.png").stat().st_size > 1000


def test_plot_without_runs_writes_nothing(tmp_path):
    res = subprocess.run([sys.executable, "-m", "scripts.plot", "--path", str(tmp_path / "none.json"), "--out", str(tmp_path / "plots")], capture_output=True, text=True)
    assert res.returncode == 0 and "no runs" in res.stdout and not (tmp_path / "plots").exists()
