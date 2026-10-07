import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest


@pytest.fixture
def fake_trt(monkeypatch):
    from tests import fake_trt as f

    f.install(monkeypatch)
    return f
