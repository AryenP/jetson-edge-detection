from bench.env import parse_nv_tegra_release, parse_nvpmodel


def test_parse_nvpmodel():
    assert parse_nvpmodel("NV Fan Mode:quiet\nNV Power Mode: 15W\n0\n") == "15W (mode 0)"
    assert parse_nvpmodel("NV Power Mode: MAXN_SUPER\n2") == "MAXN_SUPER (mode 2)"
    assert parse_nvpmodel("garbage") is None


def test_parse_nv_tegra_release():
    text = "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025"
    assert parse_nv_tegra_release(text) == "R36.4.3"
    assert parse_nv_tegra_release("") is None
