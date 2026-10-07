import pytest

from bench.power import parse_line, summarize, total_rail

ORIN_NANO = (
    "10-06-2026 12:00:01 RAM 2493/7620MB (lfb 1x4MB) SWAP 0/3810MB (cached 0MB) CPU [5%@729,3%@729,0%@729,0%@729,0%@729,1%@729] "
    "GR3D_FREQ 37% cpu@45.5C soc2@44.1C soc0@44.5C gpu@44.2C tj@45.5C soc1@44C "
    "VDD_IN 4542mW/4000mW VDD_CPU_GPU_CV 495mW/400mW VDD_SOC 1286mW/1200mW"
)
AGX_ORIN = (
    "RAM 3000/30000MB (lfb 1x1MB) SWAP 0/15000MB CPU [1%@2201,...] GR3D_FREQ 0% "
    "VDD_GPU_SOC 4876mW/4876mW VDD_CPU_CV 975mW/975mW VIN_SYS_5V0 4461mW/4461mW"
)


def test_parse_orin_nano_rails_use_instantaneous_value():
    p = parse_line(ORIN_NANO)
    assert p["VDD_IN"] == 4542 and p["VDD_CPU_GPU_CV"] == 495 and p["VDD_SOC"] == 1286
    assert p["GR3D_FREQ_PCT"] == 37
    assert p["temp_tj_c"] == 45.5 and p["temp_gpu_c"] == 44.2 and p["temp_soc1_c"] == 44.0


def test_parse_agx_orin_rails():
    p = parse_line(AGX_ORIN)
    assert p["VIN_SYS_5V0"] == 4461 and p["VDD_GPU_SOC"] == 4876


def test_total_rail_per_board():
    assert total_rail([parse_line(ORIN_NANO)]) == "VDD_IN"
    assert total_rail([parse_line(AGX_ORIN)]) == "VIN_SYS_5V0"
    assert total_rail([]) is None


def test_summarize_mean_in_watts():
    samples = [{"VDD_IN": 4000.0, "GR3D_FREQ_PCT": 50.0}, {"VDD_IN": 6000.0, "GR3D_FREQ_PCT": 100.0}, {"VDD_IN": 5000.0}]
    samples[0]["temp_tj_c"], samples[2]["temp_tj_c"] = 61.0, 70.5
    s = summarize(samples, [0.0, 0.1, 0.2], "VDD_IN")
    assert s.mean_w == 5.0 and s.n_samples == 3 and s.min_w == 4.0 and s.max_w == 6.0
    assert abs(s.duration_s - 0.2) < 1e-9 and s.gpu_load_pct == 75.0 and s.tj_max_c == 70.5


def test_summarize_unknown_rail_raises():
    with pytest.raises(ValueError):
        summarize([{"VDD_IN": 1.0}], None, "VDD_GPU_SOC")
