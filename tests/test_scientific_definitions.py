from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples.analysis import response_endpoints, classify_responses, check_structure_response_table
from dmegan_examples.morphology import microstructure_stats_single


def test_empty_field_has_finite_descriptors():
    result = microstructure_stats_single(np.zeros((64, 64)))
    assert np.isfinite(list(result.values())).all()
    assert result["area_fraction"] == 0
    assert result["component_count"] == 0


def test_components_use_four_neighbour_connectivity():
    image = np.zeros((16, 16))
    image[3, 3] = 1
    image[4, 4] = 1
    result = microstructure_stats_single(image)
    assert result["component_count"] == 2
    assert result["component_area_max"] == 1


def test_response_integrals_convert_milliseconds_to_seconds():
    endpoints = response_endpoints(ROOT / "data")
    et = next(r for r in endpoints if r["role"] == "electrothermal_reference")
    assert np.isclose(et["temperature_range_integral_K_s"], 0.4550119133198106, rtol=1e-12)
    assert np.isclose(et["maximum_temperature_rise_integral_K_s"], 4.854794161295499, rtol=1e-12)


def test_benchmark_ratios_are_dimensionless_and_self_normalized():
    records = classify_responses(response_endpoints(ROOT / "data"))
    assert len(records) == 302
    assert sum(r["role"] == "candidate" for r in records) == 300
    et = next(r for r in records if r["role"] == "electrothermal_reference")
    cm = next(r for r in records if r["role"] == "modulus_reference")
    assert et["electrothermal_ratio"] == 1
    assert cm["modulus_ratio"] == 1


def test_saved_image_collection_has_documented_sizes_and_hashes():
    import csv
    import hashlib
    from collections import Counter
    with (ROOT / "data/image_index.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert Counter(r["group"] for r in records) == {"characterization": 32, "ar": 20, "vae": 20, "diffusion": 20, "gan": 20, "dmegan": 500}
    for record in records:
        assert hashlib.sha256((ROOT / record["file"]).read_bytes()).hexdigest() == record["sha256"]


def test_structure_response_table_uses_matching_record_targets():
    result = check_structure_response_table(ROOT / "data", response_endpoints(ROOT / "data"))
    assert result["records"] == 300
    assert result["structural_features"] == 21
    assert result["maximum_response_difference"] < 1e-10
