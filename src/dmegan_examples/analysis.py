"""Image preparation and screening of retained response records."""
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

ELECTROTHERMAL_KEYS = (
    "Jmean_20ms_A_cm2", "Jmax_20ms_A_cm2",
    "temperature_range_integral_K_s", "maximum_temperature_rise_integral_K_s",
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_carbon_field(path, image_size=512):
    """Apply the retained reference-image preprocessing in numeric precision."""
    with Image.open(path) as source:
        red = np.asarray(source.convert("RGB"), dtype=np.float32)[..., 0]
    image = Image.fromarray(np.clip(red, 0, 255).astype(np.uint8))
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def response_endpoints(data_root):
    """Integrate temperature histories and retain the stated endpoint units."""
    data_root = Path(data_root)
    histories = read_csv(data_root / "responses/thermal_history.csv")
    mechanics = read_csv(data_root / "responses/mechanical_response.csv")
    if len({r["record_id"] for r in mechanics}) != len(mechanics):
        raise ValueError("Repeated mechanical record identifiers")
    if {r["record_id"] for r in histories} != {r["record_id"] for r in mechanics}:
        raise ValueError("Thermal and mechanical record identifiers differ")
    output = []
    for mechanical in mechanics:
        record_id = mechanical["record_id"]
        thermal = sorted((r for r in histories if r["record_id"] == record_id), key=lambda r: float(r["time_ms"]))
        time_ms = np.array([float(r["time_ms"]) for r in thermal])
        if not np.array_equal(time_ms, [0, 10, 20, 30, 40, 50]):
            raise ValueError(f"Unexpected time coordinates for {record_id}")
        if any(r["role"] != mechanical["role"] for r in thermal):
            raise ValueError(f"Inconsistent record roles for {record_id}")
        tmax = np.array([float(r["Tmax_C"]) for r in thermal])
        tmin = np.array([float(r["Tmin_C"]) for r in thermal])
        delta = np.array([float(r["DeltaT_K"]) for r in thermal])
        jmean = np.array([float(r["Jmean_A_cm2"]) for r in thermal])
        jmax = np.array([float(r["Jmax_A_cm2"]) for r in thermal])
        if not np.isfinite(np.stack([tmax, tmin, delta, jmean, jmax])).all():
            raise ValueError(f"Non-finite response in {record_id}")
        if not np.allclose(delta, tmax - tmin, atol=1e-9, rtol=1e-9):
            raise ValueError(f"Temperature difference does not match extrema in {record_id}")
        displacement = float(mechanical["displacement_um"])
        modulus = float(mechanical["compressive_modulus_GPa"])
        if not math.isfinite(displacement) or displacement == 0 or not math.isfinite(modulus) or modulus <= 0:
            raise ValueError(f"Invalid mechanical response in {record_id}")
        if not math.isclose(0.001 * 60 / abs(displacement), modulus, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Inconsistent compression units in {record_id}")
        output.append(dict(
            record_id=record_id, role=mechanical["role"],
            Jmean_20ms_A_cm2=float(jmean[2]), Jmax_20ms_A_cm2=float(jmax[2]),
            temperature_range_integral_K_s=float(np.trapezoid(delta, time_ms * 1e-3)),
            maximum_temperature_rise_integral_K_s=float(np.trapezoid(tmax - 20, time_ms * 1e-3)),
            compressive_modulus_GPa=modulus,
        ))
    return output


def classify_responses(endpoints):
    et = [r for r in endpoints if r["role"] == "electrothermal_reference"]
    cm = [r for r in endpoints if r["role"] == "modulus_reference"]
    if len(et) != 1 or len(cm) != 1:
        raise ValueError("Exactly one reference of each type is required")
    et, cm = et[0], cm[0]
    if any(et[k] <= 0 for k in ELECTROTHERMAL_KEYS) or cm["compressive_modulus_GPa"] <= 0:
        raise ValueError("Benchmark denominators must be positive")
    output = []
    for row in endpoints:
        item = row.copy()
        item["electrothermal_ratio"] = max(row[k] / et[k] for k in ELECTROTHERMAL_KEYS)
        item["modulus_ratio"] = row["compressive_modulus_GPa"] / cm["compressive_modulus_GPa"]
        item["meets_electrothermal_condition"] = item["electrothermal_ratio"] <= 1
        item["meets_modulus_condition"] = item["modulus_ratio"] >= 1
        item["meets_both_conditions"] = item["meets_electrothermal_condition"] and item["meets_modulus_condition"]
        output.append(item)
    return output


def check_structure_response_table(data_root, endpoints):
    """Check the retained feature table against the corresponding responses."""
    rows = read_csv(Path(data_root) / "responses/structure_response.csv")
    expected = {r["record_id"]: r for r in endpoints if r["role"] == "candidate"}
    if len(rows) != len(expected) or {r["run_id"] for r in rows} != set(expected):
        raise ValueError("Structure-response records do not match candidate identifiers")
    columns = [key for key in rows[0] if key.endswith(("_mean", "_section_sd", "_section_range"))]
    if len(columns) != 21:
        raise ValueError("Expected 21 retained structural features")
    feature_values = np.array([[float(r[k]) for k in columns] for r in rows])
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite retained structural feature")
    maximum_difference = 0.0
    pairs = [
        ("Javg_20ms_A_cm2", "Jmean_20ms_A_cm2"),
        ("Jmax_20ms_A_cm2", "Jmax_20ms_A_cm2"),
        ("cum_deltaT_Ks", "temperature_range_integral_K_s"),
        ("cum_Tmax_exposure_Ks", "maximum_temperature_rise_integral_K_s"),
        ("Keff_GPa", "compressive_modulus_GPa"),
    ]
    for row in rows:
        target = expected[row["run_id"]]
        for source_key, target_key in pairs:
            original, recalculated = float(row[source_key]), target[target_key]
            if not math.isclose(original, recalculated, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Response mismatch in {row['run_id']}: {source_key}")
            maximum_difference = max(maximum_difference, abs(original - recalculated))
    return dict(records=len(rows), structural_features=21, finite_features=True,
                maximum_response_difference=maximum_difference)
