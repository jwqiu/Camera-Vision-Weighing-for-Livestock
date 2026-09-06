#!/usr/bin/env python3
"""Measure visible Topview dorsal arcs near the PCA maximum-width station."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
VOLUME_CSV = (
    BASE
    / "elliptical_slice_volumes_two_right_boundaries"
    / "elliptical_core_torso_volumes_two_methods_comparison_61.csv"
)
OUTPUT_DIR = BASE / "topview_dorsal_arc_at_max_width"
RECORDS_CSV = OUTPUT_DIR / "topview_dorsal_arc_metrics_61.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "topview_dorsal_arc_correlations_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "topview_dorsal_arc_summary.json"

LONGITUDINAL_BIN_M = 0.01
LATERAL_BIN_M = 0.01
LATERAL_LOW_QUANTILE = 0.02
LATERAL_HIGH_QUANTILE = 0.98
SMOOTHING_BINS = 3
WINDOWS_CM = (1, 2, 5, 10, 15)
METHOD_VERSION = "topview_visible_dorsal_arc_maxwidth_1cm_bins_v1.0"

sys.path.insert(0, str(ROOT / "tools"))
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rolling_median_3(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.median(values[max(0, index - 1) : min(len(values), index + 2)])
            for index in range(len(values))
        ]
    )


def profile_arc(lateral: np.ndarray, height: np.ndarray) -> tuple[float, float] | None:
    if len(lateral) < 80:
        return None
    low, high = np.quantile(lateral, [LATERAL_LOW_QUANTILE, LATERAL_HIGH_QUANTILE])
    edges = np.arange(
        math.floor(low / LATERAL_BIN_M) * LATERAL_BIN_M,
        math.ceil(high / LATERAL_BIN_M) * LATERAL_BIN_M + LATERAL_BIN_M * 0.5,
        LATERAL_BIN_M,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.digitize(lateral, edges) - 1
    values = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        cell = height[indices == index]
        if len(cell) >= 2:
            values[index] = np.median(cell)

    valid = np.isfinite(values)
    if np.count_nonzero(valid) < max(15, 0.65 * len(values)):
        return None
    first, last = np.flatnonzero(valid)[[0, -1]]
    centers = centers[first : last + 1]
    values = values[first : last + 1]
    valid = np.isfinite(values)
    values[~valid] = np.interp(
        np.flatnonzero(~valid), np.flatnonzero(valid), values[valid]
    )
    values = rolling_median_3(values)
    arc = float(np.sum(np.sqrt(np.diff(centers) ** 2 + np.diff(values) ** 2)))
    chord = float(centers[-1] - centers[0])
    return arc, chord


def calculate_cow(row: dict[str, str]) -> dict[str, object]:
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.asarray(
        [
            float(row["pca_full_torso_axis_centroid_x_m"]),
            float(row["pca_full_torso_axis_centroid_y_m"]),
        ]
    )
    axis = np.asarray(
        [
            float(row["pca_full_torso_axis_unit_x"]),
            float(row["pca_full_torso_axis_unit_y"]),
        ]
    )
    axis /= np.linalg.norm(axis)
    normal = np.asarray([-axis[1], axis[0]])
    relative = points_xy - centroid
    longitudinal = relative @ axis
    lateral = relative @ normal
    plane_z = (
        float(row["ground_plane_a"]) * vertices["x"]
        + float(row["ground_plane_b"]) * vertices["y"]
        + float(row["ground_plane_c_m"])
    )
    height = plane_z - vertices["z"]
    maximum_station = float(row["pca_full_torso_max_width_axis_m"])

    result: dict[str, object] = {
        "cow_id": int(row["cow_id"]),
        "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
        "pca_max_width_m": float(row["pca_full_torso_max_width_m"]),
        "pca_max_width_axis_m": maximum_station,
        "method_version": METHOD_VERSION,
    }
    for window_cm in WINDOWS_CM:
        offsets = (
            np.arange(window_cm, dtype=float) - (window_cm - 1) / 2
        ) * LONGITUDINAL_BIN_M
        arcs: list[float] = []
        chords: list[float] = []
        for offset in offsets:
            center = maximum_station + offset
            selected = np.abs(longitudinal - center) < LONGITUDINAL_BIN_M / 2
            measured = profile_arc(lateral[selected], height[selected])
            if measured is not None:
                arcs.append(measured[0])
                chords.append(measured[1])
        if not arcs:
            raise ValueError(f"cow {row['cow_id']}: no valid dorsal arc in {window_cm} cm window")
        arc = float(np.median(arcs))
        chord = float(np.median(chords))
        result[f"dorsal_arc_{window_cm}cm_m"] = arc
        result[f"dorsal_chord_{window_cm}cm_m"] = chord
        result[f"dorsal_arc_to_max_width_ratio_{window_cm}cm"] = (
            arc / float(row["pca_full_torso_max_width_m"])
        )
        result[f"valid_profile_count_{window_cm}cm"] = len(arcs)
    return result


def score(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    predictions = []
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        beta = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        predictions.append(float(design[index] @ beta))
    predictions_array = np.asarray(predictions)
    r = float(np.corrcoef(x, y)[0, 1])
    fisher = np.arctanh(r)
    se = 1 / math.sqrt(len(x) - 3)
    lower, upper = np.tanh([fisher - 1.959964 * se, fisher + 1.959964 * se])
    return {
        "n": len(x),
        "pearson_r": r,
        "r_squared": r * r,
        "fisher_95_ci_lower": float(lower),
        "fisher_95_ci_upper": float(upper),
        "in_sample_rmse_kg": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "loocv_r": float(np.corrcoef(predictions_array, y)[0, 1]),
        "loocv_rmse_kg": float(np.sqrt(np.mean((predictions_array - y) ** 2))),
        "loocv_mae_kg": float(np.mean(np.abs(predictions_array - y))),
    }


def main() -> None:
    source_rows = read_csv(MAIN_CSV)
    strict_rows = [
        row
        for row in source_rows
        if row.get("pca_full_torso_analysis_eligible", "").upper() == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    records = [calculate_cow(row) for row in strict_rows]
    if len(records) != 61:
        raise ValueError(f"expected 61 strict cattle, found {len(records)}")

    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    correlations: list[dict[str, object]] = []
    for window_cm in WINDOWS_CM:
        for prefix, label in (
            ("dorsal_arc", "可见背部横向弧长"),
            ("dorsal_arc_to_max_width_ratio", "背部弧长/最大体宽"),
        ):
            field = f"{prefix}_{window_cm}cm" + ("_m" if prefix == "dorsal_arc" else "")
            values = np.asarray([float(row[field]) for row in records])
            correlations.append(
                {
                    "metric": f"{window_cm}cm窗口{label}",
                    "field": field,
                    "window_cm": window_cm,
                    **score(values, weights),
                }
            )

    by_id = {int(row["cow_id"]): row for row in strict_rows}
    comparison_inputs: list[tuple[str, np.ndarray]] = [
        (
            "Topview最大体宽",
            np.asarray([float(by_id[int(row["cow_id"])]["pca_full_torso_max_width_m"]) for row in records]),
        ),
        (
            "核心投影面积",
            np.asarray([float(by_id[int(row["cow_id"])]["pca_full_torso_projected_area_m2"]) for row in records]),
        ),
        (
            "胸围",
            np.asarray([float(by_id[int(row["cow_id"])]["ground_truth_heart_girth_cm"]) for row in records]),
        ),
    ]
    volumes = {int(row["cow_id"]): row for row in read_csv(VOLUME_CSV)}
    comparison_inputs.extend(
        [
            (
                "转换矩阵法完整切片体积",
                np.asarray([float(volumes[int(row["cow_id"])]["matrix_boundary_volume_m3"]) for row in records]),
            ),
            (
                "比例法完整切片体积",
                np.asarray([float(volumes[int(row["cow_id"])]["proportional_boundary_volume_m3"]) for row in records]),
            ),
        ]
    )
    comparisons = [
        {"metric": label, **score(values, weights)} for label, values in comparison_inputs
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RECORDS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with CORRELATIONS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)
    summary = {
        "method_version": METHOD_VERSION,
        "analysis_population": "61 strict PCA rear45/head48 cattle",
        "primary_metric": "median visible dorsal arc across ten adjacent 1cm profiles centered at PCA maximum-width station",
        "surface_scope": "Topview-visible dorsal surface only; not full body circumference",
        "lateral_profile": {
            "bin_m": LATERAL_BIN_M,
            "bounds": "2nd to 98th lateral percentile per profile",
            "height_aggregation": "median per lateral bin",
            "gap_handling": "interpolate internal missing bins",
            "smoothing": "one 3-bin rolling median",
        },
        "correlations": correlations,
        "same_61_cattle_comparisons": comparisons,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {RECORDS_CSV.relative_to(ROOT)}")
    print(f"wrote {CORRELATIONS_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
