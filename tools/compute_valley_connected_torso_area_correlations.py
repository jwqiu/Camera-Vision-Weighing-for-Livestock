#!/usr/bin/env python3
"""Compute rear-side Topview area cut by the connected upper/lower valleys."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import read_ply_vertices


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
BOUNDARY_CSV = (
    BASE
    / "pca_horizontal_60pct_upper_halfwidth_preview_61"
    / "pca_upper_halfwidth_records_61.csv"
)
OUTPUT_DIR = BASE / "valley_connected_torso_area_61"
RECORDS_CSV = OUTPUT_DIR / "valley_connected_torso_area_records_61.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "valley_connected_torso_area_correlations_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "valley_connected_torso_area_summary.json"

GRID_CELL_M = 0.01
METHOD_VERSION = "upper_lower_valley_connector_rear_halfplane_area_v1.0"
RECORDED_ON = "2026-09-06"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def projected_grid_area(points_xy: np.ndarray) -> tuple[int, float]:
    cells = np.floor(points_xy / GRID_CELL_M).astype(np.int64)
    unique_count = len(np.unique(cells, axis=0))
    return unique_count, unique_count * GRID_CELL_M**2


def rear_halfplane_mask(
    points_xy: np.ndarray,
    upper_xy: np.ndarray,
    lower_xy: np.ndarray,
    rear_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    connector = lower_xy - upper_xy
    if float(np.linalg.norm(connector)) < 1e-9:
        raise ValueError("upper/lower boundary points are coincident")

    def cross_from_line(points: np.ndarray) -> np.ndarray:
        relative = points - upper_xy
        return connector[0] * relative[..., 1] - connector[1] * relative[..., 0]

    rear_sign = float(cross_from_line(rear_xy))
    if abs(rear_sign) < 1e-9:
        raise ValueError("rear anchor lies on connector")
    point_cross = cross_from_line(points_xy)
    return point_cross * rear_sign >= 0.0, rear_sign


def loocv_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = np.empty_like(y, dtype=float)
    for test_index in range(len(y)):
        train = np.arange(len(y)) != test_index
        slope, intercept = np.polyfit(x[train], y[train], 1)
        predictions[test_index] = slope * x[test_index] + intercept
    residuals = predictions - y
    return {
        "loocv_r": float(np.corrcoef(predictions, y)[0, 1]),
        "loocv_rmse_kg": float(np.sqrt(np.mean(residuals**2))),
        "loocv_mae_kg": float(np.mean(np.abs(residuals))),
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including tied values."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def metric_summary(
    code: str, label: str, field: str, x: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    pearson = float(np.corrcoef(x, weights)[0, 1])
    spearman = float(
        np.corrcoef(rankdata_average(x), rankdata_average(weights))[0, 1]
    )
    slope, intercept = np.polyfit(x, weights, 1)
    fitted = slope * x + intercept
    residuals = fitted - weights
    z = float(np.arctanh(np.clip(pearson, -0.999999, 0.999999)))
    se = 1 / math.sqrt(len(x) - 3)
    ci_low, ci_high = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return {
        "metric_code": code,
        "metric_label_zh": label,
        "metric_field": field,
        "n": len(x),
        "pearson_r": round(pearson, 6),
        "spearman_rho": round(spearman, 6),
        "r_squared": round(pearson**2, 6),
        "pearson_95_ci_lower": round(float(ci_low), 6),
        "pearson_95_ci_upper": round(float(ci_high), 6),
        "linear_slope": round(float(slope), 6),
        "linear_intercept": round(float(intercept), 6),
        "in_sample_rmse_kg": round(float(np.sqrt(np.mean(residuals**2))), 6),
        "in_sample_mae_kg": round(float(np.mean(np.abs(residuals))), 6),
        **{key: round(value, 6) for key, value in loocv_linear(x, weights).items()},
        "outcome_field": "ground_truth_weight_kg",
        "analysis_population": "61 cattle with accepted upper/lower valley boundaries",
        "method_version": METHOD_VERSION,
        "calculated_on": RECORDED_ON,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_by_cow = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    boundary_rows = read_csv(BOUNDARY_CSV)
    if len(boundary_rows) != 61:
        raise ValueError(f"expected 61 boundary rows, found {len(boundary_rows)}")

    records: list[dict[str, object]] = []
    for boundary in boundary_rows:
        cow_id = int(boundary["cow_id"])
        main_row = main_by_cow[cow_id]
        vertices = read_ply_vertices(BASE / main_row["extracted_ply"])
        points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
        rear_xy = np.asarray(
            [float(boundary["rear_axis_x_m"]), float(boundary["rear_axis_y_m"])]
        )
        upper_xy = np.asarray(
            [
                float(boundary["pca_upper_width_valley_contour_x_m"]),
                float(boundary["pca_upper_width_valley_contour_y_m"]),
            ]
        )
        lower_xy = np.asarray(
            [
                float(boundary["pca_lower_width_valley_contour_x_m"]),
                float(boundary["pca_lower_width_valley_contour_y_m"]),
            ]
        )
        mask, rear_sign = rear_halfplane_mask(points_xy, upper_xy, lower_xy, rear_xy)
        grid_count, area_m2 = projected_grid_area(points_xy[mask])
        upper_ratio = float(boundary["pca_upper_width_valley_station_ratio_full_axis"])
        lower_ratio = float(boundary["pca_lower_width_valley_station_ratio_full_axis"])
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(main_row["ground_truth_weight_kg"]),
                "ground_truth_heart_girth_cm": float(main_row["ground_truth_heart_girth_cm"]),
                "upper_boundary_rule": boundary["pca_upper_boundary_selected_rule"],
                "lower_boundary_rule": boundary["pca_lower_boundary_selected_rule"],
                "upper_boundary_ratio_from_rear": upper_ratio,
                "lower_boundary_ratio_from_rear": lower_ratio,
                "boundary_mean_ratio_from_rear": (upper_ratio + lower_ratio) / 2,
                "boundary_side_ratio_difference": upper_ratio - lower_ratio,
                "upper_boundary_x_m": float(upper_xy[0]),
                "upper_boundary_y_m": float(upper_xy[1]),
                "lower_boundary_x_m": float(lower_xy[0]),
                "lower_boundary_y_m": float(lower_xy[1]),
                "connector_length_m": float(np.linalg.norm(lower_xy - upper_xy)),
                "rear_halfplane_sign": rear_sign,
                "rear_side_core_point_count": int(np.count_nonzero(mask)),
                "projection_grid_cell_size_m": GRID_CELL_M,
                "projection_grid_cell_count": grid_count,
                "valley_connected_rear_torso_projected_area_m2": area_m2,
                "existing_pca_45_48_core_area_m2": float(
                    main_row["pca_full_torso_projected_area_m2"]
                ),
                "source_top_ply": main_row["extracted_ply"],
                "area_definition": "projected point-cloud cells on rear side of upper-lower valley connector",
                "method_version": METHOD_VERSION,
                "status": "ok",
            }
        )

    records.sort(key=lambda row: int(row["cow_id"]))
    write_csv(RECORDS_CSV, records)
    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    metrics = [
        (
            "valley_connected_rear_torso_area",
            "上下谷底连线左侧躯干投影面积",
            "valley_connected_rear_torso_projected_area_m2",
            np.asarray(
                [
                    float(row["valley_connected_rear_torso_projected_area_m2"])
                    for row in records
                ]
            ),
        ),
        (
            "existing_pca_45_48_core_area",
            "原PCA 45%/48%核心躯干投影面积",
            "existing_pca_45_48_core_area_m2",
            np.asarray([float(row["existing_pca_45_48_core_area_m2"]) for row in records]),
        ),
        (
            "heart_girth",
            "胸围",
            "ground_truth_heart_girth_cm",
            np.asarray([float(row["ground_truth_heart_girth_cm"]) for row in records]),
        ),
    ]
    correlations = [
        metric_summary(code, label, field, values, weights)
        for code, label, field, values in metrics
    ]
    write_csv(CORRELATIONS_CSV, correlations)

    area_values = np.asarray(
        [float(row["valley_connected_rear_torso_projected_area_m2"]) for row in records]
    )
    side_differences = np.asarray(
        [float(row["boundary_side_ratio_difference"]) for row in records]
    )
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "area_definition": "connect upper/lower valley contour points and retain the half-plane containing the rear endpoint",
                "projection_grid_cell_size_m": GRID_CELL_M,
                "mean_area_m2": float(np.mean(area_values)),
                "minimum_area_m2": float(np.min(area_values)),
                "maximum_area_m2": float(np.max(area_values)),
                "mean_absolute_upper_lower_boundary_ratio_difference": float(
                    np.mean(np.abs(side_differences))
                ),
                "maximum_absolute_upper_lower_boundary_ratio_difference": float(
                    np.max(np.abs(side_differences))
                ),
                "correlations": correlations,
                "records_file": str(RECORDS_CSV.relative_to(ROOT)),
                "correlations_file": str(CORRELATIONS_CSV.relative_to(ROOT)),
                "method_version": METHOD_VERSION,
                "calculated_on": RECORDED_ON,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(correlations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
