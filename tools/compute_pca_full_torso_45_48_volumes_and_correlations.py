#!/usr/bin/env python3
"""Compute PCA 45/48 projected torso volumes and weight correlations."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


RESULT_CSV = BASE / "pca_full_torso_45_48_correlations.csv"
GRID_CELL_M = 0.01
GRID_CELL_AREA_M2 = GRID_CELL_M**2
METHOD_VERSION = "pca_full_torso_45_48_1cm_column_sum_v1.0"
RECORDED_ON = "2026-09-06"

VOLUME_FIELDS = [
    "pca_full_torso_projection_grid_cell_area_m2",
    "pca_full_torso_projection_volume_cell_count",
    "pca_full_torso_projection_mean_height_m",
    "pca_full_torso_projection_median_height_m",
    "pca_full_torso_projection_min_height_m",
    "pca_full_torso_projection_max_height_m",
    "pca_full_torso_projected_volume_m3",
    "pca_full_torso_projected_volume_l",
    "pca_full_torso_height_definition",
    "pca_full_torso_cell_height_aggregation",
    "pca_full_torso_projected_volume_method",
    "pca_full_torso_projected_volume_status",
    "pca_full_torso_projected_volume_recorded_on",
]


def cell_median_heights(vertices: np.ndarray, plane_a: float, plane_b: float, plane_c: float):
    x = vertices["x"].astype(float)
    y = vertices["y"].astype(float)
    z = vertices["z"].astype(float)
    heights = plane_a * x + plane_b * y + plane_c - z
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(heights) & (heights > 0)
    cells = np.floor(np.column_stack([x[valid], y[valid]]) / GRID_CELL_M).astype(np.int64)
    heights = heights[valid]
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    inverse_sorted = inverse[order]
    heights_sorted = heights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(inverse_sorted)) + 1]
    ends = np.r_[starts[1:], len(heights_sorted)]
    medians = np.asarray([np.median(heights_sorted[start:end]) for start, end in zip(starts, ends)])
    return medians


def compute_volume(row: dict[str, str]) -> dict[str, object]:
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.array(
        [float(row["pca_full_torso_axis_centroid_x_m"]), float(row["pca_full_torso_axis_centroid_y_m"])]
    )
    axis = np.array([float(row["pca_full_torso_axis_unit_x"]), float(row["pca_full_torso_axis_unit_y"])])
    axis /= np.linalg.norm(axis)
    longitudinal = (points_xy - centroid) @ axis
    core_mask = (
        (longitudinal >= float(row["pca_full_torso_rear_boundary_axis_m"]))
        & (longitudinal <= float(row["pca_full_torso_head_boundary_axis_m"]))
    )
    core = vertices[core_mask]
    cell_heights = cell_median_heights(
        core,
        float(row["ground_plane_a"]),
        float(row["ground_plane_b"]),
        float(row["ground_plane_c_m"]),
    )
    if len(cell_heights) == 0:
        raise ValueError(f"no valid volume cells for cow {row['cow_id']}")
    expected_count = int(row["pca_full_torso_projection_grid_cell_count"])
    if len(cell_heights) != expected_count:
        raise ValueError(
            f"grid-cell mismatch for cow {row['cow_id']}: volume={len(cell_heights)} area={expected_count}"
        )
    volume_m3 = float(np.sum(cell_heights) * GRID_CELL_AREA_M2)
    return {
        "pca_full_torso_projection_grid_cell_area_m2": GRID_CELL_AREA_M2,
        "pca_full_torso_projection_volume_cell_count": len(cell_heights),
        "pca_full_torso_projection_mean_height_m": round(float(np.mean(cell_heights)), 6),
        "pca_full_torso_projection_median_height_m": round(float(np.median(cell_heights)), 6),
        "pca_full_torso_projection_min_height_m": round(float(np.min(cell_heights)), 6),
        "pca_full_torso_projection_max_height_m": round(float(np.max(cell_heights)), 6),
        "pca_full_torso_projected_volume_m3": round(volume_m3, 6),
        "pca_full_torso_projected_volume_l": round(volume_m3 * 1000, 3),
        "pca_full_torso_height_definition": "ground_plane_z_minus_cattle_surface_z",
        "pca_full_torso_cell_height_aggregation": "median_height_within_each_1cm_xy_cell",
        "pca_full_torso_projected_volume_method": METHOD_VERSION,
        "pca_full_torso_projected_volume_status": "ok",
        "pca_full_torso_projected_volume_recorded_on": RECORDED_ON,
    }


def pearson_summary(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    lower, upper = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return {
        "n": n,
        "pearson_r": round(r, 6),
        "r_squared": round(r * r, 6),
        "fisher_95_ci_lower": round(float(lower), 6),
        "fisher_95_ci_upper": round(float(upper), 6),
    }


def main() -> None:
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    for field in VOLUME_FIELDS:
        if field not in headers:
            headers.append(field)

    included: list[dict[str, str]] = []
    for row in rows:
        if row.get("pca_full_torso_analysis_eligible") != "TRUE":
            for field in VOLUME_FIELDS:
                row[field] = ""
            continue
        result = compute_volume(row)
        row.update({key: str(value) for key, value in result.items()})
        included.append(row)

    if len(included) != 62:
        raise ValueError(f"expected 62 eligible cattle, found {len(included)}")

    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in included])
    metrics = [
        (
            "core_torso_projected_volume",
            "核心躯干投影体积",
            "pca_full_torso_projected_volume_m3",
            np.asarray([float(row["pca_full_torso_projected_volume_m3"]) for row in included]),
        ),
        (
            "core_torso_projected_area",
            "核心躯干投影面积",
            "pca_full_torso_projected_area_m2",
            np.asarray([float(row["pca_full_torso_projected_area_m2"]) for row in included]),
        ),
        (
            "heart_girth",
            "胸围",
            "ground_truth_heart_girth_cm",
            np.asarray([float(row["ground_truth_heart_girth_cm"]) for row in included]),
        ),
    ]
    correlation_rows = []
    for code, label, field, values in metrics:
        summary = pearson_summary(values, weights)
        correlation_rows.append(
            {
                "metric_code": code,
                "metric_label_zh": label,
                "metric_field": field,
                "outcome_field": "ground_truth_weight_kg",
                **summary,
                "correlation_method": "Pearson",
                "confidence_interval_method": "Fisher_z_95pct",
                "analysis_population": "62 qualified cattle including cow 150 endpoint fallback",
                "method_version": METHOD_VERSION,
                "calculated_on": RECORDED_ON,
            }
        )
    with RESULT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)

    print({"volume_records": len(included), "correlations": correlation_rows})


if __name__ == "__main__":
    main()
