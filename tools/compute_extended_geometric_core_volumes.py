#!/usr/bin/env python3
"""Compute 1 cm column-sum projected volume for geometric-centerline core torsos."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
AXIS_DIR = BASE / "skeleton_axis_comparison"
AREA_CSV = AXIS_DIR / "extended_geometric_core_area_records.csv"
SUMMARY_CSV = AXIS_DIR / "extended_geometric_core_volume_records.csv"
SUMMARY_JSON = AXIS_DIR / "extended_geometric_core_volume_summary.json"

GRID_CELL_M = 0.01
GRID_CELL_AREA_M2 = GRID_CELL_M**2
METHOD_VERSION = "extended_geometric_core_torso_1cm_column_sum_v1.0"
PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]
)


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"unexpected end of PLY header: {path}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count: {path}")
        return np.fromfile(handle, dtype=PLY_DTYPE, count=vertex_count)


def write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def cell_median_heights(vertices: np.ndarray, plane_a: float, plane_b: float, plane_c: float):
    x = vertices["x"].astype(float)
    y = vertices["y"].astype(float)
    z = vertices["z"].astype(float)
    heights = plane_a * x + plane_b * y + plane_c - z
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(heights) & (heights > 0)
    cells = np.floor(np.column_stack([x[valid], y[valid]]) / GRID_CELL_M).astype(np.int64)
    heights = heights[valid]
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    inverse_sorted = inverse[order]
    heights_sorted = heights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(inverse_sorted)) + 1]
    ends = np.r_[starts[1:], len(heights_sorted)]
    medians = np.asarray([np.median(heights_sorted[start:end]) for start, end in zip(starts, ends)])
    return unique_cells, medians, int(np.count_nonzero(valid))


def pearson_summary(x: np.ndarray, y: np.ndarray):
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if n > 3 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1 / math.sqrt(n - 3)
        lower, upper = np.tanh([z - 1.96 * se, z + 1.96 * se])
    else:
        lower = upper = r
    return {
        "n": n,
        "pearson_r": round(r, 6),
        "r_squared": round(r * r, 6),
        "fisher_95_ci_lower": round(float(lower), 6),
        "fisher_95_ci_upper": round(float(upper), 6),
    }


def update_main_csv(records_by_cow):
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    fields = [
        "extended_geometric_torso_projection_grid_cell_size_m",
        "extended_geometric_torso_projection_grid_cell_area_m2",
        "extended_geometric_torso_projection_grid_cell_count",
        "extended_geometric_torso_projection_mean_height_m",
        "extended_geometric_torso_projected_volume_m3",
        "extended_geometric_torso_projected_volume_l",
        "extended_geometric_torso_volume_difference_vs_pca_m3",
        "extended_geometric_torso_volume_absolute_difference_vs_pca_m3",
        "extended_geometric_torso_volume_difference_vs_pca_percent",
        "extended_geometric_torso_projected_volume_status",
        "extended_geometric_torso_projected_volume_method",
    ]
    for field in fields:
        if field not in headers:
            headers.append(field)
    for row in rows:
        record = records_by_cow[int(row["cow_id"])]
        row.update(
            {
                "extended_geometric_torso_projection_grid_cell_size_m": record["projection_grid_cell_size_m"],
                "extended_geometric_torso_projection_grid_cell_area_m2": record["projection_grid_cell_area_m2"],
                "extended_geometric_torso_projection_grid_cell_count": record["projection_grid_cell_count"],
                "extended_geometric_torso_projection_mean_height_m": record["mean_cell_height_m"],
                "extended_geometric_torso_projected_volume_m3": record["new_geometric_projected_volume_m3"],
                "extended_geometric_torso_projected_volume_l": record["new_geometric_projected_volume_l"],
                "extended_geometric_torso_volume_difference_vs_pca_m3": record["difference_new_minus_pca_m3"],
                "extended_geometric_torso_volume_absolute_difference_vs_pca_m3": record["absolute_difference_m3"],
                "extended_geometric_torso_volume_difference_vs_pca_percent": record["difference_vs_pca_percent"],
                "extended_geometric_torso_projected_volume_status": record["status"],
                "extended_geometric_torso_projected_volume_method": METHOD_VERSION,
            }
        )
    write_csv(MAIN_CSV, headers, rows)


def main():
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    main_by_cow = {int(row["cow_id"]): row for row in main_rows}
    with AREA_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        area_rows = list(csv.DictReader(handle))

    records = []
    for area_row in area_rows:
        cow_id = int(area_row["cow_id"])
        main_row = main_by_cow[cow_id]
        vertices = read_ply_vertices(ROOT / area_row["core_ply"])
        _, cell_heights, valid_point_count = cell_median_heights(
            vertices,
            float(main_row["ground_plane_a"]),
            float(main_row["ground_plane_b"]),
            float(main_row["ground_plane_c_m"]),
        )
        cell_count = len(cell_heights)
        expected_cell_count = int(area_row["projection_grid_cell_count"])
        if cell_count != expected_cell_count:
            raise ValueError(f"grid cell mismatch for cow {cow_id}: {cell_count} != {expected_cell_count}")
        volume_m3 = float(np.sum(cell_heights) * GRID_CELL_AREA_M2)
        old_volume_m3 = float(main_row["torso_projected_volume_m3"])
        difference = volume_m3 - old_volume_m3
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(main_row["ground_truth_weight_kg"]),
                "old_pca_projected_volume_m3": round(old_volume_m3, 6),
                "old_pca_volume_field": "torso_projected_volume_m3",
                "new_geometric_projected_volume_m3": round(volume_m3, 6),
                "new_geometric_projected_volume_l": round(volume_m3 * 1000, 3),
                "difference_new_minus_pca_m3": round(difference, 6),
                "absolute_difference_m3": round(abs(difference), 6),
                "difference_vs_pca_percent": round(difference / old_volume_m3 * 100, 4),
                "absolute_difference_vs_pca_percent": round(abs(difference) / old_volume_m3 * 100, 4),
                "projected_area_m2": round(cell_count * GRID_CELL_AREA_M2, 6),
                "projection_grid_cell_size_m": GRID_CELL_M,
                "projection_grid_cell_area_m2": GRID_CELL_AREA_M2,
                "projection_grid_cell_count": cell_count,
                "mean_cell_height_m": round(float(np.mean(cell_heights)), 6),
                "median_cell_height_m": round(float(np.median(cell_heights)), 6),
                "minimum_cell_height_m": round(float(np.min(cell_heights)), 6),
                "maximum_cell_height_m": round(float(np.max(cell_heights)), 6),
                "valid_core_point_count": valid_point_count,
                "height_definition": "ground_plane_z_minus_cattle_surface_z",
                "cell_height_aggregation": "median_height_within_each_1cm_xy_cell",
                "core_ply": area_row["core_ply"],
                "method_version": METHOD_VERSION,
                "status": "ok",
            }
        )

    old_values = np.asarray([row["old_pca_projected_volume_m3"] for row in records], dtype=float)
    new_values = np.asarray([row["new_geometric_projected_volume_m3"] for row in records], dtype=float)
    weights = np.asarray([row["ground_truth_weight_kg"] for row in records], dtype=float)
    differences = new_values - old_values
    percentages = differences / old_values * 100
    summary = {
        "n": len(records),
        "method_version": METHOD_VERSION,
        "mean_old_pca_volume_m3": round(float(np.mean(old_values)), 6),
        "mean_new_geometric_volume_m3": round(float(np.mean(new_values)), 6),
        "mean_signed_difference_m3": round(float(np.mean(differences)), 6),
        "median_signed_difference_m3": round(float(np.median(differences)), 6),
        "mean_absolute_difference_m3": round(float(np.mean(np.abs(differences))), 6),
        "mean_signed_difference_percent": round(float(np.mean(percentages)), 4),
        "mean_absolute_difference_percent": round(float(np.mean(np.abs(percentages))), 4),
        "maximum_absolute_difference_m3": round(float(np.max(np.abs(differences))), 6),
        "count_new_larger": int(np.count_nonzero(differences > 0)),
        "count_new_smaller": int(np.count_nonzero(differences < 0)),
        "old_vs_new_volume": pearson_summary(old_values, new_values),
        "new_volume_vs_weight": pearson_summary(new_values, weights),
        "old_volume_vs_weight": pearson_summary(old_values, weights),
        "review_required_cows": [],
    }
    write_csv(SUMMARY_CSV, list(records[0].keys()), records)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "summary": summary,
                "parameters": {
                    "projection_grid_cell_size_m": GRID_CELL_M,
                    "projection_grid_cell_area_m2": GRID_CELL_AREA_M2,
                    "height_definition": "ground_plane_z_minus_cattle_surface_z",
                    "cell_height_aggregation": "median_height_within_each_1cm_xy_cell",
                    "volume_formula": "sum(cell_area_m2 * median_height_above_ground_m)",
                    "core_region": "extended geometric centerline adaptive-width boundaries",
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    update_main_csv({int(row["cow_id"]): row for row in records})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
