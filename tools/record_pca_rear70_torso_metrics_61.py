#!/usr/bin/env python3
"""Record a second torso definition: rear PCA endpoint through 70% of full PCA length."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE / "pca_rear70_torso_metrics_61"
RECORDS_CSV = OUTPUT_DIR / "pca_rear70_torso_records_61.csv"
COMPARISON_CSV = OUTPUT_DIR / "pca_torso_definitions_comparison_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "pca_rear70_torso_summary.json"

TARGET_COUNT = 61
END_RATIO_FROM_REAR = 0.70
GRID_CELL_M = 0.01
GRID_CELL_AREA_M2 = GRID_CELL_M**2
METHOD_VERSION = "pca_rear_endpoint_to_70pct_1cm_column_sum_v1.0"
RECORDED_ON = "2026-09-06"

NEW_FIELDS = [
    "pca_rear70_torso_analysis_eligible",
    "pca_rear70_torso_method_version",
    "pca_rear70_torso_recorded_on",
    "pca_rear70_torso_start_ratio_from_rear",
    "pca_rear70_torso_end_ratio_from_rear",
    "pca_rear70_torso_full_length_m",
    "pca_rear70_torso_start_x_m",
    "pca_rear70_torso_start_y_m",
    "pca_rear70_torso_end_x_m",
    "pca_rear70_torso_end_y_m",
    "pca_rear70_torso_length_m",
    "pca_rear70_torso_projection_grid_cell_size_m",
    "pca_rear70_torso_projection_grid_cell_count",
    "pca_rear70_torso_projected_area_m2",
    "pca_rear70_torso_projection_mean_height_m",
    "pca_rear70_torso_projection_median_height_m",
    "pca_rear70_torso_projected_volume_m3",
    "pca_rear70_torso_projected_volume_l",
    "pca_rear70_torso_height_definition",
    "pca_rear70_torso_cell_height_aggregation",
    "pca_rear70_torso_region_definition",
    "pca_rear70_torso_status",
]


def unique_cell_median_heights(
    vertices: np.ndarray,
    plane_a: float,
    plane_b: float,
    plane_c: float,
) -> tuple[int, np.ndarray]:
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
    medians = np.asarray(
        [np.median(heights_sorted[start:end]) for start, end in zip(starts, ends)],
        dtype=float,
    )
    return len(unique_cells), medians


def calculate(row: dict[str, str]) -> dict[str, object]:
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    rear = np.asarray(
        [float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])],
        dtype=float,
    )
    head = np.asarray(
        [float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])],
        dtype=float,
    )
    full_axis = head - rear
    full_length = float(np.linalg.norm(full_axis))
    if full_length <= 0:
        raise ValueError(f"invalid PCA length for cow {row['cow_id']}")
    axis = full_axis / full_length
    end_distance = END_RATIO_FROM_REAR * full_length
    longitudinal = (points_xy - rear) @ axis
    core_mask = (longitudinal >= 0.0) & (longitudinal <= end_distance)
    core = vertices[core_mask]
    if len(core) == 0:
        raise ValueError(f"no rear-70% torso points for cow {row['cow_id']}")

    cell_count, cell_heights = unique_cell_median_heights(
        core,
        float(row["ground_plane_a"]),
        float(row["ground_plane_b"]),
        float(row["ground_plane_c_m"]),
    )
    if cell_count == 0:
        raise ValueError(f"no projection cells for cow {row['cow_id']}")
    end_xy = rear + end_distance * axis
    area_m2 = cell_count * GRID_CELL_AREA_M2
    volume_m3 = float(np.sum(cell_heights) * GRID_CELL_AREA_M2)
    return {
        "pca_rear70_torso_analysis_eligible": "TRUE",
        "pca_rear70_torso_method_version": METHOD_VERSION,
        "pca_rear70_torso_recorded_on": RECORDED_ON,
        "pca_rear70_torso_start_ratio_from_rear": 0.0,
        "pca_rear70_torso_end_ratio_from_rear": END_RATIO_FROM_REAR,
        "pca_rear70_torso_full_length_m": round(full_length, 6),
        "pca_rear70_torso_start_x_m": round(float(rear[0]), 6),
        "pca_rear70_torso_start_y_m": round(float(rear[1]), 6),
        "pca_rear70_torso_end_x_m": round(float(end_xy[0]), 6),
        "pca_rear70_torso_end_y_m": round(float(end_xy[1]), 6),
        "pca_rear70_torso_length_m": round(end_distance, 6),
        "pca_rear70_torso_projection_grid_cell_size_m": GRID_CELL_M,
        "pca_rear70_torso_projection_grid_cell_count": cell_count,
        "pca_rear70_torso_projected_area_m2": round(area_m2, 6),
        "pca_rear70_torso_projection_mean_height_m": round(float(np.mean(cell_heights)), 6),
        "pca_rear70_torso_projection_median_height_m": round(float(np.median(cell_heights)), 6),
        "pca_rear70_torso_projected_volume_m3": round(volume_m3, 6),
        "pca_rear70_torso_projected_volume_l": round(volume_m3 * 1000, 3),
        "pca_rear70_torso_height_definition": "ground_plane_z_minus_cattle_surface_z",
        "pca_rear70_torso_cell_height_aggregation": "median_height_within_each_1cm_xy_cell",
        "pca_rear70_torso_region_definition": "PCA rear endpoint 0pct through 70pct of full PCA endpoint length",
        "pca_rear70_torso_status": "ok",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    strict_rows = [
        row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible") == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(strict_rows) != TARGET_COUNT:
        raise ValueError(f"expected {TARGET_COUNT} strict cattle, found {len(strict_rows)}")

    for field in NEW_FIELDS:
        if field not in headers:
            headers.append(field)

    calculated_by_id: dict[int, dict[str, object]] = {}
    record_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for row in strict_rows:
        cow_id = int(row["cow_id"])
        result = calculate(row)
        calculated_by_id[cow_id] = result
        record_rows.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                **result,
                "source_top_ply": row["source_top_ply"],
                "extracted_top_ply": row["extracted_ply"],
            }
        )
        comparison_rows.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "existing_definition": "rear_first_below_45pct_to_head_first_below_48pct",
                "existing_torso_length_m": float(row["pca_full_torso_core_length_m"]),
                "existing_projected_area_m2": float(row["pca_full_torso_projected_area_m2"]),
                "existing_projected_volume_m3": float(row["pca_full_torso_projected_volume_m3"]),
                "new_definition": "pca_rear_endpoint_0pct_to_full_length_70pct",
                "new_torso_length_m": result["pca_rear70_torso_length_m"],
                "new_projected_area_m2": result["pca_rear70_torso_projected_area_m2"],
                "new_projected_volume_m3": result["pca_rear70_torso_projected_volume_m3"],
                "length_difference_new_minus_existing_m": round(
                    float(result["pca_rear70_torso_length_m"])
                    - float(row["pca_full_torso_core_length_m"]),
                    6,
                ),
                "area_difference_new_minus_existing_m2": round(
                    float(result["pca_rear70_torso_projected_area_m2"])
                    - float(row["pca_full_torso_projected_area_m2"]),
                    6,
                ),
                "volume_difference_new_minus_existing_m3": round(
                    float(result["pca_rear70_torso_projected_volume_m3"])
                    - float(row["pca_full_torso_projected_volume_m3"]),
                    6,
                ),
            }
        )

    for row in rows:
        cow_id = int(row["cow_id"])
        if cow_id in calculated_by_id:
            row.update({key: str(value) for key, value in calculated_by_id[cow_id].items()})
        else:
            for field in NEW_FIELDS:
                row[field] = "FALSE" if field == "pca_rear70_torso_analysis_eligible" else ""

    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    record_rows.sort(key=lambda item: int(item["cow_id"]))
    comparison_rows.sort(key=lambda item: int(item["cow_id"]))
    write_csv(RECORDS_CSV, record_rows)
    write_csv(COMPARISON_CSV, comparison_rows)

    summary = {
        "cattle_count": TARGET_COUNT,
        "existing_definition": "PCA rear first below 45% max width to head first below 48% max width",
        "new_definition": "PCA rear endpoint through 70% of full PCA endpoint length",
        "new_fields": NEW_FIELDS,
        "grid_cell_size_m": GRID_CELL_M,
        "height_definition": "ground plane height minus cattle surface depth",
        "cell_height_aggregation": "median within each 1cm x 1cm XY cell",
        "method_version": METHOD_VERSION,
        "recorded_on": RECORDED_ON,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
