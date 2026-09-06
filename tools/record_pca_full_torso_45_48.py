#!/usr/bin/env python3
"""Record the accepted PCA-axis 45% rear / 48% head torso definition."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import (
    BASE,
    BIN_M,
    HEAD_RATIO,
    MAIN_CSV,
    REAR_RATIO,
    SMOOTHING_BINS,
    detect_head,
    detect_rear,
    pca_width_profile,
    read_ply_vertices,
)


METHOD_VERSION = "pca_first_below_width_ratio_rear45_head48_v1.0"
RECORDED_ON = "2026-09-05"
GRID_CELL_M = 0.01

FIELDS = [
    "pca_full_torso_analysis_eligible",
    "pca_full_torso_record_type",
    "pca_full_torso_method_version",
    "pca_full_torso_recorded_on",
    "pca_full_torso_axis_direction",
    "pca_full_torso_axis_centroid_x_m",
    "pca_full_torso_axis_centroid_y_m",
    "pca_full_torso_axis_unit_x",
    "pca_full_torso_axis_unit_y",
    "pca_full_torso_profile_bin_m",
    "pca_full_torso_profile_smoothing_m",
    "pca_full_torso_max_width_m",
    "pca_full_torso_max_width_axis_m",
    "pca_full_torso_rear_width_threshold_ratio",
    "pca_full_torso_head_width_threshold_ratio",
    "pca_full_torso_rear_boundary_axis_m",
    "pca_full_torso_head_boundary_axis_m",
    "pca_full_torso_rear_boundary_x_m",
    "pca_full_torso_rear_boundary_y_m",
    "pca_full_torso_head_boundary_x_m",
    "pca_full_torso_head_boundary_y_m",
    "pca_full_torso_core_length_m",
    "pca_full_torso_projected_area_m2",
    "pca_full_torso_projection_grid_cell_size_m",
    "pca_full_torso_projection_grid_cell_count",
    "pca_full_torso_core_point_count",
    "pca_full_torso_rear_boundary_reason",
    "pca_full_torso_head_boundary_reason",
    "pca_full_torso_boundary_status",
]


def projected_grid_area(points_xy: np.ndarray) -> tuple[int, float]:
    cells = np.floor(points_xy / GRID_CELL_M).astype(np.int64)
    count = len(np.unique(cells, axis=0))
    return count, count * GRID_CELL_M * GRID_CELL_M


def calculate(row: dict[str, str]) -> dict[str, object]:
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.array([float(row["centroid_x_m"]), float(row["centroid_y_m"])])
    axis = np.array([float(row["long_axis_unit_x"]), float(row["long_axis_unit_y"])])
    axis /= np.linalg.norm(axis)
    longitudinal, _, stations, widths = pca_width_profile(points_xy, centroid, axis)
    max_index = int(np.nanargmax(widths))
    max_width = float(widths[max_index])
    rear_index, rear_reason = detect_rear(widths, max_index, max_width)
    head_index, head_reason = detect_head(widths, max_index, max_width)
    if rear_index >= head_index:
        raise ValueError(f"invalid boundaries for cow {row['cow_id']}")

    rear_station = float(stations[rear_index])
    head_station = float(stations[head_index])
    core_mask = (longitudinal >= rear_station) & (longitudinal <= head_station)
    grid_count, area_m2 = projected_grid_area(points_xy[core_mask])
    rear_xy = centroid + rear_station * axis
    head_xy = centroid + head_station * axis
    status = "ok" if "兜底" not in rear_reason + head_reason else "review_required"

    return {
        "pca_full_torso_analysis_eligible": "TRUE",
        "pca_full_torso_record_type": "straight_pca_axis_threshold_boundaries",
        "pca_full_torso_method_version": METHOD_VERSION,
        "pca_full_torso_recorded_on": RECORDED_ON,
        "pca_full_torso_axis_direction": row.get("torso_head_direction") or "positive_long_axis_camera_x",
        "pca_full_torso_axis_centroid_x_m": round(float(centroid[0]), 6),
        "pca_full_torso_axis_centroid_y_m": round(float(centroid[1]), 6),
        "pca_full_torso_axis_unit_x": round(float(axis[0]), 9),
        "pca_full_torso_axis_unit_y": round(float(axis[1]), 9),
        "pca_full_torso_profile_bin_m": BIN_M,
        "pca_full_torso_profile_smoothing_m": BIN_M * SMOOTHING_BINS,
        "pca_full_torso_max_width_m": round(max_width, 6),
        "pca_full_torso_max_width_axis_m": round(float(stations[max_index]), 6),
        "pca_full_torso_rear_width_threshold_ratio": REAR_RATIO,
        "pca_full_torso_head_width_threshold_ratio": HEAD_RATIO,
        "pca_full_torso_rear_boundary_axis_m": round(rear_station, 6),
        "pca_full_torso_head_boundary_axis_m": round(head_station, 6),
        "pca_full_torso_rear_boundary_x_m": round(float(rear_xy[0]), 6),
        "pca_full_torso_rear_boundary_y_m": round(float(rear_xy[1]), 6),
        "pca_full_torso_head_boundary_x_m": round(float(head_xy[0]), 6),
        "pca_full_torso_head_boundary_y_m": round(float(head_xy[1]), 6),
        "pca_full_torso_core_length_m": round(head_station - rear_station, 6),
        "pca_full_torso_projected_area_m2": round(area_m2, 6),
        "pca_full_torso_projection_grid_cell_size_m": GRID_CELL_M,
        "pca_full_torso_projection_grid_cell_count": grid_count,
        "pca_full_torso_core_point_count": int(np.count_nonzero(core_mask)),
        "pca_full_torso_rear_boundary_reason": "first_below_45pct",
        "pca_full_torso_head_boundary_reason": (
            "first_below_48pct" if head_reason == "首次低于48%" else "head_endpoint_fallback"
        ),
        "pca_full_torso_boundary_status": status,
    }


def main() -> None:
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    for field in FIELDS:
        if field not in headers:
            headers.append(field)

    recorded = 0
    review_required: list[int] = []
    for row in rows:
        if row.get("analysis_include", "TRUE").upper() != "TRUE":
            for field in FIELDS:
                row[field] = "FALSE" if field == "pca_full_torso_analysis_eligible" else ""
            continue
        result = calculate(row)
        row.update({key: str(value) for key, value in result.items()})
        recorded += 1
        if result["pca_full_torso_boundary_status"] != "ok":
            review_required.append(int(row["cow_id"]))

    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print({"recorded": recorded, "review_required": review_required, "method": METHOD_VERSION})


if __name__ == "__main__":
    main()
