#!/usr/bin/env python3
"""Record Bi et al. (2023)-style top-view metrics for the 62 selected cattle."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_CSV = ROOT / "cow_visual_annotations.csv"
METHOD_VERSION = "bi2023_min_area_rect_mean_height_pca45_48_v1.0"
RECORDED_ON = "2026-09-06"
TARGET_COUNT = 62

FIELDS = [
    "bi2023_dorsal_length_m",
    "bi2023_abdominal_width_m",
    "bi2023_average_back_height_m",
    "bi2023_projected_volume_m3",
    "bi2023_projected_volume_l",
    "bi2023_length_width_method",
    "bi2023_height_method",
    "bi2023_volume_source_field",
    "bi2023_core_region",
    "bi2023_method_version",
    "bi2023_metric_status",
    "bi2023_recorded_on",
]


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def minimum_area_rectangle_dimensions(points_xy: np.ndarray) -> tuple[float, float]:
    hull = convex_hull(points_xy)
    if len(hull) < 3:
        raise ValueError("not enough points for minimum-area rectangle")
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2))
    best_area = math.inf
    best_dims = None
    for angle in angles:
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        rotated = hull @ rotation.T
        dims = np.ptp(rotated, axis=0)
        area = float(dims[0] * dims[1])
        if area < best_area:
            best_area = area
            best_dims = dims
    assert best_dims is not None
    return float(np.max(best_dims)), float(np.min(best_dims))


def calculate_metrics(row: dict[str, str]) -> dict[str, object]:
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
    core_xy = points_xy[core_mask]
    dorsal_length, abdominal_width = minimum_area_rectangle_dimensions(core_xy)
    status = "review_required_head_endpoint_fallback" if int(row["cow_id"]) == 150 else "ok"
    return {
        "bi2023_dorsal_length_m": round(dorsal_length, 6),
        "bi2023_abdominal_width_m": round(abdominal_width, 6),
        "bi2023_average_back_height_m": row["pca_full_torso_projection_mean_height_m"],
        "bi2023_projected_volume_m3": row["pca_full_torso_projected_volume_m3"],
        "bi2023_projected_volume_l": row["pca_full_torso_projected_volume_l"],
        "bi2023_length_width_method": "minimum_area_bounding_rectangle_long_and_short_sides",
        "bi2023_height_method": "mean_of_1cm_cell_median_heights_above_ground",
        "bi2023_volume_source_field": "pca_full_torso_projected_volume_m3",
        "bi2023_core_region": "PCA rear_first_below_45pct to head_first_below_48pct",
        "bi2023_method_version": METHOD_VERSION,
        "bi2023_metric_status": status,
        "bi2023_recorded_on": RECORDED_ON,
    }


def update_main_csv() -> None:
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    for field in FIELDS:
        if field not in headers:
            headers.append(field)

    selected = [row for row in rows if row.get("pca_full_torso_analysis_eligible") == "TRUE"]
    if len(selected) != TARGET_COUNT:
        raise ValueError(f"expected {TARGET_COUNT} selected cattle, found {len(selected)}")

    for row in rows:
        if row.get("pca_full_torso_analysis_eligible") != "TRUE":
            for field in FIELDS:
                row[field] = ""
            continue
        row.update({key: str(value) for key, value in calculate_metrics(row).items()})
        if int(row["cow_id"]) == 150:
            row.update(
                {
                    "analysis_data_quality_code": "pending",
                    "analysis_include": "TRUE",
                    "analysis_data_quality_label_zh": "待定",
                    "analysis_data_quality_source": "user_review",
                    "analysis_data_quality_annotated_on": RECORDED_ON,
                    "analysis_data_quality_notes": "头侧采用端点兜底，暂时搁置，待后续复核",
                }
            )

    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def update_annotation_csv() -> None:
    with ANNOTATION_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    found = False
    for row in rows:
        if int(row["cow_id"]) == 150:
            row.update(
                {
                    "analysis_data_quality_code": "pending",
                    "analysis_include": "TRUE",
                    "analysis_data_quality_label_zh": "待定",
                    "analysis_data_quality_source": "user_review",
                    "analysis_data_quality_annotated_on": RECORDED_ON,
                    "analysis_data_quality_notes": "头侧采用端点兜底，暂时搁置，待后续复核",
                }
            )
            found = True
    if not found:
        raise ValueError("cow 150 missing from annotation CSV")
    with ANNOTATION_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    update_main_csv()
    update_annotation_csv()
    print({"recorded_metrics": TARGET_COUNT, "pending_cows": [150], "method": METHOD_VERSION})


if __name__ == "__main__":
    main()
