#!/usr/bin/env python3
"""Compute core torso projected area using the extended geometric centerline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
AXIS_DIR = BASE / "skeleton_axis_comparison"
MAIN_CSV = BASE / "long_axis_records.csv"
WIDTH_SUMMARY_CSV = AXIS_DIR / "extended_geometric_max_width_records.csv"
WIDTH_PROFILE_CSV = AXIS_DIR / "extended_geometric_width_profiles.csv"
SUMMARY_CSV = AXIS_DIR / "extended_geometric_core_area_records.csv"
COMPARISON_JSON = AXIS_DIR / "extended_geometric_core_area_comparison.json"
CORE_PLY_DIR = BASE / "extended_geometric_core_torso_point_clouds"

METHOD_VERSION = "extended_geometric_adaptive_width_core_area_v1.0"
HEAD_WIDTH_RATIO = 0.575
HEAD_CONTINUOUS_LENGTH_M = 0.05
BUTT_WIDTH_RATIO = 0.625
BUTT_CONTINUOUS_LENGTH_M = 0.03
IMMEDIATE_STOP_RATIO = 0.50
PROFILE_STEP_M = 0.01
GRID_CELL_M = 0.01

PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]
)


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count: {path}")
        return np.fromfile(handle, dtype=PLY_DTYPE, count=vertex_count)


def write_ply(path: Path, vertices: np.ndarray, cow_id: int):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment Core torso points for cow {cow_id}\n"
        f"comment Method {METHOD_VERSION}\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def detect_boundary(widths, max_index, direction, threshold_ratio, continuous_length_m, max_width):
    required = max(1, int(math.ceil(continuous_length_m / PROFILE_STEP_M)))
    indices = range(max_index - 1, -1, -1) if direction == "butt" else range(max_index + 1, len(widths))
    run = []
    for index in indices:
        width = widths[index]
        if not np.isfinite(width):
            run = []
            continue
        ratio = width / max_width
        if ratio < IMMEDIATE_STOP_RATIO:
            return index, "immediate_below_50pct"
        if ratio < threshold_ratio:
            run.append(index)
            if len(run) >= required:
                return run[0], "continuous_threshold"
        else:
            run = []
    endpoint = 0 if direction == "butt" else len(widths) - 1
    return endpoint, "endpoint_fallback"


def nearest_centerline_station(points_xy: np.ndarray, centers_xy: np.ndarray, stations: np.ndarray):
    assigned = np.empty(len(points_xy), dtype=float)
    for start in range(0, len(points_xy), 3000):
        batch = points_xy[start : start + 3000]
        distance_sq = ((batch[:, None, :] - centers_xy[None, :, :]) ** 2).sum(axis=2)
        assigned[start : start + len(batch)] = stations[np.argmin(distance_sq, axis=1)]
    return assigned


def projected_grid_area(points_xy: np.ndarray):
    cells = np.floor(points_xy / GRID_CELL_M).astype(np.int64)
    unique_cells = np.unique(cells, axis=0)
    return len(unique_cells), len(unique_cells) * GRID_CELL_M * GRID_CELL_M


def update_main_csv(summary_by_cow):
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    fields = [
        "extended_geometric_torso_butt_boundary_station_m",
        "extended_geometric_torso_head_boundary_station_m",
        "extended_geometric_torso_butt_boundary_x_m",
        "extended_geometric_torso_butt_boundary_y_m",
        "extended_geometric_torso_head_boundary_x_m",
        "extended_geometric_torso_head_boundary_y_m",
        "extended_geometric_torso_core_length_m",
        "extended_geometric_torso_projected_area_m2",
        "extended_geometric_torso_area_difference_vs_pca_m2",
        "extended_geometric_torso_area_absolute_difference_vs_pca_m2",
        "extended_geometric_torso_area_difference_vs_pca_percent",
        "extended_geometric_torso_core_point_count",
        "extended_geometric_torso_core_ply",
        "extended_geometric_torso_head_boundary_reason",
        "extended_geometric_torso_butt_boundary_reason",
        "extended_geometric_torso_boundary_status",
        "extended_geometric_torso_method_version",
    ]
    for field in fields:
        if field not in headers:
            headers.append(field)
    for row in rows:
        result = summary_by_cow[int(row["cow_id"])]
        row.update(
            {
                "extended_geometric_torso_butt_boundary_station_m": result["butt_boundary_station_m"],
                "extended_geometric_torso_head_boundary_station_m": result["head_boundary_station_m"],
                "extended_geometric_torso_butt_boundary_x_m": result["butt_boundary_x_m"],
                "extended_geometric_torso_butt_boundary_y_m": result["butt_boundary_y_m"],
                "extended_geometric_torso_head_boundary_x_m": result["head_boundary_x_m"],
                "extended_geometric_torso_head_boundary_y_m": result["head_boundary_y_m"],
                "extended_geometric_torso_core_length_m": result["core_length_m"],
                "extended_geometric_torso_projected_area_m2": result["new_geometric_core_area_m2"],
                "extended_geometric_torso_area_difference_vs_pca_m2": result["difference_new_minus_pca_m2"],
                "extended_geometric_torso_area_absolute_difference_vs_pca_m2": result["absolute_difference_m2"],
                "extended_geometric_torso_area_difference_vs_pca_percent": result["difference_vs_pca_percent"],
                "extended_geometric_torso_core_point_count": result["core_point_count"],
                "extended_geometric_torso_core_ply": result["core_ply"],
                "extended_geometric_torso_head_boundary_reason": result["head_boundary_reason"],
                "extended_geometric_torso_butt_boundary_reason": result["butt_boundary_reason"],
                "extended_geometric_torso_boundary_status": result["status"],
                "extended_geometric_torso_method_version": METHOD_VERSION,
            }
        )
    write_csv(MAIN_CSV, headers, rows)


def main():
    CORE_PLY_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    rows_by_cow = {int(row["cow_id"]): row for row in main_rows}
    with WIDTH_SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        width_summaries = {int(row["cow_id"]): row for row in csv.DictReader(handle)}
    profiles = {}
    with WIDTH_PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            profiles.setdefault(int(row["cow_id"]), []).append(row)

    records = []
    for cow_id in sorted(width_summaries):
        main_row = rows_by_cow[cow_id]
        width_summary = width_summaries[cow_id]
        profile = profiles[cow_id]
        stations = np.asarray([float(row["station_from_rear_m"]) for row in profile])
        centers = np.asarray([[float(row["center_x_m"]), float(row["center_y_m"])] for row in profile])
        widths = np.asarray([float(row["smoothed_width_m"]) if row["smoothed_width_m"] else np.nan for row in profile])
        max_width = float(width_summary["extended_geometric_max_width_m"])
        max_station = float(width_summary["max_width_station_m"])
        max_index = int(np.argmin(np.abs(stations - max_station)))
        butt_index, butt_reason = detect_boundary(widths, max_index, "butt", BUTT_WIDTH_RATIO, BUTT_CONTINUOUS_LENGTH_M, max_width)
        head_index, head_reason = detect_boundary(widths, max_index, "head", HEAD_WIDTH_RATIO, HEAD_CONTINUOUS_LENGTH_M, max_width)
        if butt_index >= head_index:
            raise ValueError(f"invalid boundary order for cow {cow_id}: {butt_index} >= {head_index}")

        cattle = read_ply_vertices(BASE / main_row["extracted_ply"])
        cattle_xy = np.column_stack([cattle["x"], cattle["y"]]).astype(float)
        assigned_station = nearest_centerline_station(cattle_xy, centers, stations)
        butt_station = float(stations[butt_index])
        head_station = float(stations[head_index])
        core_mask = (assigned_station >= butt_station) & (assigned_station <= head_station)
        core = cattle[core_mask]
        grid_cell_count, new_area = projected_grid_area(cattle_xy[core_mask])
        old_area = float(main_row["torso_projected_area_m2"])
        difference = new_area - old_area
        core_ply_path = CORE_PLY_DIR / f"cow_{cow_id:03d}_core_torso.ply"
        write_ply(core_ply_path, core, cow_id)
        status = "ok" if butt_reason != "endpoint_fallback" and head_reason != "endpoint_fallback" else "review_required"
        record = {
            "cow_id": cow_id,
            "old_pca_core_area_m2": round(old_area, 6),
            "old_pca_area_field": "torso_projected_area_m2",
            "new_geometric_core_area_m2": round(new_area, 6),
            "difference_new_minus_pca_m2": round(difference, 6),
            "absolute_difference_m2": round(abs(difference), 6),
            "difference_vs_pca_percent": round(difference / old_area * 100, 4),
            "absolute_difference_vs_pca_percent": round(abs(difference) / old_area * 100, 4),
            "new_max_body_width_m": round(max_width, 6),
            "butt_boundary_station_m": round(butt_station, 6),
            "head_boundary_station_m": round(head_station, 6),
            "core_length_m": round(head_station - butt_station, 6),
            "butt_boundary_x_m": round(float(centers[butt_index, 0]), 6),
            "butt_boundary_y_m": round(float(centers[butt_index, 1]), 6),
            "head_boundary_x_m": round(float(centers[head_index, 0]), 6),
            "head_boundary_y_m": round(float(centers[head_index, 1]), 6),
            "butt_boundary_reason": butt_reason,
            "head_boundary_reason": head_reason,
            "core_point_count": int(np.count_nonzero(core_mask)),
            "projection_grid_cell_size_m": GRID_CELL_M,
            "projection_grid_cell_count": grid_cell_count,
            "core_ply": str(core_ply_path.relative_to(ROOT)),
            "method_version": METHOD_VERSION,
            "status": status,
        }
        records.append(record)

    old_values = np.asarray([row["old_pca_core_area_m2"] for row in records], dtype=float)
    new_values = np.asarray([row["new_geometric_core_area_m2"] for row in records], dtype=float)
    differences = new_values - old_values
    absolute = np.abs(differences)
    percentages = differences / old_values * 100
    diff_sd = float(np.std(differences, ddof=1))
    comparison = {
        "n": len(records),
        "old_measure": "torso_projected_area_m2 (PCA-axis core boundaries)",
        "new_measure": "extended_geometric_torso_projected_area_m2 (curved-centerline core boundaries)",
        "method_version": METHOD_VERSION,
        "mean_old_area_m2": round(float(np.mean(old_values)), 6),
        "mean_new_area_m2": round(float(np.mean(new_values)), 6),
        "mean_signed_difference_m2": round(float(np.mean(differences)), 6),
        "median_signed_difference_m2": round(float(np.median(differences)), 6),
        "mean_absolute_difference_m2": round(float(np.mean(absolute)), 6),
        "median_absolute_difference_m2": round(float(np.median(absolute)), 6),
        "rmse_difference_m2": round(float(np.sqrt(np.mean(differences**2))), 6),
        "mean_signed_difference_percent": round(float(np.mean(percentages)), 4),
        "mean_absolute_difference_percent": round(float(np.mean(np.abs(percentages))), 4),
        "maximum_absolute_difference_m2": round(float(np.max(absolute)), 6),
        "pearson_correlation_old_vs_new": round(float(np.corrcoef(old_values, new_values)[0, 1]), 6),
        "difference_standard_deviation_m2": round(diff_sd, 6),
        "agreement_lower_95_m2": round(float(np.mean(differences) - 1.96 * diff_sd), 6),
        "agreement_upper_95_m2": round(float(np.mean(differences) + 1.96 * diff_sd), 6),
        "count_new_larger": int(np.count_nonzero(differences > 0)),
        "count_new_smaller": int(np.count_nonzero(differences < 0)),
        "count_absolute_difference_over_0_05_m2": int(np.count_nonzero(absolute > 0.05)),
        "count_absolute_percent_over_10": int(np.count_nonzero(np.abs(percentages) > 10)),
        "review_required_cows": [row["cow_id"] for row in records if row["status"] != "ok"],
    }
    write_csv(SUMMARY_CSV, list(records[0].keys()), records)
    COMPARISON_JSON.write_text(
        json.dumps(
            {
                "comparison": comparison,
                "parameters": {
                    "head_width_threshold_ratio": HEAD_WIDTH_RATIO,
                    "head_continuous_length_m": HEAD_CONTINUOUS_LENGTH_M,
                    "butt_width_threshold_ratio": BUTT_WIDTH_RATIO,
                    "butt_continuous_length_m": BUTT_CONTINUOUS_LENGTH_M,
                    "immediate_stop_ratio": IMMEDIATE_STOP_RATIO,
                    "profile_step_m": PROFILE_STEP_M,
                    "projection_grid_cell_size_m": GRID_CELL_M,
                    "point_assignment": "nearest station on extended geometric centerline",
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
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
