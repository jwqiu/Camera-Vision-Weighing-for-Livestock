#!/usr/bin/env python3
"""Record 11-station width and height metrics for two PCA torso definitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


OUTPUT_DIR = BASE / "pca_two_torso_11_station_metrics_61"
STATION_CSV = OUTPUT_DIR / "pca_two_torso_11_station_records_61.csv"
WIDE_CSV = OUTPUT_DIR / "pca_two_torso_11_station_summary_61.csv"
METHOD_JSON = OUTPUT_DIR / "pca_two_torso_11_station_method.json"

TARGET_CATTLE = 61
STATION_RATIOS = np.linspace(0.05, 0.95, 11)
SLAB_HALF_WIDTH_M = 0.005
HEIGHT_CENTER_HALF_WIDTH_M = 0.025
WIDTH_EDGE_QUANTILE = 0.02
CLUSTER_GAP_M = 0.02
METHOD_VERSION = "two_pca_torso_definitions_11_stations_width_height_v1.0"
RECORDED_ON = "2026-09-06"

DEFINITIONS = (
    ("pca_4548", "rear_first_below_45pct_to_head_first_below_48pct"),
    ("pca_rear70", "pca_rear_endpoint_0pct_to_full_length_70pct"),
)

SUMMARY_FIELDS = [
    "pca_full_torso_11pt_mean_width_m",
    "pca_full_torso_11pt_median_width_m",
    "pca_full_torso_11pt_mean_height_m",
    "pca_full_torso_11pt_median_height_m",
    "pca_full_torso_11pt_metric_status",
    "pca_rear70_torso_11pt_mean_width_m",
    "pca_rear70_torso_11pt_median_width_m",
    "pca_rear70_torso_11pt_mean_height_m",
    "pca_rear70_torso_11pt_median_height_m",
    "pca_rear70_torso_11pt_metric_status",
    "pca_two_torso_11pt_station_ratios",
    "pca_two_torso_11pt_width_definition",
    "pca_two_torso_11pt_height_definition",
    "pca_two_torso_11pt_method_version",
    "pca_two_torso_11pt_recorded_on",
]


def select_central_cluster(across: np.ndarray) -> np.ndarray:
    ordered = np.sort(across[np.isfinite(across)])
    if len(ordered) < 8:
        return ordered
    groups = np.split(ordered, np.where(np.diff(ordered) > CLUSTER_GAP_M)[0] + 1)
    groups = [group for group in groups if len(group) >= 8]
    if not groups:
        return ordered
    containing_axis = [group for group in groups if group[0] <= 0 <= group[-1]]
    if containing_axis:
        return max(containing_axis, key=len)
    return min(groups, key=lambda group: abs(float(np.median(group))))


def torso_endpoints(row: dict[str, str], code: str) -> tuple[np.ndarray, np.ndarray]:
    if code == "pca_4548":
        start = np.asarray(
            [float(row["pca_full_torso_rear_boundary_x_m"]), float(row["pca_full_torso_rear_boundary_y_m"])],
            dtype=float,
        )
        end = np.asarray(
            [float(row["pca_full_torso_head_boundary_x_m"]), float(row["pca_full_torso_head_boundary_y_m"])],
            dtype=float,
        )
        return start, end
    if code == "pca_rear70":
        start = np.asarray(
            [float(row["pca_rear70_torso_start_x_m"]), float(row["pca_rear70_torso_start_y_m"])],
            dtype=float,
        )
        end = np.asarray(
            [float(row["pca_rear70_torso_end_x_m"]), float(row["pca_rear70_torso_end_y_m"])],
            dtype=float,
        )
        return start, end
    raise ValueError(code)


def station_measurement(
    points_xy: np.ndarray,
    heights: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> dict[str, object]:
    normal = np.asarray([-axis[1], axis[0]])
    delta = points_xy - center
    along = delta @ axis
    across = delta @ normal
    selected = np.abs(along) <= SLAB_HALF_WIDTH_M
    if np.count_nonzero(selected) < 8:
        selected = np.abs(along) <= 2 * SLAB_HALF_WIDTH_M
    if np.count_nonzero(selected) < 8:
        selected = np.abs(along) <= 3 * SLAB_HALF_WIDTH_M
    finite = selected & np.isfinite(across) & np.isfinite(heights) & (heights > 0)
    cluster = select_central_cluster(across[finite])
    if len(cluster) < 2:
        raise ValueError("insufficient cross-section points")
    low = float(np.quantile(cluster, WIDTH_EDGE_QUANTILE))
    high = float(np.quantile(cluster, 1 - WIDTH_EDGE_QUANTILE))

    center_patch = finite & (np.abs(across) <= HEIGHT_CENTER_HALF_WIDTH_M)
    if np.count_nonzero(center_patch) < 3:
        candidates = np.flatnonzero(finite)
        if len(candidates) == 0:
            raise ValueError("insufficient height points")
        nearest = candidates[np.argsort(np.abs(across[candidates]))[: min(7, len(candidates))]]
        local_heights = heights[nearest]
        height_method = "median_of_nearest_up_to_7_points_to_cross_section_midpoint"
    else:
        local_heights = heights[center_patch]
        height_method = "median_in_1cm_longitudinal_by_5cm_center_patch"
    return {
        "width_m": high - low,
        "left_offset_m": low,
        "right_offset_m": high,
        "height_m": float(np.median(local_heights)),
        "cross_section_point_count": int(np.count_nonzero(finite)),
        "height_point_count": int(len(local_heights)),
        "height_station_method": height_method,
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
    strict = [
        row
        for row in rows
        if row.get("pca_rear70_torso_analysis_eligible") == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(strict) != TARGET_CATTLE:
        raise ValueError(f"expected {TARGET_CATTLE} cattle, found {len(strict)}")

    station_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    summary_by_cow: dict[int, dict[str, object]] = {}
    for row in sorted(strict, key=lambda item: int(item["cow_id"])):
        cow_id = int(row["cow_id"])
        vertices = read_ply_vertices(BASE / row["extracted_ply"])
        points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
        plane_z = (
            float(row["ground_plane_a"]) * vertices["x"].astype(float)
            + float(row["ground_plane_b"]) * vertices["y"].astype(float)
            + float(row["ground_plane_c_m"])
        )
        heights = plane_z - vertices["z"].astype(float)
        full_rear = np.asarray(
            [float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])], dtype=float
        )
        full_head = np.asarray(
            [float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])], dtype=float
        )
        full_vector = full_head - full_rear
        full_length = float(np.linalg.norm(full_vector))
        full_axis = full_vector / full_length

        wide: dict[str, object] = {
            "cow_id": cow_id,
            "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
        }
        master_summary: dict[str, object] = {}
        for code, definition in DEFINITIONS:
            start, end = torso_endpoints(row, code)
            torso_vector = end - start
            torso_length = float(np.linalg.norm(torso_vector))
            axis = torso_vector / torso_length
            widths: list[float] = []
            local_heights: list[float] = []
            for station_index, ratio in enumerate(STATION_RATIOS, start=1):
                center = start + float(ratio) * torso_vector
                measured = station_measurement(points_xy, heights, center, axis)
                width = float(measured["width_m"])
                height = float(measured["height_m"])
                widths.append(width)
                local_heights.append(height)
                ratio_full = float((center - full_rear) @ full_axis / full_length)
                station_rows.append(
                    {
                        "cow_id": cow_id,
                        "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                        "torso_definition_code": code,
                        "torso_definition": definition,
                        "torso_length_m": round(torso_length, 6),
                        "station_index": station_index,
                        "station_ratio_within_torso": round(float(ratio), 4),
                        "station_ratio_full_pca": round(ratio_full, 6),
                        "station_center_x_m": round(float(center[0]), 6),
                        "station_center_y_m": round(float(center[1]), 6),
                        "width_m": round(width, 6),
                        "height_m": round(height, 6),
                        "left_offset_m": round(float(measured["left_offset_m"]), 6),
                        "right_offset_m": round(float(measured["right_offset_m"]), 6),
                        "cross_section_point_count": measured["cross_section_point_count"],
                        "height_point_count": measured["height_point_count"],
                        "height_station_method": measured["height_station_method"],
                        "method_version": METHOD_VERSION,
                        "status": "ok",
                    }
                )
                wide[f"{code}_width_{station_index:02d}_m"] = round(width, 6)
                wide[f"{code}_height_{station_index:02d}_m"] = round(height, 6)

            mean_width = float(np.mean(widths))
            median_width = float(np.median(widths))
            mean_height = float(np.mean(local_heights))
            median_height = float(np.median(local_heights))
            wide[f"{code}_mean_width_m"] = round(mean_width, 6)
            wide[f"{code}_median_width_m"] = round(median_width, 6)
            wide[f"{code}_mean_height_m"] = round(mean_height, 6)
            wide[f"{code}_median_height_m"] = round(median_height, 6)
            wide[f"{code}_status"] = "ok"
            if code == "pca_4548":
                prefix = "pca_full_torso_11pt"
            else:
                prefix = "pca_rear70_torso_11pt"
            master_summary.update(
                {
                    f"{prefix}_mean_width_m": round(mean_width, 6),
                    f"{prefix}_median_width_m": round(median_width, 6),
                    f"{prefix}_mean_height_m": round(mean_height, 6),
                    f"{prefix}_median_height_m": round(median_height, 6),
                    f"{prefix}_metric_status": "ok",
                }
            )
        wide["method_version"] = METHOD_VERSION
        wide["recorded_on"] = RECORDED_ON
        wide_rows.append(wide)
        summary_by_cow[cow_id] = master_summary

    common_metadata = {
        "pca_two_torso_11pt_station_ratios": "0.05,0.14,0.23,0.32,0.41,0.50,0.59,0.68,0.77,0.86,0.95",
        "pca_two_torso_11pt_width_definition": "2pct_to_98pct_width_of_central_contiguous_points_in_1cm_PCA_normal_slab",
        "pca_two_torso_11pt_height_definition": "median_ground_relative_height_in_5cm_center_patch_at_same_station",
        "pca_two_torso_11pt_method_version": METHOD_VERSION,
        "pca_two_torso_11pt_recorded_on": RECORDED_ON,
    }
    for field in SUMMARY_FIELDS:
        if field not in headers:
            headers.append(field)
    for row in rows:
        cow_id = int(row["cow_id"])
        if cow_id in summary_by_cow:
            row.update({key: str(value) for key, value in summary_by_cow[cow_id].items()})
            row.update(common_metadata)
        else:
            for field in SUMMARY_FIELDS:
                row[field] = ""

    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    write_csv(STATION_CSV, station_rows)
    write_csv(WIDE_CSV, wide_rows)

    method = {
        "cattle_count": TARGET_CATTLE,
        "torso_definitions": {code: definition for code, definition in DEFINITIONS},
        "station_count_per_definition": 11,
        "station_ratios_within_each_torso": [round(float(value), 2) for value in STATION_RATIOS],
        "endpoint_exclusion_each_side": "5% of each torso definition",
        "width": {
            "view": "topview point cloud XY projection",
            "cross_section": "perpendicular to the PCA centerline",
            "slab_thickness_m": 2 * SLAB_HALF_WIDTH_M,
            "edge_quantiles": [WIDTH_EDGE_QUANTILE, 1 - WIDTH_EDGE_QUANTILE],
            "cluster_gap_m": CLUSTER_GAP_M,
        },
        "height": {
            "view": "topview depth point cloud",
            "definition": "ground plane minus cattle surface depth at cross-section midpoint",
            "robust_implementation": "median height in 1cm longitudinal by 5cm lateral center patch",
        },
        "aggregates": ["arithmetic mean of 11 positions", "median of 11 positions"],
        "method_version": METHOD_VERSION,
        "recorded_on": RECORDED_ON,
    }
    METHOD_JSON.write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cattle": len(wide_rows), "station_rows": len(station_rows), **method}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
