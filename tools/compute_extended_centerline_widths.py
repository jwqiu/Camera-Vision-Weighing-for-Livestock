#!/usr/bin/env python3
"""Compute local cross-section widths along the rear-extended centerline."""

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
CENTERLINE_JSON = AXIS_DIR / "extended_geometric_centerline_records.json"
SUMMARY_CSV = AXIS_DIR / "extended_geometric_max_width_records.csv"
PROFILE_CSV = AXIS_DIR / "extended_geometric_width_profiles.csv"
COMPARISON_JSON = AXIS_DIR / "extended_geometric_max_width_comparison.json"

METHOD_VERSION = "extended_geometric_local_cross_section_width_v1.0"
PROFILE_STEP_M = 0.01
SLAB_HALF_THICKNESS_M = 0.005
LINE_SMOOTHING_M = 0.05
WIDTH_SMOOTHING_M = 0.03
EDGE_QUANTILE = 0.01
CLUSTER_GAP_M = 0.02
REAR_MAX_WIDTH_EXCLUSION_M = 0.05
HEAD_MAX_WIDTH_EXCLUSION_M = 0.25

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


def write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def moving_average_xy(xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return xy.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(xy, ((pad_left, pad_right), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)])


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for index in range(len(values)):
        lo, hi = max(0, index - half), min(len(values), index + half + 1)
        local = values[lo:hi]
        finite = local[np.isfinite(local)]
        if len(finite):
            result[index] = float(np.median(finite))
    return result


def resample_centerline(points_xy: np.ndarray):
    segments = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-6])
    cumulative = cumulative[keep]
    points_xy = points_xy[keep]
    stations = np.arange(0.0, cumulative[-1] + PROFILE_STEP_M * 0.5, PROFILE_STEP_M)
    stations[-1] = min(stations[-1], cumulative[-1])
    resampled = np.column_stack(
        [np.interp(stations, cumulative, points_xy[:, axis]) for axis in range(2)]
    )
    smooth_window = max(3, int(round(LINE_SMOOTHING_M / PROFILE_STEP_M)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = moving_average_xy(resampled, smooth_window)
    return stations, smoothed


def select_contiguous_width(across: np.ndarray):
    if len(across) < 8:
        return None
    ordered = np.sort(across)
    split_indices = np.where(np.diff(ordered) > CLUSTER_GAP_M)[0] + 1
    groups = np.split(ordered, split_indices)
    groups = [group for group in groups if len(group) >= 8]
    if not groups:
        return None
    containing = [group for group in groups if group[0] <= 0 <= group[-1]]
    if containing:
        selected = max(containing, key=len)
    else:
        selected = min(groups, key=lambda group: abs(float(np.median(group))))
    low = float(np.quantile(selected, EDGE_QUANTILE))
    high = float(np.quantile(selected, 1 - EDGE_QUANTILE))
    return high - low, low, high, len(selected)


def compute_profile(centerline_xy: np.ndarray, cattle_xy: np.ndarray):
    stations, smooth_xy = resample_centerline(centerline_xy)
    raw_widths = np.full(len(stations), np.nan)
    lows = np.full(len(stations), np.nan)
    highs = np.full(len(stations), np.nan)
    counts = np.zeros(len(stations), dtype=int)
    tangents = np.zeros_like(smooth_xy)
    for index, center in enumerate(smooth_xy):
        lo_index = max(0, index - 2)
        hi_index = min(len(smooth_xy) - 1, index + 2)
        tangent = smooth_xy[hi_index] - smooth_xy[lo_index]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-8:
            continue
        tangent /= norm
        tangents[index] = tangent
        normal = np.array([-tangent[1], tangent[0]])
        delta = cattle_xy - center
        along = delta @ tangent
        selected = np.abs(along) <= SLAB_HALF_THICKNESS_M
        if np.count_nonzero(selected) < 8:
            continue
        across = delta[selected] @ normal
        result = select_contiguous_width(across)
        if result is None:
            continue
        raw_widths[index], lows[index], highs[index], counts[index] = result
    width_window = max(3, int(round(WIDTH_SMOOTHING_M / PROFILE_STEP_M)))
    if width_window % 2 == 0:
        width_window += 1
    smoothed_widths = rolling_median(raw_widths, width_window)
    return stations, smooth_xy, tangents, raw_widths, smoothed_widths, lows, highs, counts


def update_main_csv(summary_by_cow):
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    fields = [
        "extended_geometric_max_width_m",
        "extended_geometric_max_width_station_m",
        "extended_geometric_max_width_center_x_m",
        "extended_geometric_max_width_center_y_m",
        "extended_geometric_max_width_method_version",
        "extended_geometric_max_width_status",
        "extended_geometric_max_width_difference_vs_pca_m",
        "extended_geometric_max_width_absolute_difference_vs_pca_m",
        "extended_geometric_max_width_difference_vs_pca_percent",
        "extended_geometric_width_profile_file",
    ]
    for field in fields:
        if field not in headers:
            headers.append(field)
    for row in rows:
        result = summary_by_cow[int(row["cow_id"])]
        row.update(
            {
                "extended_geometric_max_width_m": result["extended_geometric_max_width_m"],
                "extended_geometric_max_width_station_m": result["max_width_station_m"],
                "extended_geometric_max_width_center_x_m": result["max_width_center_x_m"],
                "extended_geometric_max_width_center_y_m": result["max_width_center_y_m"],
                "extended_geometric_max_width_method_version": METHOD_VERSION,
                "extended_geometric_max_width_status": result["status"],
                "extended_geometric_max_width_difference_vs_pca_m": result["difference_new_minus_pca_m"],
                "extended_geometric_max_width_absolute_difference_vs_pca_m": result["absolute_difference_m"],
                "extended_geometric_max_width_difference_vs_pca_percent": result["difference_vs_pca_percent"],
                "extended_geometric_width_profile_file": str(PROFILE_CSV.relative_to(ROOT)),
            }
        )
    write_csv(MAIN_CSV, headers, rows)


def main():
    centerline_payload = json.loads(CENTERLINE_JSON.read_text(encoding="utf-8"))
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    rows_by_cow = {int(row["cow_id"]): row for row in main_rows}
    centerlines = {int(record["cow_id"]): record for record in centerline_payload["records"]}
    summary_rows = []
    profile_rows = []

    for cow_id in sorted(centerlines):
        row = rows_by_cow[cow_id]
        record = centerlines[cow_id]
        centerline_xy = np.asarray([[point["camera_x_m"], point["camera_y_m"]] for point in record["points"]])
        cattle = read_ply_vertices(BASE / row["extracted_ply"])
        cattle_xy = np.column_stack([cattle["x"], cattle["y"]]).astype(float)
        profile = compute_profile(centerline_xy, cattle_xy)
        stations, centers, tangents, raw, smooth, lows, highs, counts = profile
        valid = np.isfinite(smooth)
        if not np.any(valid):
            raise ValueError(f"no valid width profile for cow {cow_id}")
        search_start = REAR_MAX_WIDTH_EXCLUSION_M
        search_end = float(stations[-1] - HEAD_MAX_WIDTH_EXCLUSION_M)
        search_region = valid & (stations >= search_start) & (stations <= search_end)
        if not np.any(search_region):
            raise ValueError(f"no valid torso max-width search region for cow {cow_id}")
        searchable_widths = np.where(search_region, smooth, np.nan)
        max_index = int(np.nanargmax(searchable_widths))
        new_width = float(smooth[max_index])
        old_width = float(row["torso_max_body_width_m"])
        difference = new_width - old_width
        result = {
            "cow_id": cow_id,
            "old_pca_max_body_width_m": round(old_width, 6),
            "old_pca_width_field": "torso_max_body_width_m",
            "extended_geometric_max_width_m": round(new_width, 6),
            "difference_new_minus_pca_m": round(difference, 6),
            "absolute_difference_m": round(abs(difference), 6),
            "difference_vs_pca_percent": round(difference / old_width * 100, 4),
            "absolute_difference_vs_pca_percent": round(abs(difference) / old_width * 100, 4),
            "max_width_station_m": round(float(stations[max_index]), 6),
            "max_width_center_x_m": round(float(centers[max_index, 0]), 6),
            "max_width_center_y_m": round(float(centers[max_index, 1]), 6),
            "max_width_search_start_m": round(search_start, 6),
            "max_width_search_end_m": round(search_end, 6),
            "profile_station_count": len(stations),
            "valid_profile_station_count": int(np.count_nonzero(valid)),
            "method_version": METHOD_VERSION,
            "status": "ok" if np.count_nonzero(valid) >= 0.8 * len(stations) else "review_required",
        }
        summary_rows.append(result)
        for index in range(len(stations)):
            profile_rows.append(
                {
                    "cow_id": cow_id,
                    "station_index": index,
                    "station_from_rear_m": round(float(stations[index]), 6),
                    "center_x_m": round(float(centers[index, 0]), 6),
                    "center_y_m": round(float(centers[index, 1]), 6),
                    "tangent_x": round(float(tangents[index, 0]), 8),
                    "tangent_y": round(float(tangents[index, 1]), 8),
                    "raw_width_m": round(float(raw[index]), 6) if np.isfinite(raw[index]) else "",
                    "smoothed_width_m": round(float(smooth[index]), 6) if np.isfinite(smooth[index]) else "",
                    "left_offset_m": round(float(lows[index]), 6) if np.isfinite(lows[index]) else "",
                    "right_offset_m": round(float(highs[index]), 6) if np.isfinite(highs[index]) else "",
                    "cross_section_point_count": int(counts[index]),
                    "is_maximum_width_station": index == max_index,
                    "is_in_max_width_search_region": bool(search_region[index]),
                    "method_version": METHOD_VERSION,
                }
            )

    old_values = np.asarray([row["old_pca_max_body_width_m"] for row in summary_rows], dtype=float)
    new_values = np.asarray([row["extended_geometric_max_width_m"] for row in summary_rows], dtype=float)
    differences = new_values - old_values
    absolute = np.abs(differences)
    percentages = differences / old_values * 100
    correlation = float(np.corrcoef(old_values, new_values)[0, 1])
    diff_sd = float(np.std(differences, ddof=1))
    comparison = {
        "n": len(summary_rows),
        "old_measure": "torso_max_body_width_m (PCA-axis width profile)",
        "new_measure": "extended_geometric_max_width_m (local normals along extended geometric centerline)",
        "method_version": METHOD_VERSION,
        "mean_old_width_m": round(float(np.mean(old_values)), 6),
        "mean_new_width_m": round(float(np.mean(new_values)), 6),
        "mean_signed_difference_m": round(float(np.mean(differences)), 6),
        "median_signed_difference_m": round(float(np.median(differences)), 6),
        "mean_absolute_difference_m": round(float(np.mean(absolute)), 6),
        "median_absolute_difference_m": round(float(np.median(absolute)), 6),
        "rmse_difference_m": round(float(np.sqrt(np.mean(differences**2))), 6),
        "mean_signed_difference_percent": round(float(np.mean(percentages)), 4),
        "mean_absolute_difference_percent": round(float(np.mean(np.abs(percentages))), 4),
        "maximum_absolute_difference_m": round(float(np.max(absolute)), 6),
        "pearson_correlation_old_vs_new": round(correlation, 6),
        "difference_standard_deviation_m": round(diff_sd, 6),
        "agreement_lower_95_m": round(float(np.mean(differences) - 1.96 * diff_sd), 6),
        "agreement_upper_95_m": round(float(np.mean(differences) + 1.96 * diff_sd), 6),
        "count_new_larger": int(np.count_nonzero(differences > 0)),
        "count_new_smaller": int(np.count_nonzero(differences < 0)),
        "count_absolute_difference_over_5cm": int(np.count_nonzero(absolute > 0.05)),
        "count_absolute_percent_over_10": int(np.count_nonzero(np.abs(percentages) > 10)),
        "review_required_cows": [row["cow_id"] for row in summary_rows if row["status"] != "ok"],
    }

    write_csv(SUMMARY_CSV, list(summary_rows[0].keys()), summary_rows)
    write_csv(PROFILE_CSV, list(profile_rows[0].keys()), profile_rows)
    COMPARISON_JSON.write_text(
        json.dumps(
            {
                "comparison": comparison,
                "parameters": {
                    "profile_step_m": PROFILE_STEP_M,
                    "cross_section_slab_thickness_m": 2 * SLAB_HALF_THICKNESS_M,
                    "centerline_smoothing_m": LINE_SMOOTHING_M,
                    "width_profile_smoothing_m": WIDTH_SMOOTHING_M,
                    "edge_quantiles": [EDGE_QUANTILE, 1 - EDGE_QUANTILE],
                    "cross_section_cluster_gap_m": CLUSTER_GAP_M,
                    "rear_max_width_exclusion_m": REAR_MAX_WIDTH_EXCLUSION_M,
                    "head_max_width_exclusion_m": HEAD_MAX_WIDTH_EXCLUSION_M,
                },
                "records": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    update_main_csv({int(row["cow_id"]): row for row in summary_rows})
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
