#!/usr/bin/env python3
"""Compute 1 cm elliptical core-torso volumes for two Rightview boundary methods."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from compute_rightview_median_body_depth import compute_body_depth
from preview_pca_full_torso_boundaries import pca_width_profile, read_ply_vertices


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
RECORDS = BASE / "long_axis_records.csv"
BOUNDARY_BASE = BASE / "top_right_transform_experiment_15x5"
OUTPUT = BASE / "elliptical_slice_volumes_two_right_boundaries"
GRID_M = 0.01
VOLUME_METHOD = "top_width_right_depth_elliptical_slices_1cm_v1.0"

BOUNDARY_METHODS = {
    "matrix": BOUNDARY_BASE / "rightview_core_boundaries_matrix_method_61.csv",
    "proportional": BOUNDARY_BASE / "rightview_core_boundaries_proportional_method_61.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strict_record_map() -> dict[int, dict[str, str]]:
    rows = read_csv(RECORDS)
    return {
        int(row["cow_id"]): row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible") == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    }


def top_width_slices(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.array([float(row["centroid_x_m"]), float(row["centroid_y_m"])])
    axis = np.array([float(row["long_axis_unit_x"]), float(row["long_axis_unit_y"])])
    axis /= np.linalg.norm(axis)
    _, _, stations, widths = pca_width_profile(xy, centroid, axis)
    rear = float(row["pca_full_torso_rear_boundary_axis_m"])
    head = float(row["pca_full_torso_head_boundary_axis_m"])
    length = head - rear
    count = max(1, int(round(length / GRID_M)))
    thickness = length / count
    centers = rear + (np.arange(count) + 0.5) * thickness
    slice_widths = np.interp(centers, stations, widths)
    return centers, slice_widths, thickness


def right_depth_profile(row: dict[str, str]) -> dict[str, object]:
    result = compute_body_depth(ROOT / row["rightview_source_ply"])
    centers = np.asarray(result["_centers"], dtype=float)
    back = np.asarray(result["_back"], dtype=float)
    belly = np.asarray(result["_belly"], dtype=float)
    counts = np.asarray(result["_counts"], dtype=int)
    depths = back - belly
    valid = (
        np.isfinite(centers)
        & np.isfinite(depths)
        & (counts >= 12)
        & (depths > 0.25)
        & (depths < 1.25)
    )
    if np.count_nonzero(valid) < 20:
        raise ValueError(f"cow {row['cow_id']}: insufficient full Rightview depth profile")

    # Boundary records are in raw Right-camera x.  The existing depth curve uses
    # the mildly rotated u coordinate.  Fit their central relationship using all
    # segmented cattle points, which is stable because the correction angle is small.
    raw_x = np.asarray(result["_x"], dtype=float)
    rotated_u = np.asarray(result["_u"], dtype=float)
    u_from_x = np.polyfit(raw_x, rotated_u, 1)
    return {
        "centers": centers[valid],
        "depths": depths[valid],
        "u_from_x": u_from_x,
        "valid_profile_low_u_m": float(np.min(centers[valid])),
        "valid_profile_high_u_m": float(np.max(centers[valid])),
        "rightview_horizontal_rotation_deg": result["rightview_horizontal_rotation_deg"],
    }


def compute_one(
    row: dict[str, str],
    boundary: dict[str, str],
    boundary_method: str,
    top_centers: np.ndarray,
    widths: np.ndarray,
    thickness: float,
    right: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rear_x = float(boundary["rightview_rear_boundary_x_m"])
    head_x = float(boundary["rightview_head_boundary_x_m"])
    u_from_x = np.asarray(right["u_from_x"])
    rear_u = float(np.polyval(u_from_x, rear_x))
    head_u = float(np.polyval(u_from_x, head_x))
    relative = (top_centers - top_centers.min() + thickness / 2) / (len(top_centers) * thickness)
    target_u = rear_u + relative * (head_u - rear_u)

    profile_u = np.asarray(right["centers"])
    profile_depth = np.asarray(right["depths"])
    order = np.argsort(profile_u)
    profile_u = profile_u[order]
    profile_depth = profile_depth[order]
    outside = (target_u < profile_u[0]) | (target_u > profile_u[-1])
    depths = np.interp(target_u, profile_u, profile_depth)
    areas = np.pi * widths * depths / 4.0
    slice_volumes = areas * thickness
    volume = float(np.sum(slice_volumes))
    extrapolated_fraction = float(np.mean(outside))
    boundary_outside = bool(
        rear_u < profile_u[0] - 0.02
        or rear_u > profile_u[-1] + 0.02
        or head_u < profile_u[0] - 0.02
        or head_u > profile_u[-1] + 0.02
    )
    status = "review_required" if extrapolated_fraction > 0.05 or boundary_outside else "ok"

    summary = {
        "cow_id": int(row["cow_id"]),
        "boundary_method": boundary_method,
        "boundary_method_version": boundary["method_version"],
        "volume_method_version": VOLUME_METHOD,
        "top_source_ply": row["source_top_ply"],
        "right_source_ply": row["rightview_source_ply"],
        "top_core_length_m": float(row["pca_full_torso_core_length_m"]),
        "right_rear_boundary_x_m": rear_x,
        "right_head_boundary_x_m": head_x,
        "right_core_length_m": float(boundary["rightview_core_length_m"]),
        "right_rear_boundary_profile_u_m": rear_u,
        "right_head_boundary_profile_u_m": head_u,
        "slice_count": len(widths),
        "slice_thickness_m": thickness,
        "mean_top_width_m": float(np.mean(widths)),
        "mean_corrected_right_depth_m": float(np.mean(depths)),
        "min_corrected_right_depth_m": float(np.min(depths)),
        "max_corrected_right_depth_m": float(np.max(depths)),
        "elliptical_core_torso_volume_m3": volume,
        "elliptical_core_torso_volume_l": volume * 1000.0,
        "right_profile_extrapolated_slice_fraction": extrapolated_fraction,
        "right_boundary_outside_valid_profile": boundary_outside,
        "rightview_horizontal_rotation_deg": right["rightview_horizontal_rotation_deg"],
        "leg_udder_handling": "41cm rolling 90pct corrected belly contour",
        "cross_section_assumption": "ellipse_area=pi*top_width*corrected_right_depth/4",
        "status": status,
    }
    slices = [
        {
            "cow_id": int(row["cow_id"]),
            "boundary_method": boundary_method,
            "slice_index": index,
            "relative_core_position": float(relative[index]),
            "top_axis_position_m": float(top_centers[index]),
            "top_width_m": float(widths[index]),
            "right_profile_position_u_m": float(target_u[index]),
            "corrected_right_depth_m": float(depths[index]),
            "ellipse_area_m2": float(areas[index]),
            "slice_thickness_m": thickness,
            "slice_volume_m3": float(slice_volumes[index]),
            "right_depth_extrapolated": bool(outside[index]),
        }
        for index in range(len(widths))
    ]
    return summary, slices


def main() -> None:
    records = strict_record_map()
    boundaries = {
        name: {int(row["cow_id"]): row for row in read_csv(path)}
        for name, path in BOUNDARY_METHODS.items()
    }
    expected = set(records)
    for name, values in boundaries.items():
        if set(values) != expected:
            raise ValueError(f"{name}: boundary cattle IDs do not match strict 61-cow set")

    summaries: dict[str, list[dict[str, object]]] = {name: [] for name in boundaries}
    all_slices: list[dict[str, object]] = []
    for position, cow_id in enumerate(sorted(records), 1):
        row = records[cow_id]
        top_centers, widths, thickness = top_width_slices(row)
        right = right_depth_profile(row)
        for name in boundaries:
            summary, slices = compute_one(
                row, boundaries[name][cow_id], name, top_centers, widths, thickness, right
            )
            summaries[name].append(summary)
            all_slices.extend(slices)
        print(f"[{position:02d}/61] cow {cow_id:03d}", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in summaries.items():
        write_csv(OUTPUT / f"elliptical_core_torso_volumes_{name}_method_61.csv", rows)
    write_csv(OUTPUT / "elliptical_core_torso_slice_details_two_methods_61.csv", all_slices)

    matrix_map = {int(row["cow_id"]): row for row in summaries["matrix"]}
    proportional_map = {int(row["cow_id"]): row for row in summaries["proportional"]}
    comparison = []
    for cow_id in sorted(records):
        matrix_volume = float(matrix_map[cow_id]["elliptical_core_torso_volume_m3"])
        proportional_volume = float(proportional_map[cow_id]["elliptical_core_torso_volume_m3"])
        difference = matrix_volume - proportional_volume
        comparison.append({
            "cow_id": cow_id,
            "matrix_boundary_volume_m3": matrix_volume,
            "proportional_boundary_volume_m3": proportional_volume,
            "matrix_minus_proportional_m3": difference,
            "absolute_difference_m3": abs(difference),
            "difference_vs_proportional_percent": difference / proportional_volume * 100.0,
            "matrix_status": matrix_map[cow_id]["status"],
            "proportional_status": proportional_map[cow_id]["status"],
            "volume_method_version": VOLUME_METHOD,
        })
    write_csv(OUTPUT / "elliptical_core_torso_volumes_two_methods_comparison_61.csv", comparison)

    summary_json = {
        "method": VOLUME_METHOD,
        "cattle_count_per_method": 61,
        "slice_grid_m": GRID_M,
        "cross_section": "ellipse",
        "top_measurement": "PCA-axis smoothed width profile within accepted rear45/head48 core",
        "right_measurement": "corrected back-to-belly depth profile; legs and udder suppressed",
        "methods": {},
    }
    for name, rows in summaries.items():
        volumes = np.array([float(row["elliptical_core_torso_volume_m3"]) for row in rows])
        summary_json["methods"][name] = {
            "boundary_source": str(BOUNDARY_METHODS[name].relative_to(ROOT)),
            "volume_min_m3": float(np.min(volumes)),
            "volume_median_m3": float(np.median(volumes)),
            "volume_mean_m3": float(np.mean(volumes)),
            "volume_max_m3": float(np.max(volumes)),
            "ok_count": sum(row["status"] == "ok" for row in rows),
            "review_required_count": sum(row["status"] != "ok" for row in rows),
        }
    differences = np.array([float(row["matrix_minus_proportional_m3"]) for row in comparison])
    summary_json["comparison"] = {
        "mean_matrix_minus_proportional_m3": float(np.mean(differences)),
        "median_matrix_minus_proportional_m3": float(np.median(differences)),
        "mean_absolute_difference_m3": float(np.mean(np.abs(differences))),
        "max_absolute_difference_m3": float(np.max(np.abs(differences))),
    }
    summary_json["interpretation"] = (
        "Experimental core-torso volumes, not whole-animal anatomical volumes and not ground-truth validated."
    )
    with (OUTPUT / "elliptical_core_torso_volume_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
