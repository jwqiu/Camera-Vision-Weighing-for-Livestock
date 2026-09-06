#!/usr/bin/env python3
"""Recompute and record maximum widths along the extended skeleton centerline for 61 cattle."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
AXIS_DIR = BASE / "skeleton_axis_comparison"
MAIN_CSV = BASE / "long_axis_records.csv"
CENTERLINE_JSON = AXIS_DIR / "extended_geometric_centerline_records.json"
OUTPUT_DIR = BASE / "skeleton_max_width_61"
OUTPUT_CSV = OUTPUT_DIR / "skeleton_max_width_records_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "skeleton_max_width_summary_61.json"

sys.path.insert(0, str(ROOT / "tools"))
from compute_extended_centerline_widths import (  # noqa: E402
    METHOD_VERSION,
    HEAD_MAX_WIDTH_EXCLUSION_M,
    REAR_MAX_WIDTH_EXCLUSION_M,
    compute_profile,
    read_ply_vertices,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_csv(MAIN_CSV)
    strict_rows = [
        row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible", "").upper() == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(strict_rows) != 61:
        raise ValueError(f"expected 61 strict cattle, found {len(strict_rows)}")

    payload = json.loads(CENTERLINE_JSON.read_text(encoding="utf-8"))
    centerlines = {int(record["cow_id"]): record for record in payload["records"]}
    records: list[dict[str, object]] = []
    for row in strict_rows:
        cow_id = int(row["cow_id"])
        centerline_record = centerlines[cow_id]
        centerline_xy = np.asarray(
            [
                [point["camera_x_m"], point["camera_y_m"]]
                for point in centerline_record["points"]
            ]
        )
        cattle = read_ply_vertices(BASE / row["extracted_ply"])
        cattle_xy = np.column_stack([cattle["x"], cattle["y"]]).astype(float)
        stations, centers, tangents, raw, smooth, lows, highs, counts = compute_profile(
            centerline_xy, cattle_xy
        )
        valid = np.isfinite(smooth)
        search_start = REAR_MAX_WIDTH_EXCLUSION_M
        search_end = float(stations[-1] - HEAD_MAX_WIDTH_EXCLUSION_M)
        search = valid & (stations >= search_start) & (stations <= search_end)
        if not np.any(search):
            raise ValueError(f"cow {cow_id}: no valid skeleton maximum-width search region")
        maximum_index = int(np.nanargmax(np.where(search, smooth, np.nan)))
        width = float(smooth[maximum_index])
        pca_width = float(row["pca_full_torso_max_width_m"])
        tangent = tangents[maximum_index]
        normal = np.asarray([-tangent[1], tangent[0]])
        center = centers[maximum_index]
        left = center + float(lows[maximum_index]) * normal
        right = center + float(highs[maximum_index]) * normal
        difference = width - pca_width
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "skeleton_max_width_m": round(width, 6),
                "skeleton_max_width_station_from_rear_m": round(float(stations[maximum_index]), 6),
                "skeleton_max_width_center_x_m": round(float(center[0]), 6),
                "skeleton_max_width_center_y_m": round(float(center[1]), 6),
                "skeleton_local_tangent_x": round(float(tangent[0]), 8),
                "skeleton_local_tangent_y": round(float(tangent[1]), 8),
                "skeleton_left_boundary_x_m": round(float(left[0]), 6),
                "skeleton_left_boundary_y_m": round(float(left[1]), 6),
                "skeleton_right_boundary_x_m": round(float(right[0]), 6),
                "skeleton_right_boundary_y_m": round(float(right[1]), 6),
                "raw_width_at_selected_station_m": round(float(raw[maximum_index]), 6),
                "cross_section_point_count": int(counts[maximum_index]),
                "pca_45_48_max_width_m": round(pca_width, 6),
                "skeleton_minus_pca_m": round(difference, 6),
                "absolute_difference_m": round(abs(difference), 6),
                "difference_vs_pca_percent": round(difference / pca_width * 100, 4),
                "valid_profile_fraction": round(float(np.mean(valid)), 6),
                "status": "ok" if np.mean(valid) >= 0.8 else "review_required",
                "method_version": METHOD_VERSION,
                "centerline_source": "extended_geometric_centerline_records.json",
            }
        )

    skeleton = np.asarray([float(row["skeleton_max_width_m"]) for row in records])
    pca = np.asarray([float(row["pca_45_48_max_width_m"]) for row in records])
    weight = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    difference = skeleton - pca
    summary = {
        "n": len(records),
        "population": "61 cattle with strict PCA rear45/head48 inclusion; PCA is used only for sample inclusion and comparison",
        "measurement_axis": "rear-extended curved skeleton centerline",
        "width_definition": "local cross-section perpendicular to skeleton tangent; 1cm slab; 1st-to-99th percentile contiguous width; 3cm rolling-median width profile",
        "maximum_search_region": {
            "rear_exclusion_m": REAR_MAX_WIDTH_EXCLUSION_M,
            "head_exclusion_m": HEAD_MAX_WIDTH_EXCLUSION_M,
        },
        "method_version": METHOD_VERSION,
        "mean_skeleton_max_width_m": float(np.mean(skeleton)),
        "mean_pca_max_width_m": float(np.mean(pca)),
        "mean_skeleton_minus_pca_m": float(np.mean(difference)),
        "mean_absolute_difference_m": float(np.mean(np.abs(difference))),
        "maximum_absolute_difference_m": float(np.max(np.abs(difference))),
        "pearson_skeleton_vs_pca": float(np.corrcoef(skeleton, pca)[0, 1]),
        "pearson_skeleton_width_vs_weight": float(np.corrcoef(skeleton, weight)[0, 1]),
        "pearson_pca_width_vs_weight": float(np.corrcoef(pca, weight)[0, 1]),
        "review_required_cows": [row["cow_id"] for row in records if row["status"] != "ok"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_CSV, records)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
