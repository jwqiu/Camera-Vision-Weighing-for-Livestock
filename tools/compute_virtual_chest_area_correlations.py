#!/usr/bin/env python3
"""Compute virtual chest-width/depth area proxies and weight correlations."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
DATASET = ROOT / "dataset"
MAIN_CSV = BASE / "long_axis_records.csv"
VALLEY_CSV = (
    BASE
    / "pca_horizontal_60pct_upper_halfwidth_preview_61"
    / "pca_upper_halfwidth_records_61.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
RECORDS_CSV = OUTPUT_DIR / "virtual_chest_area_records_61.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "virtual_chest_area_correlations_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "virtual_chest_area_summary.json"

DEPTH_OFFSETS_M = np.arange(-0.05, 0.0501, 0.01)
METHOD_VERSION = "dual_valley_virtual_width_rightview_chest_depth_v1.0"
RECORDED_ON = "2026-09-06"

sys.path.insert(0, str(ROOT / "tools"))
from compute_three_view_simulated_girth import (  # noqa: E402
    prepare_side,
    vertical_surface_profile,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def loocv(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        predictions[index] = slope * x[index] + intercept
    residual = predictions - y
    return (
        predictions,
        float(np.corrcoef(predictions, y)[0, 1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(np.abs(residual))),
    )


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    clipped = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(clipped)
    se = 1 / math.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def metric_summary(
    field: str, label: str, unit: str, values: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    pearson = float(np.corrcoef(values, weights)[0, 1])
    spearman = float(
        np.corrcoef(average_ranks(values), average_ranks(weights))[0, 1]
    )
    ci_low, ci_high = fisher_ci(pearson, len(values))
    slope, intercept = np.polyfit(values, weights, 1)
    fitted = slope * values + intercept
    residual = fitted - weights
    _, loo_r, loo_rmse, loo_mae = loocv(values, weights)
    return {
        "metric_field": field,
        "metric_label_zh": label,
        "unit": unit,
        "n": len(values),
        "pearson_r": round(pearson, 6),
        "pearson_ci95_low": round(ci_low, 6),
        "pearson_ci95_high": round(ci_high, 6),
        "spearman_rho": round(spearman, 6),
        "r_squared": round(pearson**2, 6),
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "in_sample_rmse_kg": round(float(np.sqrt(np.mean(residual**2))), 6),
        "in_sample_mae_kg": round(float(np.mean(np.abs(residual))), 6),
        "loocv_r": round(loo_r, 6),
        "loocv_rmse_kg": round(loo_rmse, 6),
        "loocv_mae_kg": round(loo_mae, 6),
        "method_version": METHOD_VERSION,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    valleys = sorted(read_csv(VALLEY_CSV), key=lambda row: int(row["cow_id"]))
    if len(valleys) != 61:
        raise ValueError(f"expected 61 cattle, found {len(valleys)}")

    records: list[dict[str, object]] = []
    for position, valley in enumerate(valleys, start=1):
        cow_id = int(valley["cow_id"])
        row = main_rows[cow_id]
        upper_ratio = float(valley["pca_upper_width_valley_station_ratio_full_axis"])
        lower_ratio = float(valley["pca_lower_width_valley_station_ratio_full_axis"])
        chest_ratio = (upper_ratio + lower_ratio) / 2
        upper_halfwidth = float(valley["pca_upper_width_valley_m"])
        lower_halfwidth = float(valley["pca_lower_width_valley_m"])
        virtual_width = upper_halfwidth + lower_halfwidth

        right_paths = list((DATASET / str(cow_id)).glob("right-*.ply"))
        if len(right_paths) != 1:
            raise ValueError(f"cow {cow_id}: missing or duplicate right-view PLY")
        side = prepare_side(right_paths[0], cow_id, "right")
        rear_u = float(side["rear_u"])
        head_u = float(side["head_u"])
        center_u = rear_u + chest_ratio * (head_u - rear_u)
        depths: list[float] = []
        coverages: list[float] = []
        for offset in DEPTH_OFFSETS_M:
            profile = vertical_surface_profile(side, center_u + float(offset))
            if profile is None:
                continue
            heights = profile[0]
            depths.append(float(heights[-1] - heights[0]))
            coverages.append(float(profile[4]))
        if len(depths) < 7:
            raise ValueError(f"cow {cow_id}: only {len(depths)} valid chest-depth profiles")
        chest_depth = float(np.median(depths))
        torso_length = float(row["pca_full_torso_core_length_m"])
        ellipse_area = math.pi * virtual_width * chest_depth / 4
        area_length = ellipse_area * torso_length
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "ground_truth_heart_girth_cm": float(row["ground_truth_heart_girth_cm"]),
                "upper_valley_ratio_from_rear": upper_ratio,
                "lower_valley_ratio_from_rear": lower_ratio,
                "valley_longitudinal_offset_m": abs(
                    float(valley["pca_upper_width_valley_station_from_rear_m"])
                    - float(valley["pca_lower_width_valley_station_from_rear_m"])
                ),
                "virtual_chest_ratio_from_rear": chest_ratio,
                "upper_valley_halfwidth_m": upper_halfwidth,
                "lower_valley_halfwidth_m": lower_halfwidth,
                "virtual_chest_width_m": virtual_width,
                "rightview_chest_depth_m": chest_depth,
                "ellipse_chest_area_m2": ellipse_area,
                "pca_torso_length_m": torso_length,
                "ellipse_area_times_torso_length_m3": area_length,
                "rightview_depth_valid_profile_count": len(depths),
                "rightview_depth_profile_coverage_median": float(np.median(coverages)),
                "rightview_ground_fit_inlier_fraction": float(side["ground_fraction"]),
                "rightview_source_ply": str(right_paths[0].relative_to(ROOT)),
                "virtual_width_formula": "upper valley halfwidth + lower valley halfwidth",
                "ellipse_area_formula": "pi * virtual chest width * rightview chest depth / 4",
                "depth_window_definition": "median of 1cm-spaced profiles within +/-5cm of mean dual-valley station",
                "method_version": METHOD_VERSION,
                "status": "ok_experimental_virtual_chest_proxy",
            }
        )
        print(
            f"[{position:02d}/61] cow {cow_id:03d}: "
            f"W={virtual_width:.3f}m D={chest_depth:.3f}m A={ellipse_area:.3f}m2",
            flush=True,
        )

    write_csv(RECORDS_CSV, records)
    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    metric_specs = (
        ("virtual_chest_width_m", "虚拟胸宽", "m"),
        ("rightview_chest_depth_m", "Rightview胸区体深", "m"),
        ("ellipse_chest_area_m2", "椭圆胸截面积", "m²"),
        ("ellipse_area_times_torso_length_m3", "椭圆胸截面积×躯干长度", "m³"),
        ("ground_truth_heart_girth_cm", "人工胸围", "cm"),
    )
    metrics: list[dict[str, object]] = []
    for field, label, unit in metric_specs:
        values = np.asarray([float(row[field]) for row in records])
        metrics.append(metric_summary(field, label, unit, values, weights))
    write_csv(CORRELATIONS_CSV, metrics)

    best_visual = max(metrics[:-1], key=lambda item: float(item["loocv_r"]))
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "metric_definitions": {
                    "virtual_chest_width_m": "sum of independently detected upper/lower Topview valley half-widths",
                    "rightview_chest_depth_m": "median corrected back-to-belly depth within +/-5cm of mean dual-valley station",
                    "ellipse_chest_area_m2": "pi * width * depth / 4",
                    "ellipse_area_times_torso_length_m3": "ellipse chest area * PCA torso core length",
                },
                "correlations": metrics,
                "best_visual_metric_by_loocv_r": best_visual,
                "records_file": str(RECORDS_CSV.relative_to(ROOT)),
                "correlations_file": str(CORRELATIONS_CSV.relative_to(ROOT)),
                "method_version": METHOD_VERSION,
                "recorded_on": RECORDED_ON,
                "limitations": [
                    "virtual width combines half-widths measured at two longitudinally offset valley stations",
                    "right-view depth is aligned by normalized rear-to-head position because no cross-camera extrinsic calibration is available",
                    "ellipse area assumes a constant cross-sectional shape factor across cattle",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(SUMMARY_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
