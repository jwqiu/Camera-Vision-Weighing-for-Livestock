#!/usr/bin/env python3
"""Evaluate the historical PCA 60/65 continuous-width torso rule on 61 cattle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
RECORDS_CSV = OUTPUT_DIR / "pca_60_65_torso_area_records_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "pca_60_65_torso_area_summary_61.json"

BIN_M = 0.01
SMOOTHING_BINS = 3
HEAD_RATIO = 0.60
HEAD_RUN_BINS = 5
REAR_RATIO = 0.65
REAR_RUN_BINS = 3
IMMEDIATE_RATIO = 0.50
METHOD_VERSION = "pca_width_head60_5cm_rear65_3cm_immediate50_trapezoid_v1.0"


def rolling_median_nearest(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window]) for index in range(len(values))],
        dtype=float,
    )


def width_profile(
    points_xy: np.ndarray,
    centroid: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array([-axis[1], axis[0]], dtype=float)
    relative = points_xy - centroid
    longitudinal = relative @ axis
    lateral = relative @ normal
    low = math.floor(float(np.percentile(longitudinal, 0.2)) / BIN_M) * BIN_M
    high = math.ceil(float(np.percentile(longitudinal, 99.8)) / BIN_M) * BIN_M
    edges = np.arange(low, high + BIN_M * 1.1, BIN_M)
    centers = (edges[:-1] + edges[1:]) / 2
    raw = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        cross = lateral[(longitudinal >= left) & (longitudinal < right)]
        counts[index] = len(cross)
        if len(cross) >= 4:
            raw[index] = np.percentile(cross, 97.5) - np.percentile(cross, 2.5)
        elif len(cross) >= 2:
            raw[index] = np.max(cross) - np.min(cross)
    valid = np.isfinite(raw)
    if np.count_nonzero(valid) < 20:
        raise ValueError("too few valid width sections")
    filled = np.interp(np.arange(len(raw)), np.flatnonzero(valid), raw[valid])
    smooth = rolling_median_nearest(filled, SMOOTHING_BINS)
    return longitudinal, centers, smooth, counts


def find_boundary(
    widths: np.ndarray,
    counts: np.ndarray,
    peak_index: int,
    direction: int,
    ratio: float,
    run_bins: int,
    maximum_width: float,
) -> tuple[int, bool, str]:
    valid_indices = np.flatnonzero(counts >= 2)
    endpoint = int(valid_indices[-1] if direction > 0 else valid_indices[0])
    run: list[int] = []
    index = peak_index + direction
    while index <= endpoint if direction > 0 else index >= endpoint:
        value = widths[index]
        if value < IMMEDIATE_RATIO * maximum_width:
            return index, True, "single_below_50pct"
        if value < ratio * maximum_width:
            run.append(index)
            if len(run) >= run_bins:
                return run[0], True, f"continuous_{run_bins}cm_below_{int(ratio * 100)}pct"
        else:
            run.clear()
        index += direction
    return endpoint, False, "threshold_not_met_before_available_body_end"


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end
    return ranks


def metrics(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    pearson = float(np.corrcoef(values, weights)[0, 1])
    spearman = float(np.corrcoef(rank_average(values), rank_average(weights))[0, 1])
    design = np.column_stack([np.ones(len(values)), values])
    coefficients = np.linalg.lstsq(design, weights, rcond=None)[0]
    fitted = design @ coefficients
    residual_sum = float(np.sum((weights - fitted) ** 2))
    total_sum = float(np.sum((weights - np.mean(weights)) ** 2))
    predictions = []
    for held_out in range(len(values)):
        keep = np.arange(len(values)) != held_out
        train_design = np.column_stack([np.ones(np.count_nonzero(keep)), values[keep]])
        train_coefficients = np.linalg.lstsq(train_design, weights[keep], rcond=None)[0]
        predictions.append(float(np.array([1.0, values[held_out]]) @ train_coefficients))
    predictions_array = np.asarray(predictions)
    errors = predictions_array - weights
    return {
        "n": len(values),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "in_sample_r_squared": 1 - residual_sum / total_sum,
        "loocv_rmse_kg": float(np.sqrt(np.mean(errors**2))),
        "loocv_mae_kg": float(np.mean(np.abs(errors))),
        "linear_intercept_kg": float(coefficients[0]),
        "linear_slope_kg_per_m2": float(coefficients[1]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    population = [
        row
        for row in rows
        if row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(population) != 61:
        raise ValueError(f"expected 61 cattle, found {len(population)}")

    records: list[dict[str, object]] = []
    for row in population:
        vertices = read_ply_vertices(BASE / row["extracted_ply"])
        points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
        centroid = np.asarray([float(row["centroid_x_m"]), float(row["centroid_y_m"])])
        axis = np.asarray([float(row["long_axis_unit_x"]), float(row["long_axis_unit_y"])])
        axis /= np.linalg.norm(axis)
        longitudinal, stations, widths, counts = width_profile(points_xy, centroid, axis)
        peak_index = int(np.argmax(widths))
        maximum_width = float(widths[peak_index])
        head_index, head_found, head_reason = find_boundary(
            widths, counts, peak_index, 1, HEAD_RATIO, HEAD_RUN_BINS, maximum_width
        )
        rear_index, rear_found, rear_reason = find_boundary(
            widths, counts, peak_index, -1, REAR_RATIO, REAR_RUN_BINS, maximum_width
        )
        low_index, high_index = sorted((rear_index, head_index))
        projected_area = float(np.trapezoid(widths[low_index : high_index + 1], stations[low_index : high_index + 1]))
        rear_station = float(stations[rear_index])
        head_station = float(stations[head_index])
        records.append(
            {
                "cow_id": int(row["cow_id"]),
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "ground_truth_heart_girth_cm": float(row["ground_truth_heart_girth_cm"]),
                "maximum_smoothed_width_m": round(maximum_width, 6),
                "rear_boundary_axis_m": round(rear_station, 6),
                "head_boundary_axis_m": round(head_station, 6),
                "torso_length_m": round(head_station - rear_station, 6),
                "projected_area_m2": round(projected_area, 6),
                "rear_boundary_found": rear_found,
                "head_boundary_found": head_found,
                "rear_stop_reason": rear_reason,
                "head_stop_reason": head_reason,
                "rule_suitable": rear_found and head_found,
                "method_version": METHOD_VERSION,
            }
        )

    records.sort(key=lambda item: int(item["cow_id"]))
    with RECORDS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    suitable = [record for record in records if record["rule_suitable"]]
    areas = np.asarray([float(record["projected_area_m2"]) for record in suitable])
    weights = np.asarray([float(record["ground_truth_weight_kg"]) for record in suitable])
    girths = np.asarray([float(record["ground_truth_heart_girth_cm"]) for record in suitable])
    summary = {
        "method_version": METHOD_VERSION,
        "population_definition": "61 cattle with strict PCA 45/48 boundaries in long_axis_records.csv",
        "rule": {
            "head": "from maximum width toward head: 5 consecutive 1cm sections below 60%",
            "rear": "from maximum width toward rear: 3 consecutive 1cm sections below 65%",
            "either_side": "stop immediately at one section below 50%",
            "area": "trapezoidal integral of smoothed top-view width profile between boundaries",
        },
        "evaluated_count": len(records),
        "suitable_count": len(suitable),
        "unsuitable_cow_ids": [int(record["cow_id"]) for record in records if not record["rule_suitable"]],
        "projected_area_vs_weight": metrics(areas, weights),
        "heart_girth_vs_weight_same_suitable_population": metrics(girths, weights),
        "projected_area_distribution_m2": {
            "mean": float(np.mean(areas)),
            "minimum": float(np.min(areas)),
            "maximum": float(np.max(areas)),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
