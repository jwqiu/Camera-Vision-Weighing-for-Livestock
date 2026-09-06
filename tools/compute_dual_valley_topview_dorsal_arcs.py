#!/usr/bin/env python3
"""Compute two dual-valley Topview dorsal-arc proxies for 61 cattle."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
VALLEY_CSV = (
    BASE
    / "pca_horizontal_60pct_upper_halfwidth_preview_61"
    / "pca_upper_halfwidth_records_61.csv"
)
RIDGE_POINTS_CSV = (
    BASE
    / "topview_back_high_band_centerline_61"
    / "back_high_band_centerline_points_61.csv"
)
OUTPUT_DIR = BASE / "dual_valley_topview_dorsal_arcs_61"
RECORDS_CSV = OUTPUT_DIR / "dual_valley_dorsal_arc_records_61.csv"
METRICS_CSV = OUTPUT_DIR / "dual_valley_dorsal_arc_correlations_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "dual_valley_dorsal_arc_summary.json"

PROFILE_SLAB_HALF_WIDTH_M = 0.01
LATERAL_BIN_M = 0.01
LATERAL_SMOOTHING_M = 0.05
METHOD1_OFFSETS_M = np.arange(-0.02, 0.0201, 0.01)
METHOD2_OFFSETS_M = np.arange(-0.05, 0.0501, 0.01)
METHOD_VERSION = "dual_valley_topview_dorsal_arc_v1.0"
RECORDED_ON = "2026-09-06"

sys.path.insert(0, str(ROOT / "tools"))
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def surface_profile(
    lateral: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(lateral) < 80:
        return None
    low, high = np.quantile(lateral, [0.02, 0.98])
    edges = np.arange(
        math.floor(low / LATERAL_BIN_M) * LATERAL_BIN_M,
        math.ceil(high / LATERAL_BIN_M) * LATERAL_BIN_M + LATERAL_BIN_M * 0.5,
        LATERAL_BIN_M,
    )
    if len(edges) < 13:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.digitize(lateral, edges) - 1
    values = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        cell = height[indices == index]
        if len(cell) >= 2:
            values[index] = np.median(cell)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < max(12, 0.65 * len(values)):
        return None
    first, last = np.flatnonzero(valid)[[0, -1]]
    centers = centers[first : last + 1]
    values = values[first : last + 1]
    valid = np.isfinite(values)
    values[~valid] = np.interp(
        np.flatnonzero(~valid), np.flatnonzero(valid), values[valid]
    )
    width = max(3, int(round(LATERAL_SMOOTHING_M / LATERAL_BIN_M)))
    if width % 2 == 0:
        width += 1
    return centers, smooth(values, width)


def curve_arc(q: np.ndarray, h: np.ndarray, start: float, end: float) -> float:
    lo = float(np.clip(min(start, end), q[0], q[-1]))
    hi = float(np.clip(max(start, end), q[0], q[-1]))
    if hi - lo < 0.03:
        return float("nan")
    inner = q[(q > lo) & (q < hi)]
    samples_q = np.r_[lo, inner, hi]
    samples_h = np.interp(samples_q, q, h)
    return float(np.sum(np.sqrt(np.diff(samples_q) ** 2 + np.diff(samples_h) ** 2)))


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


def loocv(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    prediction = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        prediction[index] = slope * x[index] + intercept
    residual = prediction - y
    return (
        float(np.corrcoef(prediction, y)[0, 1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(np.abs(residual))),
    )


def metric_summary(
    field: str, label: str, values: np.ndarray, outcome: np.ndarray, outcome_name: str
) -> dict[str, object]:
    pearson = float(np.corrcoef(values, outcome)[0, 1])
    spearman = float(
        np.corrcoef(average_ranks(values), average_ranks(outcome))[0, 1]
    )
    loo_r, loo_rmse, loo_mae = loocv(values, outcome)
    return {
        "metric_field": field,
        "metric_label_zh": label,
        "outcome_field": outcome_name,
        "n": len(values),
        "pearson_r": round(pearson, 6),
        "spearman_rho": round(spearman, 6),
        "r_squared": round(pearson**2, 6),
        "loocv_r": round(loo_r, 6),
        "loocv_rmse": round(loo_rmse, 6),
        "loocv_mae": round(loo_mae, 6),
        "method_version": METHOD_VERSION,
    }


def ridge_by_cow() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in read_csv(RIDGE_POINTS_CSV):
        grouped.setdefault(int(row["cow_id"]), []).append(
            (
                float(row["station_axis_from_centroid_m"]),
                float(row["smoothed_high_band_center_lateral_m"]),
            )
        )
    result = {}
    for cow_id, values in grouped.items():
        values.sort()
        result[cow_id] = (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )
    return result


def median_valid(values: list[float], minimum: int) -> float:
    valid = np.asarray([value for value in values if np.isfinite(value)])
    if len(valid) < minimum:
        return float("nan")
    return float(np.median(valid))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_by_cow = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    valleys = sorted(read_csv(VALLEY_CSV), key=lambda row: int(row["cow_id"]))
    ridges = ridge_by_cow()
    if len(valleys) != 61:
        raise ValueError(f"expected 61 valley records, found {len(valleys)}")

    records: list[dict[str, object]] = []
    for valley in valleys:
        cow_id = int(valley["cow_id"])
        row = main_by_cow[cow_id]
        vertices = read_ply_vertices(BASE / row["extracted_ply"])
        xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
        centroid = np.asarray(
            [
                float(row["pca_full_torso_axis_centroid_x_m"]),
                float(row["pca_full_torso_axis_centroid_y_m"]),
            ]
        )
        axis = np.asarray(
            [
                float(row["pca_full_torso_axis_unit_x"]),
                float(row["pca_full_torso_axis_unit_y"]),
            ]
        )
        axis /= np.linalg.norm(axis)
        normal = np.asarray([-axis[1], axis[0]])
        relative = xy - centroid
        longitudinal = relative @ axis
        lateral = relative @ normal
        ground_z = (
            float(row["ground_plane_a"]) * vertices["x"]
            + float(row["ground_plane_b"]) * vertices["y"]
            + float(row["ground_plane_c_m"])
        )
        height = ground_z - vertices["z"]

        upper_xy = np.asarray(
            [
                float(valley["pca_upper_width_valley_contour_x_m"]),
                float(valley["pca_upper_width_valley_contour_y_m"]),
            ]
        )
        lower_xy = np.asarray(
            [
                float(valley["pca_lower_width_valley_contour_x_m"]),
                float(valley["pca_lower_width_valley_contour_y_m"]),
            ]
        )
        upper_relative = upper_xy - centroid
        lower_relative = lower_xy - centroid
        upper_station = float(upper_relative @ axis)
        lower_station = float(lower_relative @ axis)
        upper_lateral = float(upper_relative @ normal)
        lower_lateral = float(lower_relative @ normal)
        midpoint_station = (upper_station + lower_station) / 2
        ridge_t, ridge_q = ridges[cow_id]

        def reference_q(station: float, reference: str) -> float:
            if reference == "ridge":
                return float(np.interp(station, ridge_t, ridge_q))
            return 0.0

        def profile_at(station: float):
            selected = np.abs(longitudinal - station) < PROFILE_SLAB_HALF_WIDTH_M
            return surface_profile(lateral[selected], height[selected])

        def half_arcs(
            center_station: float, side_lateral: float, reference: str
        ) -> tuple[list[float], list[float]]:
            arcs: list[float] = []
            deficits: list[float] = []
            for offset in METHOD1_OFFSETS_M:
                station = center_station + float(offset)
                profile = profile_at(station)
                if profile is None:
                    continue
                q, h = profile
                ref = reference_q(station, reference)
                endpoint = float(q[-1] if side_lateral > 0 else q[0])
                arcs.append(curve_arc(q, h, ref, endpoint))
                deficits.append(float(np.max(h) - np.interp(ref, q, h)))
            return arcs, deficits

        upper_ridge, upper_ridge_deficit = half_arcs(
            upper_station, upper_lateral, "ridge"
        )
        lower_ridge, lower_ridge_deficit = half_arcs(
            lower_station, lower_lateral, "ridge"
        )
        upper_pca, upper_pca_deficit = half_arcs(
            upper_station, upper_lateral, "pca"
        )
        lower_pca, lower_pca_deficit = half_arcs(
            lower_station, lower_lateral, "pca"
        )
        upper_ridge_median = median_valid(upper_ridge, 3)
        lower_ridge_median = median_valid(lower_ridge, 3)
        upper_pca_median = median_valid(upper_pca, 3)
        lower_pca_median = median_valid(lower_pca, 3)
        method1_ridge = upper_ridge_median + lower_ridge_median
        method1_pca = upper_pca_median + lower_pca_median

        full_arcs: list[float] = []
        full_split_ridge: list[float] = []
        full_split_pca: list[float] = []
        for offset in METHOD2_OFFSETS_M:
            station = midpoint_station + float(offset)
            profile = profile_at(station)
            if profile is None:
                continue
            q, h = profile
            full_arcs.append(curve_arc(q, h, float(q[0]), float(q[-1])))
            ridge_reference = reference_q(station, "ridge")
            full_split_ridge.append(
                curve_arc(q, h, float(q[0]), ridge_reference)
                + curve_arc(q, h, ridge_reference, float(q[-1]))
            )
            full_split_pca.append(
                curve_arc(q, h, float(q[0]), 0.0)
                + curve_arc(q, h, 0.0, float(q[-1]))
            )
        method2 = median_valid(full_arcs, 7)
        method2_ridge_split = median_valid(full_split_ridge, 7)
        method2_pca_split = median_valid(full_split_pca, 7)

        if not all(
            np.isfinite(value)
            for value in (method1_ridge, method1_pca, method2)
        ):
            raise ValueError(f"cow {cow_id}: insufficient valid profiles")
        difference = method1_ridge - method2
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "ground_truth_heart_girth_cm": float(row["ground_truth_heart_girth_cm"]),
                "upper_valley_ratio_from_rear": float(
                    valley["pca_upper_width_valley_station_ratio_full_axis"]
                ),
                "lower_valley_ratio_from_rear": float(
                    valley["pca_lower_width_valley_station_ratio_full_axis"]
                ),
                "upper_lower_valley_distance_m": abs(upper_station - lower_station),
                "midpoint_station_axis_from_centroid_m": midpoint_station,
                "method1_upper_half_arc_ridge_m": upper_ridge_median,
                "method1_lower_half_arc_ridge_m": lower_ridge_median,
                "method1_separate_valley_half_arcs_ridge_m": method1_ridge,
                "method1_separate_valley_half_arcs_pca_m": method1_pca,
                "method1_ridge_minus_pca_m": method1_ridge - method1_pca,
                "method2_midpoint_full_arc_m": method2,
                "method2_midpoint_full_arc_split_at_ridge_m": method2_ridge_split,
                "method2_midpoint_full_arc_split_at_pca_m": method2_pca_split,
                "method1_minus_method2_m": difference,
                "method1_vs_method2_absolute_difference_m": abs(difference),
                "method1_vs_method2_absolute_difference_percent": abs(difference)
                / ((method1_ridge + method2) / 2)
                * 100,
                "ridge_reference_median_height_deficit_m": float(
                    np.median(upper_ridge_deficit + lower_ridge_deficit)
                ),
                "pca_reference_median_height_deficit_m": float(
                    np.median(upper_pca_deficit + lower_pca_deficit)
                ),
                "method1_valid_upper_profile_count": len(upper_ridge),
                "method1_valid_lower_profile_count": len(lower_ridge),
                "method2_valid_profile_count": len(full_arcs),
                "method1_window_definition": "five 1cm-spaced profiles centered on each side valley; median per side then sum",
                "method2_window_definition": "eleven 1cm-spaced full profiles centered on mean valley station; median full arc",
                "surface_scope": "Topview-visible dorsal surface only",
                "method_version": METHOD_VERSION,
                "status": "ok",
            }
        )

    write_csv(RECORDS_CSV, records)
    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    girths = np.asarray([float(row["ground_truth_heart_girth_cm"]) for row in records])
    metric_fields = (
        (
            "method1_separate_valley_half_arcs_ridge_m",
            "方法1：两侧凹点半弧相加（高位带中心线）",
        ),
        (
            "method1_separate_valley_half_arcs_pca_m",
            "方法1：两侧凹点半弧相加（PCA线）",
        ),
        ("method2_midpoint_full_arc_m", "方法2：两凹点中间位置完整背弧"),
    )
    metrics: list[dict[str, object]] = []
    for field, label in metric_fields:
        values = np.asarray([float(row[field]) for row in records])
        metrics.append(metric_summary(field, label, values, girths, "ground_truth_heart_girth_cm"))
        metrics.append(metric_summary(field, label, values, weights, "ground_truth_weight_kg"))
    write_csv(METRICS_CSV, metrics)

    differences = np.asarray(
        [float(row["method1_vs_method2_absolute_difference_m"]) for row in records]
    )
    difference_percent = np.asarray(
        [float(row["method1_vs_method2_absolute_difference_percent"]) for row in records]
    )
    ridge_deficit = np.asarray(
        [float(row["ridge_reference_median_height_deficit_m"]) for row in records]
    )
    pca_deficit = np.asarray(
        [float(row["pca_reference_median_height_deficit_m"]) for row in records]
    )
    method1 = np.asarray(
        [float(row["method1_separate_valley_half_arcs_ridge_m"]) for row in records]
    )
    method2 = np.asarray([float(row["method2_midpoint_full_arc_m"]) for row in records])
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "method1": "median upper half arc around upper valley plus median lower half arc around lower valley",
                "method2": "median full dorsal arc in +/-5cm window around mean upper/lower valley station",
                "method1_vs_method2_pearson_r": float(np.corrcoef(method1, method2)[0, 1]),
                "method1_vs_method2_mean_absolute_difference_cm": float(np.mean(differences) * 100),
                "method1_vs_method2_median_absolute_difference_cm": float(np.median(differences) * 100),
                "method1_vs_method2_maximum_absolute_difference_cm": float(np.max(differences) * 100),
                "method1_vs_method2_mean_absolute_difference_percent": float(np.mean(difference_percent)),
                "ridge_reference_median_height_below_profile_peak_cm": float(np.median(ridge_deficit) * 100),
                "pca_reference_median_height_below_profile_peak_cm": float(np.median(pca_deficit) * 100),
                "method2_split_line_note": "a complete side-to-side arc is mathematically unchanged by whether it is split at the ridge or PCA line",
                "correlations": metrics,
                "records_file": str(RECORDS_CSV.relative_to(ROOT)),
                "correlations_file": str(METRICS_CSV.relative_to(ROOT)),
                "method_version": METHOD_VERSION,
                "recorded_on": RECORDED_ON,
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
