#!/usr/bin/env python3
"""Evaluate ellipse volume + torso height + virtual chest width on 61 cattle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


VOLUME_CSV = (
    BASE
    / "elliptical_slice_volumes_two_right_boundaries"
    / "elliptical_core_torso_volumes_proportional_method_61.csv"
)
OUTPUT_ROOT = BASE.parent / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
CHEST_CSV = OUTPUT_ROOT / "virtual_chest_area_records_61.csv"
SUMMARY_CSV = OUTPUT_ROOT / "elliptical_volume_height_virtual_chest_width_summary_61.csv"
PREDICTIONS_CSV = OUTPUT_ROOT / "elliptical_volume_height_virtual_chest_width_predictions_61.csv"
DETAIL_CSV = OUTPUT_ROOT / "elliptical_volume_height_virtual_chest_width_features_61.csv"
SUMMARY_JSON = OUTPUT_ROOT / "elliptical_volume_height_virtual_chest_width_summary_61.json"
SLICE_M = 0.01
RUMP_STATION_RATIO = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def core_slice_median_height(row: dict[str, str]) -> tuple[float, int]:
    """Median of per-1-cm longitudinal-slice median surface heights."""
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    x = vertices["x"].astype(float)
    y = vertices["y"].astype(float)
    z = vertices["z"].astype(float)
    points_xy = np.column_stack([x, y])
    centroid = np.array(
        [float(row["pca_full_torso_axis_centroid_x_m"]), float(row["pca_full_torso_axis_centroid_y_m"])]
    )
    axis = np.array(
        [float(row["pca_full_torso_axis_unit_x"]), float(row["pca_full_torso_axis_unit_y"])], dtype=float
    )
    axis /= np.linalg.norm(axis)
    station = (points_xy - centroid) @ axis
    rear = float(row["pca_full_torso_rear_boundary_axis_m"])
    head = float(row["pca_full_torso_head_boundary_axis_m"])
    height = float(row["ground_plane_a"]) * x + float(row["ground_plane_b"]) * y + float(
        row["ground_plane_c_m"]
    ) - z
    valid = (
        np.isfinite(station)
        & np.isfinite(height)
        & (height > 0)
        & (station >= rear)
        & (station <= head)
    )
    bins = np.floor((station[valid] - rear) / SLICE_M).astype(int)
    values = height[valid]
    medians = [float(np.median(values[bins == index])) for index in np.unique(bins)]
    if not medians:
        raise ValueError(f"no valid 1 cm slices for cow {row['cow_id']}")
    return float(np.median(medians)), len(medians)


def rump_5pct_measurement(row: dict[str, str]) -> tuple[float, float, float, int]:
    """Measure height and width in a 1-cm slice at 5% from PCA rear end."""
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    x = vertices["x"].astype(float)
    y = vertices["y"].astype(float)
    z = vertices["z"].astype(float)
    points_xy = np.column_stack([x, y])
    rear = np.array([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    head = np.array([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    vector = head - rear
    length = float(np.linalg.norm(vector))
    axis = vector / length
    normal = np.array([-axis[1], axis[0]])
    relative = points_xy - rear
    longitudinal = relative @ axis
    lateral = relative @ normal
    station = RUMP_STATION_RATIO * length
    height = float(row["ground_plane_a"]) * x + float(row["ground_plane_b"]) * y + float(
        row["ground_plane_c_m"]
    ) - z
    valid = (
        np.isfinite(longitudinal)
        & np.isfinite(lateral)
        & np.isfinite(height)
        & (height > 0)
        & (np.abs(longitudinal - station) <= SLICE_M / 2)
    )
    if np.sum(valid) < 2:
        raise ValueError(f"insufficient rump slice points for cow {row['cow_id']}")
    # Height and literal width follow the user's exact extrema-based definition.
    # The robust width is retained as a sensitivity check for isolated speckles.
    rump_height = float(np.max(height[valid]))
    rump_width = float(np.max(lateral[valid]) - np.min(lateral[valid]))
    rump_width_robust = float(np.percentile(lateral[valid], 98) - np.percentile(lateral[valid], 2))
    return rump_height, rump_width, rump_width_robust, int(np.sum(valid))


def loocv_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(y), dtype=float)
    for test_index in range(len(y)):
        train = np.arange(len(y)) != test_index
        design = np.column_stack([np.ones(np.sum(train)), x[train]])
        coefficients, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        predictions[test_index] = np.r_[1.0, x[test_index]] @ coefficients
    return predictions


def metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - y
    return {
        "loocv_pearson_r": float(np.corrcoef(y, predictions)[0, 1]),
        "loocv_predictive_r_squared": float(1 - np.sum(errors**2) / np.sum((y - np.mean(y)) ** 2)),
        "loocv_rmse_kg": float(np.sqrt(np.mean(errors**2))),
        "loocv_mae_kg": float(np.mean(np.abs(errors))),
    }


def main() -> None:
    main_rows = {row["cow_id"]: row for row in read_csv(MAIN_CSV)}
    volume_rows = {row["cow_id"]: row for row in read_csv(VOLUME_CSV)}
    chest_rows = {row["cow_id"]: row for row in read_csv(CHEST_CSV)}
    cow_ids = sorted(set(main_rows) & set(volume_rows) & set(chest_rows), key=int)
    if len(cow_ids) != 61:
        raise ValueError(f"expected 61 matched cattle, found {len(cow_ids)}")

    feature_rows: list[dict[str, object]] = []
    for cow_id in cow_ids:
        row = main_rows[cow_id]
        slice_height, slice_count = core_slice_median_height(row)
        rump_height, rump_width, rump_width_robust, rump_point_count = rump_5pct_measurement(row)
        feature_rows.append(
            {
                "cow_id": int(cow_id),
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "elliptical_core_torso_volume_m3": float(
                    volume_rows[cow_id]["elliptical_core_torso_volume_m3"]
                ),
                "rightview_median_body_depth_m": float(row["rightview_median_body_depth_m"]),
                "core_1cm_slice_median_height_m": slice_height,
                "core_1cm_slice_count": slice_count,
                "core_1cm_grid_cell_median_height_m": float(
                    row["pca_full_torso_projection_median_height_m"]
                ),
                "virtual_chest_width_m": float(chest_rows[cow_id]["virtual_chest_width_m"]),
                "rump_5pct_height_m": rump_height,
                "rump_5pct_width_m": rump_width,
                "rump_5pct_width_robust_p98_p02_m": rump_width_robust,
                "rump_5pct_slice_point_count": rump_point_count,
                "rump_station_ratio_from_rear": RUMP_STATION_RATIO,
                "rump_slice_thickness_m": SLICE_M,
                "rump_height_definition": "maximum_ground_relative_height_in_1cm_slice",
                "rump_width_definition": "maximum_lateral_minus_minimum_lateral_in_1cm_slice",
                "rump_width_robust_definition": "lateral_98th_minus_2nd_percentile_in_1cm_slice",
                "virtual_chest_width_definition": (
                    "upper_and_lower_valley_perpendicular_halfwidths_to_pca_centerline_summed"
                ),
            }
        )

    columns = {
        "V": "elliptical_core_torso_volume_m3",
        "D": "rightview_median_body_depth_m",
        "H_slice": "core_1cm_slice_median_height_m",
        "H_grid": "core_1cm_grid_cell_median_height_m",
        "W_chest": "virtual_chest_width_m",
        "H_rump": "rump_5pct_height_m",
        "W_rump": "rump_5pct_width_m",
        "W_rump_robust": "rump_5pct_width_robust_p98_p02_m",
    }
    models = [
        ("ellipse_volume", ["V"]),
        ("ellipse_volume_plus_rightview_median_depth_original", ["V", "D"]),
        ("ellipse_volume_plus_slice_median_height", ["V", "H_slice"]),
        ("ellipse_volume_plus_virtual_chest_width", ["V", "W_chest"]),
        ("ellipse_volume_plus_slice_median_height_plus_virtual_chest_width", ["V", "H_slice", "W_chest"]),
        ("ellipse_volume_plus_grid_median_height_plus_virtual_chest_width_sensitivity", ["V", "H_grid", "W_chest"]),
        ("ellipse_volume_plus_rump_5pct_height", ["V", "H_rump"]),
        ("ellipse_volume_plus_rump_5pct_width", ["V", "W_rump"]),
        ("ellipse_volume_plus_rump_5pct_height_plus_width", ["V", "H_rump", "W_rump"]),
        ("ellipse_volume_plus_rump_5pct_height_plus_robust_width_sensitivity", ["V", "H_rump", "W_rump_robust"]),
    ]
    y = np.asarray([float(row["ground_truth_weight_kg"]) for row in feature_rows])
    prediction_rows = [{"cow_id": row["cow_id"], "ground_truth_weight_kg": y[i]} for i, row in enumerate(feature_rows)]
    summary_rows: list[dict[str, object]] = []
    for model_name, feature_codes in models:
        x = np.column_stack(
            [np.asarray([float(row[columns[code]]) for row in feature_rows]) for code in feature_codes]
        )
        predictions = loocv_ols(x, y)
        result = {
            "model": model_name,
            "n": len(y),
            "model_type": "ordinary_least_squares_linear_regression",
            "validation": "leave_one_out_cross_validation",
            "features": "+".join(feature_codes),
            **metrics(y, predictions),
        }
        summary_rows.append(result)
        for index, prediction in enumerate(predictions):
            prediction_rows[index][f"predicted_weight_kg__{model_name}"] = float(prediction)

    numeric_fields = ["ground_truth_weight_kg", *columns.values()]
    matrix = np.column_stack(
        [np.asarray([float(row[field]) for row in feature_rows]) for field in numeric_fields]
    )
    correlations = {
        first: {second: float(value) for second, value in zip(numeric_fields, values)}
        for first, values in zip(numeric_fields, np.corrcoef(matrix, rowvar=False))
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path, rows in [(DETAIL_CSV, feature_rows), (SUMMARY_CSV, summary_rows), (PREDICTIONS_CSV, prediction_rows)]:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "height_primary_definition": (
                    "within PCA 45%/48% core torso, split longitudinally every 1 cm; "
                    "take median ground-relative surface height in each slice, then median across slices"
                ),
                "height_sensitivity_definition": (
                    "median of ground-relative median heights in occupied 1 cm x 1 cm XY cells"
                ),
                "rump_definition": (
                    "1 cm thick perpendicular slice centered at 5% of full PCA length from rear to head; "
                    "height is maximum ground-relative point height; literal width is maximum minus minimum lateral; "
                    "P98 minus P2 width is retained as a robustness sensitivity check"
                ),
                "models": summary_rows,
                "pearson_correlations": correlations,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
