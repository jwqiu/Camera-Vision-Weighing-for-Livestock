#!/usr/bin/env python3
"""Compare proportional elliptical volume plus 11-position width/height profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from compute_two_torso_continuous_profile_models import (
    nested_loocv_pls,
    nested_loocv_ridge,
    regression_metrics,
    loocv_ols,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
VOLUME_CSV = (
    BASE
    / "elliptical_slice_volumes_two_right_boundaries"
    / "elliptical_core_torso_volumes_proportional_method_61.csv"
)
STATIONS_CSV = (
    BASE
    / "pca_two_torso_11_station_metrics_61"
    / "pca_two_torso_11_station_records_61.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
SUMMARY_CSV = OUTPUT_DIR / "elliptical_volume_plus_11_profiles_model_summary_61.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "elliptical_volume_plus_11_profiles_predictions_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "elliptical_volume_plus_11_profiles_summary_61.json"
METHOD_VERSION = "proportional_elliptical_volume_plus_pca4548_11_profiles_nested_cv_v1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_record(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    result = regression_metrics(prediction, actual)
    return {
        "validation_r": result["r"],
        "validation_r_squared_correlation": result["r"] ** 2,
        "validation_predictive_r_squared": result["r_squared"],
        "validation_rmse_kg": result["rmse_kg"],
        "validation_mae_kg": result["mae_kg"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_by_id = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    volume_by_id = {
        int(row["cow_id"]): float(row["elliptical_core_torso_volume_m3"])
        for row in read_csv(VOLUME_CSV)
        if row["boundary_method"] == "proportional"
    }
    stations = [
        row
        for row in read_csv(STATIONS_CSV)
        if row["torso_definition_code"] == "pca_4548" and row["status"] == "ok"
    ]
    profile: dict[int, dict[int, tuple[float, float]]] = {}
    for row in stations:
        profile.setdefault(int(row["cow_id"]), {})[int(row["station_index"])] = (
            float(row["width_m"]),
            float(row["height_m"]),
        )
    cow_ids = sorted(set(volume_by_id) & set(profile))
    if len(cow_ids) != 61 or any(len(profile[cow_id]) != 11 for cow_id in cow_ids):
        raise ValueError("expected 61 cattle with complete proportional volume and 11-position profiles")

    actual = np.asarray([float(main_by_id[cow_id]["ground_truth_weight_kg"]) for cow_id in cow_ids])
    volume = np.asarray([volume_by_id[cow_id] for cow_id in cow_ids])[:, None]
    median_depth = np.asarray(
        [float(main_by_id[cow_id]["rightview_median_body_depth_m"]) for cow_id in cow_ids]
    )[:, None]
    widths = np.asarray(
        [[profile[cow_id][station][0] for station in range(1, 12)] for cow_id in cow_ids]
    )
    heights = np.asarray(
        [[profile[cow_id][station][1] for station in range(1, 12)] for cow_id in cow_ids]
    )
    groups = {
        "elliptical_volume": ("比例法椭圆切片体积", volume),
        "elliptical_volume_plus_median_depth": (
            "比例法椭圆切片体积+Rightview中位体深",
            np.column_stack([volume, median_depth]),
        ),
        "elliptical_volume_plus_width11": (
            "比例法椭圆切片体积+W1-W11",
            np.column_stack([volume, widths]),
        ),
        "elliptical_volume_plus_height11": (
            "比例法椭圆切片体积+H1-H11",
            np.column_stack([volume, heights]),
        ),
        "elliptical_volume_plus_width11_height11": (
            "比例法椭圆切片体积+W1-W11+H1-H11",
            np.column_stack([volume, widths, heights]),
        ),
    }

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = [
        {
            "cow_id": cow_id,
            "ground_truth_weight_kg": float(actual[index]),
            "proportional_elliptical_volume_m3": float(volume[index, 0]),
            **{f"W{station}_m": float(widths[index, station - 1]) for station in range(1, 12)},
            **{f"H{station}_m": float(heights[index, station - 1]) for station in range(1, 12)},
        }
        for index, cow_id in enumerate(cow_ids)
    ]
    json_groups: dict[str, object] = {}
    for group_code, (group_label, features) in groups.items():
        model_predictions: dict[str, np.ndarray] = {"ols_loocv": loocv_ols(features, actual)}
        selected: dict[str, object] = {"ols_loocv": {}}
        if group_code not in {"elliptical_volume", "elliptical_volume_plus_median_depth"}:
            ridge_prediction, ridge_alpha = nested_loocv_ridge(features, actual)
            pls_prediction, pls_components = nested_loocv_pls(features, actual)
            model_predictions["ridge_nested_loocv"] = ridge_prediction
            model_predictions["pls_nested_loocv"] = pls_prediction
            selected["ridge_nested_loocv"] = {
                "outer_selected_alpha_median": float(np.median(ridge_alpha))
            }
            selected["pls_nested_loocv"] = {
                "outer_selected_components_median": float(np.median(pls_components))
            }
        group_results = []
        for model_code, prediction in model_predictions.items():
            result = metric_record(prediction, actual)
            row = {
                "feature_group": group_code,
                "feature_group_label_zh": group_label,
                "predictor_count": features.shape[1],
                "model": model_code,
                **result,
                **selected[model_code],
                "method_version": METHOD_VERSION,
            }
            summary_rows.append(row)
            group_results.append(row)
            for index, prediction_row in enumerate(prediction_rows):
                prediction_row[f"{group_code}_{model_code}_prediction_kg"] = float(prediction[index])
        json_groups[group_code] = group_results

    for row in prediction_rows:
        row["method_version"] = METHOD_VERSION
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(PREDICTIONS_CSV, prediction_rows)
    summary = {
        "cattle_count": len(cow_ids),
        "torso_definition": "PCA rear first below 45% to head first below 48%",
        "station_ratios_within_torso": [0.05, 0.14, 0.23, 0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95],
        "width_definition": "11 separate top-view body widths; not averaged",
        "height_definition": "11 separate ground-relative top-view heights; not averaged",
        "validation": "outer leave-one-cow-out; Ridge/PLS tuning performed inside each outer training fold",
        "groups": json_groups,
        "method_version": METHOD_VERSION,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
