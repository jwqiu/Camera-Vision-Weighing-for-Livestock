#!/usr/bin/env python3
"""Nested selection of 1-3 local width/height stations added to elliptical volume."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compute_two_torso_continuous_profile_models import (
    fit_ridge,
    inner_folds,
    predict_ridge,
    regression_metrics,
    select_alpha,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
VOLUME_CSV = BASE / "elliptical_slice_volumes_two_right_boundaries" / "elliptical_core_torso_volumes_proportional_method_61.csv"
STATIONS_CSV = BASE / "pca_two_torso_11_station_metrics_61" / "pca_two_torso_11_station_records_61.csv"
OUTPUT_DIR = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
SUMMARY_CSV = OUTPUT_DIR / "elliptical_volume_selected_1_to_3_local_profiles_summary_61.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "elliptical_volume_selected_1_to_3_local_profiles_predictions_61.csv"
FREQUENCY_CSV = OUTPUT_DIR / "elliptical_volume_selected_local_profile_frequency_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "elliptical_volume_selected_1_to_3_local_profiles_summary_61.json"
METHOD_VERSION = "elliptical_volume_nested_forward_select_1to3_pca4548_local_profiles_v1.0"


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


def ols_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    beta = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return np.column_stack([np.ones(len(test_x)), test_x]) @ beta


def forward_select_inside_training(
    base: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    folds = inner_folds(len(y))
    for _ in range(count):
        losses: list[tuple[float, int]] = []
        for candidate in range(candidates.shape[1]):
            if candidate in selected:
                continue
            columns = selected + [candidate]
            x = np.column_stack([base, candidates[:, columns]])
            errors: list[float] = []
            for validation in folds:
                keep = np.ones(len(y), dtype=bool)
                keep[validation] = False
                pred = ols_predict(x[keep], y[keep], x[validation])
                errors.extend(((pred - y[validation]) ** 2).tolist())
            losses.append((float(np.mean(errors)), candidate))
        selected.append(min(losses)[1])
    return tuple(selected)


def nested_predictions(
    base: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    count: int,
    model: str,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    predictions = np.empty(len(y))
    selections: list[tuple[int, ...]] = []
    tuning = np.full(len(y), np.nan)
    for outer in range(len(y)):
        keep = np.arange(len(y)) != outer
        selected = forward_select_inside_training(base[keep], candidates[keep], y[keep], count)
        selections.append(selected)
        train_x = np.column_stack([base[keep], candidates[keep][:, selected]])
        test_x = np.column_stack([base[outer : outer + 1], candidates[outer : outer + 1, selected]])
        if model == "ols":
            predictions[outer] = ols_predict(train_x, y[keep], test_x)[0]
        elif model == "ridge":
            alpha = select_alpha(train_x, y[keep])
            tuning[outer] = alpha
            predictions[outer] = predict_ridge(test_x, fit_ridge(train_x, y[keep], alpha))[0]
        else:
            raise ValueError(model)
    return predictions, selections, tuning


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
    volumes = {
        int(row["cow_id"]): float(row["elliptical_core_torso_volume_m3"])
        for row in read_csv(VOLUME_CSV)
        if row["boundary_method"] == "proportional"
    }
    profile: dict[int, dict[int, tuple[float, float]]] = {}
    for row in read_csv(STATIONS_CSV):
        if row["torso_definition_code"] != "pca_4548" or row["status"] != "ok":
            continue
        profile.setdefault(int(row["cow_id"]), {})[int(row["station_index"])] = (
            float(row["width_m"]),
            float(row["height_m"]),
        )
    cow_ids = sorted(set(volumes) & set(profile))
    if len(cow_ids) != 61 or any(len(profile[cow_id]) != 11 for cow_id in cow_ids):
        raise ValueError("expected 61 complete cattle")

    y = np.asarray([float(main_by_id[cow_id]["ground_truth_weight_kg"]) for cow_id in cow_ids])
    base = np.asarray([volumes[cow_id] for cow_id in cow_ids])[:, None]
    widths = np.asarray([[profile[cow_id][i][0] for i in range(1, 12)] for cow_id in cow_ids])
    heights = np.asarray([[profile[cow_id][i][1] for i in range(1, 12)] for cow_id in cow_ids])
    candidate_groups = {
        "width": (widths, [f"W{i}" for i in range(1, 12)]),
        "height": (heights, [f"H{i}" for i in range(1, 12)]),
        "width_height": (np.column_stack([widths, heights]), [f"W{i}" for i in range(1, 12)] + [f"H{i}" for i in range(1, 12)]),
    }

    summary_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    prediction_rows = [
        {
            "cow_id": cow_id,
            "ground_truth_weight_kg": float(y[index]),
            "proportional_elliptical_volume_m3": float(base[index, 0]),
        }
        for index, cow_id in enumerate(cow_ids)
    ]
    for group_code, (candidate_values, candidate_names) in candidate_groups.items():
        for count in (1, 2, 3):
            for model in ("ols", "ridge"):
                prediction, selections, tuning = nested_predictions(base, candidate_values, y, count, model)
                result = metric_record(prediction, y)
                selection_count = Counter(index for selected in selections for index in selected)
                row = {
                    "candidate_group": group_code,
                    "added_position_count": count,
                    "total_predictor_count": count + 1,
                    "model": model,
                    "selection_method": "forward selection by inner 5-fold RMSE inside each outer LOOCV training set",
                    **result,
                    "outer_selected_alpha_median": float(np.nanmedian(tuning)) if model == "ridge" else "",
                    "most_frequently_selected_positions": ",".join(
                        candidate_names[index]
                        for index, _ in selection_count.most_common(count)
                    ),
                    "method_version": METHOD_VERSION,
                }
                summary_rows.append(row)
                for index, prediction_row in enumerate(prediction_rows):
                    prediction_row[f"{group_code}_top{count}_{model}_prediction_kg"] = float(prediction[index])
                for candidate_index, name in enumerate(candidate_names):
                    frequency_rows.append(
                        {
                            "candidate_group": group_code,
                            "added_position_count": count,
                            "model": model,
                            "position": name,
                            "outer_fold_selection_count": selection_count[candidate_index],
                            "outer_fold_selection_rate": selection_count[candidate_index] / len(y),
                            "method_version": METHOD_VERSION,
                        }
                    )

    for row in prediction_rows:
        row["method_version"] = METHOD_VERSION
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(PREDICTIONS_CSV, prediction_rows)
    write_csv(FREQUENCY_CSV, frequency_rows)
    summary = {
        "cattle_count": len(cow_ids),
        "fixed_base_feature": "proportional elliptical slice volume using PCA 45/48 torso boundaries",
        "candidate_positions": "11 separate widths and/or 11 separate ground-relative heights from the same PCA 45/48 torso",
        "selection_and_validation": "outer leave-one-cow-out; forward position selection and Ridge alpha tuning use only each outer training set",
        "results": summary_rows,
        "method_version": METHOD_VERSION,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
