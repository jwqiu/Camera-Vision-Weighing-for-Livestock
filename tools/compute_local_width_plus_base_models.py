#!/usr/bin/env python3
"""Combine a few locally informative widths with volume, area and torso length."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compute_two_torso_continuous_profile_models import (
    ALPHAS,
    DEFINITIONS,
    STATION_CSV,
    MAIN_CSV,
    fit_pls1,
    fit_ridge,
    inner_folds,
    loocv_ols,
    nested_loocv_pls,
    nested_loocv_ridge,
    predict_pls1,
    predict_ridge,
    read_csv,
    regression_metrics,
    select_pls_components,
    select_alpha,
    write_csv,
)

OUT = Path(__file__).resolve().parents[1] / "cattle_3d_extraction_66" / "pca_two_torso_11_station_metrics_61"
SUMMARY_CSV = OUT / "local_width_plus_base_summary_61.csv"
PREDICTIONS_CSV = OUT / "local_width_plus_base_predictions_61.csv"
SELECTION_CSV = OUT / "local_width_nested_selection_frequency_61.csv"
SUMMARY_JSON = OUT / "local_width_plus_base_summary_61.json"
METHOD_VERSION = "local_width_plus_base_nested_selection_v1.0"
FIXED_WIDTHS = {
    "pca_4548": (3, 2, 9),
    "pca_rear70": (9, 3, 10),
}


def top_widths(width: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    correlations = []
    for j in range(width.shape[1]):
        if np.std(width[:, j]) == 0:
            correlations.append(0.0)
        else:
            correlations.append(abs(float(np.corrcoef(width[:, j], y)[0, 1])))
    return np.argsort(correlations)[::-1][:k]


def nested_selected_predictions(base: np.ndarray, width: np.ndarray, y: np.ndarray, k: int, model_kind: str):
    prediction = np.empty(len(y))
    selections: list[tuple[int, ...]] = []
    tuning_values = np.empty(len(y))
    for outer in range(len(y)):
        outer_keep = np.arange(len(y)) != outer
        b_train, w_train, y_train = base[outer_keep], width[outer_keep], y[outer_keep]
        fold_indices = inner_folds(len(y_train))
        if model_kind == "ridge":
            candidates = ALPHAS
        elif model_kind == "pls":
            candidates = np.arange(1, min(6, 3 + k) + 1)
        else:
            candidates = np.asarray([0])
        losses = []
        for candidate in candidates:
            errors = []
            for val in fold_indices:
                keep = np.ones(len(y_train), dtype=bool)
                keep[val] = False
                selected = top_widths(w_train[keep], y_train[keep], k)
                x_fit = np.column_stack((b_train[keep], w_train[keep][:, selected]))
                x_val = np.column_stack((b_train[val], w_train[val][:, selected]))
                if model_kind == "ridge":
                    pred = predict_ridge(x_val, fit_ridge(x_fit, y_train[keep], float(candidate)))
                elif model_kind == "pls":
                    pred = predict_pls1(x_val, fit_pls1(x_fit, y_train[keep], int(candidate)))
                else:
                    design = np.column_stack((np.ones(keep.sum()), x_fit))
                    beta, *_ = np.linalg.lstsq(design, y_train[keep], rcond=None)
                    pred = np.column_stack((np.ones(len(val)), x_val)) @ beta
                errors.extend((pred - y_train[val]) ** 2)
            losses.append(float(np.mean(errors)))
        chosen = candidates[int(np.argmin(losses))]
        selected = top_widths(w_train, y_train, k)
        selections.append(tuple(int(v + 1) for v in selected))
        x_train = np.column_stack((b_train, w_train[:, selected]))
        x_test = np.column_stack((base[outer : outer + 1], width[outer : outer + 1, selected]))
        if model_kind == "ridge":
            prediction[outer] = predict_ridge(x_test, fit_ridge(x_train, y_train, float(chosen)))[0]
        elif model_kind == "pls":
            prediction[outer] = predict_pls1(x_test, fit_pls1(x_train, y_train, int(chosen)))[0]
        else:
            design = np.column_stack((np.ones(len(y_train)), x_train))
            beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
            prediction[outer] = np.r_[1.0, x_test[0]] @ beta
        tuning_values[outer] = float(chosen)
    return prediction, selections, tuning_values


def in_sample_fit(x: np.ndarray, y: np.ndarray, model: str):
    if model == "ols":
        design = np.column_stack((np.ones(len(y)), x))
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return design @ beta, ""
    if model == "ridge":
        alpha = select_alpha(x, y)
        return predict_ridge(x, fit_ridge(x, y, alpha)), alpha
    components = select_pls_components(x, y)
    return predict_pls1(x, fit_pls1(x, y, components)), components


def main() -> None:
    main_by_id = {int(r["cow_id"]): r for r in read_csv(MAIN_CSV)}
    station_rows = read_csv(STATION_CSV)
    weights = {int(r["cow_id"]): float(r["ground_truth_weight_kg"]) for r in station_rows}
    cow_ids = sorted(weights)
    y = np.asarray([weights[cow] for cow in cow_ids])
    cube = {code: {cow: {} for cow in cow_ids} for code in DEFINITIONS}
    for row in station_rows:
        cube[row["torso_definition_code"]][int(row["cow_id"])][int(row["station_index"])] = float(row["width_m"])

    summaries: list[dict[str, object]] = []
    predictions = [{"cow_id": cow, "ground_truth_weight_kg": weights[cow]} for cow in cow_ids]
    selection_rows: list[dict[str, object]] = []
    json_results = {}
    for code, label in DEFINITIONS.items():
        width = np.asarray([[cube[code][cow][i] for i in range(1, 12)] for cow in cow_ids])
        source_rows = [main_by_id[cow] for cow in cow_ids]
        if code == "pca_4548":
            base = np.asarray([[float(r["pca_full_torso_projected_volume_m3"]), float(r["pca_full_torso_projected_area_m2"]), float(r["pca_full_torso_core_length_m"])] for r in source_rows])
        else:
            base = np.asarray([[float(r["pca_rear70_torso_projected_volume_m3"]), float(r["pca_rear70_torso_projected_area_m2"]), float(r["pca_rear70_torso_length_m"])] for r in source_rows])

        groups = [("base3", "体积+面积+长度", base, "predefined")]
        for k in (1, 2, 3):
            selected = np.asarray(FIXED_WIDTHS[code][:k]) - 1
            groups.append((f"fixed_top{k}", f"基础3项+固定局部体宽{','.join('W'+str(i+1) for i in selected)}", np.column_stack((base, width[:, selected])), "selected_on_full_sample"))

        for group_code, group_label, x, selection_method in groups:
            for model in ("ols", "ridge", "pls"):
                fitted, tuning = in_sample_fit(x, y, model)
                if model == "ols": validated = loocv_ols(x, y)
                elif model == "ridge": validated, _ = nested_loocv_ridge(x, y)
                else: validated, _ = nested_loocv_pls(x, y)
                in_m, val_m = regression_metrics(fitted, y), regression_metrics(validated, y)
                summaries.append({
                    "torso_definition": code, "torso_definition_label_zh": label,
                    "feature_group": group_code, "feature_group_label_zh": group_label,
                    "selection_method": selection_method, "model": model,
                    "model_label_zh": {"ols":"OLS","ridge":"岭回归","pls":"PLS"}[model],
                    "predictor_count": x.shape[1], "in_sample_r": in_m["r"],
                    "in_sample_r_squared": in_m["r_squared"], "in_sample_rmse_kg": in_m["rmse_kg"],
                    "validation_method": {"ols":"LOOCV OLS","ridge":"Nested LOOCV Ridge","pls":"Nested LOOCV PLS"}[model],
                    "validation_r": val_m["r"], "validation_r_squared": val_m["r_squared"],
                    "validation_rmse_kg": val_m["rmse_kg"], "validation_mae_kg": val_m["mae_kg"],
                    "tuning_value": tuning, "method_version": METHOD_VERSION,
                })
                for i, record in enumerate(predictions):
                    record[f"{code}_{group_code}_{model}_prediction_kg"] = float(validated[i])

        for k in (1, 2, 3):
            for model in ("ols", "ridge", "pls"):
                validated, selections, tuning = nested_selected_predictions(base, width, y, k, model)
                full_selected = top_widths(width, y, k)
                x_full = np.column_stack((base, width[:, full_selected]))
                fitted, full_tuning = in_sample_fit(x_full, y, model)
                in_m, val_m = regression_metrics(fitted, y), regression_metrics(validated, y)
                group_code = f"train_selected_top{k}"
                group_label = f"基础3项+训练集内选择{k}个局部体宽"
                summaries.append({
                    "torso_definition": code, "torso_definition_label_zh": label,
                    "feature_group": group_code, "feature_group_label_zh": group_label,
                    "selection_method": "selected_inside_each_validation_training_set", "model": model,
                    "model_label_zh": {"ols":"OLS","ridge":"岭回归","pls":"PLS"}[model],
                    "predictor_count": 3 + k, "in_sample_r": in_m["r"],
                    "in_sample_r_squared": in_m["r_squared"], "in_sample_rmse_kg": in_m["rmse_kg"],
                    "validation_method": {"ols":"LOOCV with training-only width selection","ridge":"Nested LOOCV Ridge with training-only width selection","pls":"Nested LOOCV PLS with training-only width selection"}[model],
                    "validation_r": val_m["r"], "validation_r_squared": val_m["r_squared"],
                    "validation_rmse_kg": val_m["rmse_kg"], "validation_mae_kg": val_m["mae_kg"],
                    "tuning_value": full_tuning, "method_version": METHOD_VERSION,
                })
                for i, record in enumerate(predictions):
                    record[f"{code}_{group_code}_{model}_prediction_kg"] = float(validated[i])
                counts = Counter(v for selected in selections for v in selected)
                for station in range(1, 12):
                    selection_rows.append({
                        "torso_definition": code, "torso_definition_label_zh": label,
                        "selected_width_count": k, "model": model,
                        "station": f"W{station}", "outer_fold_selection_count": counts[station],
                        "outer_fold_selection_rate": counts[station] / len(y),
                        "method_version": METHOD_VERSION,
                    })
        json_results[code] = {"label_zh": label, "fixed_width_ranking": [f"W{i}" for i in FIXED_WIDTHS[code]]}

    for row in predictions: row["method_version"] = METHOD_VERSION
    write_csv(SUMMARY_CSV, summaries)
    write_csv(PREDICTIONS_CSV, predictions)
    write_csv(SELECTION_CSV, selection_rows)
    SUMMARY_JSON.write_text(json.dumps({"cattle_count":61,"definitions":json_results,"results":summaries,"method_version":METHOD_VERSION},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        code: [r for r in summaries if r["torso_definition"] == code and r["model"] in ("ridge","pls")]
        for code in DEFINITIONS
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
