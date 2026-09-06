#!/usr/bin/env python3
"""Compare five weight predictors under two PCA torso definitions for 61 cattle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
OUT = BASE / "pca_two_torso_11_station_metrics_61"
MAIN_CSV = BASE / "long_axis_records.csv"
RECORDS_CSV = OUT / "two_torso_five_metric_records_61.csv"
SUMMARY_CSV = OUT / "two_torso_five_metric_summary.csv"
COEFFICIENTS_CSV = OUT / "two_torso_five_metric_coefficients.csv"
CORRELATIONS_CSV = OUT / "two_torso_five_metric_feature_correlations.csv"
SUMMARY_JSON = OUT / "two_torso_five_metric_summary.json"

DEFINITIONS = {
    "pca_4548": {
        "label": "原45%/48%边界",
        "features": (
            ("projected_volume_m3", "投影体积", "pca_full_torso_projected_volume_m3"),
            ("projected_area_m2", "Topview投影面积", "pca_full_torso_projected_area_m2"),
            ("torso_length_m", "躯干长度", "pca_full_torso_core_length_m"),
            ("mean_width_11pt_m", "11点平均体宽", "pca_full_torso_11pt_mean_width_m"),
            ("mean_height_11pt_m", "11点平均高度", "pca_full_torso_11pt_mean_height_m"),
        ),
    },
    "pca_rear70": {
        "label": "臀端0%至70%",
        "features": (
            ("projected_volume_m3", "投影体积", "pca_rear70_torso_projected_volume_m3"),
            ("projected_area_m2", "Topview投影面积", "pca_rear70_torso_projected_area_m2"),
            ("torso_length_m", "躯干长度", "pca_rear70_torso_length_m"),
            ("mean_width_11pt_m", "11点平均体宽", "pca_rear70_torso_11pt_mean_width_m"),
            ("mean_height_11pt_m", "11点平均高度", "pca_rear70_torso_11pt_mean_height_m"),
        ),
    },
}
ALPHAS = np.logspace(-4, 4, 41)
METHOD_VERSION = "two_torso_five_metrics_loocv_v1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    residual = prediction - actual
    sse = float(np.sum(residual**2))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r": float(np.corrcoef(prediction, actual)[0, 1]),
        "r_squared": 1.0 - sse / sst,
        "rmse_kg": float(np.sqrt(np.mean(residual**2))),
        "mae_kg": float(np.mean(np.abs(residual))),
    }


def loocv_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        design = np.column_stack((np.ones(keep.sum()), x[keep]))
        beta, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        prediction[i] = np.r_[1.0, x[i]] @ beta
    return prediction


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    z = (x - mean) / scale
    y_mean = float(y.mean())
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - y_mean))
    return mean, scale, beta, y_mean, z @ beta + y_mean


def predict_ridge(x: np.ndarray, model) -> np.ndarray:
    mean, scale, beta, y_mean, _ = model
    return ((x - mean) / scale) @ beta + y_mean


def folds(n: int, count: int = 5) -> list[np.ndarray]:
    return [np.arange(n)[i::count] for i in range(count)]


def select_alpha(x: np.ndarray, y: np.ndarray) -> float:
    losses = []
    for alpha in ALPHAS:
        errors = []
        for validation in folds(len(y)):
            keep = np.ones(len(y), dtype=bool)
            keep[validation] = False
            pred = predict_ridge(x[validation], fit_ridge(x[keep], y[keep], float(alpha)))
            errors.extend((pred - y[validation]) ** 2)
        losses.append(float(np.mean(errors)))
    return float(ALPHAS[int(np.argmin(losses))])


def nested_loocv_ridge(x: np.ndarray, y: np.ndarray):
    prediction = np.empty(len(y))
    selected = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        alpha = select_alpha(x[keep], y[keep])
        prediction[i] = predict_ridge(x[i : i + 1], fit_ridge(x[keep], y[keep], alpha))[0]
        selected[i] = alpha
    return prediction, selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv(MAIN_CSV)
    rows = [
        row for row in all_rows
        if row.get("pca_rear70_torso_analysis_eligible", "").upper() == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    rows.sort(key=lambda row: int(row["cow_id"]))
    if len(rows) != 61 or len({row["cow_id"] for row in rows}) != 61:
        raise ValueError(f"Expected 61 unique cattle, got {len(rows)}")

    y = np.asarray([float(row["ground_truth_weight_kg"]) for row in rows])
    detail = [{"cow_id": int(row["cow_id"]), "ground_truth_weight_kg": float(row["ground_truth_weight_kg"])} for row in rows]
    summary: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    json_defs: dict[str, object] = {}

    for definition, spec in DEFINITIONS.items():
        features = spec["features"]
        x = np.asarray([[float(row[source]) for _, _, source in features] for row in rows])
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite predictor in {definition}")

        for j, (field, label, _) in enumerate(features):
            values = x[:, j]
            raw_r = float(np.corrcoef(values, y)[0, 1])
            loo = loocv_ols(values[:, None], y)
            slope, intercept = np.polyfit(values, y, 1)
            fitted = slope * values + intercept
            in_m = metrics(fitted, y)
            val_m = metrics(loo, y)
            summary.append({
                "torso_definition": definition, "torso_definition_label_zh": spec["label"],
                "model": field, "model_label_zh": label, "predictor_count": 1,
                "raw_pearson_r": raw_r, "in_sample_r": in_m["r"],
                "in_sample_r_squared": in_m["r_squared"], "in_sample_rmse_kg": in_m["rmse_kg"],
                "validation_method": "LOOCV OLS", "validation_r": val_m["r"],
                "validation_r_squared": val_m["r_squared"], "validation_rmse_kg": val_m["rmse_kg"],
                "validation_mae_kg": val_m["mae_kg"], "ridge_alpha": "", "method_version": METHOD_VERSION,
            })
            for i, record in enumerate(detail):
                record[f"{definition}_{field}"] = float(values[i])
                record[f"{definition}_{field}_loocv_prediction_kg"] = float(loo[i])

        design = np.column_stack((np.ones(len(y)), x))
        ols_beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        ols_fitted = design @ ols_beta
        ols_loo = loocv_ols(x, y)
        full_alpha = select_alpha(x, y)
        ridge_model = fit_ridge(x, y, full_alpha)
        ridge_fitted = ridge_model[-1]
        ridge_loo, selected = nested_loocv_ridge(x, y)
        for model, label, fitted, validated, method, alpha in (
            ("five_metric_ols", "五指标OLS", ols_fitted, ols_loo, "LOOCV OLS", ""),
            ("five_metric_ridge", "五指标岭回归", ridge_fitted, ridge_loo, "Nested LOOCV Ridge", full_alpha),
        ):
            in_m, val_m = metrics(fitted, y), metrics(validated, y)
            summary.append({
                "torso_definition": definition, "torso_definition_label_zh": spec["label"],
                "model": model, "model_label_zh": label, "predictor_count": 5,
                "raw_pearson_r": "", "in_sample_r": in_m["r"],
                "in_sample_r_squared": in_m["r_squared"], "in_sample_rmse_kg": in_m["rmse_kg"],
                "validation_method": method, "validation_r": val_m["r"],
                "validation_r_squared": val_m["r_squared"], "validation_rmse_kg": val_m["rmse_kg"],
                "validation_mae_kg": val_m["mae_kg"], "ridge_alpha": alpha, "method_version": METHOD_VERSION,
            })
        for i, record in enumerate(detail):
            record[f"{definition}_five_metric_ols_loocv_prediction_kg"] = float(ols_loo[i])
            record[f"{definition}_five_metric_ols_loocv_residual_kg"] = float(ols_loo[i] - y[i])
            record[f"{definition}_five_metric_ridge_nested_loocv_prediction_kg"] = float(ridge_loo[i])
            record[f"{definition}_five_metric_ridge_nested_loocv_residual_kg"] = float(ridge_loo[i] - y[i])
            record[f"{definition}_ridge_outer_alpha"] = float(selected[i])

        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        zy = (y - y.mean()) / y.std(ddof=0)
        standardized_ols, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(y)), z)), zy, rcond=None)
        ridge_standardized = ridge_model[2] / y.std(ddof=0)
        for j, (field, label, _) in enumerate(features):
            other = np.delete(z, j, axis=1)
            other_design = np.column_stack((np.ones(len(y)), other))
            feature_fit = other_design @ np.linalg.lstsq(other_design, z[:, j], rcond=None)[0]
            feature_r2 = 1 - np.sum((z[:, j] - feature_fit) ** 2) / np.sum((z[:, j] - z[:, j].mean()) ** 2)
            coefficients.append({
                "torso_definition": definition, "torso_definition_label_zh": spec["label"],
                "feature": field, "feature_label_zh": label,
                "raw_pearson_r": float(np.corrcoef(x[:, j], y)[0, 1]),
                "standardized_ols_beta": float(standardized_ols[j + 1]),
                "standardized_ridge_beta": float(ridge_standardized[j]),
                "vif": float(1 / (1 - feature_r2)), "method_version": METHOD_VERSION,
            })

        labels = [label for _, label, _ in features] + ["体重"]
        matrix = np.corrcoef(np.column_stack((x, y)), rowvar=False)
        for i, row_label in enumerate(labels):
            correlations.append({
                "torso_definition": definition, "torso_definition_label_zh": spec["label"],
                "variable": row_label, **{labels[j]: float(matrix[i, j]) for j in range(len(labels))},
            })
        json_defs[definition] = {
            "label_zh": spec["label"], "full_sample_ridge_alpha": full_alpha,
            "outer_alpha_median": float(np.median(selected)),
        }

    for record in detail:
        record["method_version"] = METHOD_VERSION
    write_csv(RECORDS_CSV, detail)
    write_csv(SUMMARY_CSV, summary)
    write_csv(COEFFICIENTS_CSV, coefficients)
    write_csv(CORRELATIONS_CSV, correlations)
    SUMMARY_JSON.write_text(json.dumps({
        "cattle_count": 61, "definitions": json_defs, "results": summary,
        "method_version": METHOD_VERSION,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cattle_count": 61, "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
