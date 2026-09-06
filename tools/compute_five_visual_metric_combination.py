#!/usr/bin/env python3
"""Evaluate a five-feature visual weight model for the same 61 cattle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
OUTPUT_DIR = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec"
MAIN_CSV = BASE / "long_axis_records.csv"
CHEST_CSV = OUTPUT_DIR / "virtual_chest_area_records_61.csv"
RECORDS_CSV = OUTPUT_DIR / "five_visual_metric_model_records_61.csv"
SUMMARY_CSV = OUTPUT_DIR / "five_visual_metric_model_summary.csv"
COEFFICIENTS_CSV = OUTPUT_DIR / "five_visual_metric_model_coefficients.csv"
CORRELATION_MATRIX_CSV = OUTPUT_DIR / "five_visual_metric_feature_correlations.csv"
SUMMARY_JSON = OUTPUT_DIR / "five_visual_metric_model_summary.json"
METHOD_VERSION = "five_visual_metrics_ols_nested_ridge_v1.0"
RECORDED_ON = "2026-09-06"

FEATURES = (
    ("topview_max_width_m", "Topview最大体宽", "pca_full_torso_max_width_m"),
    ("virtual_chest_width_m", "虚拟胸宽", "virtual_chest_width_m"),
    ("torso_length_m", "PCA躯干长度", "pca_full_torso_core_length_m"),
    ("rightview_chest_depth_m", "Rightview胸区体深", "rightview_chest_depth_m"),
    ("topview_torso_height_m", "Topview躯干最大离地高度", "pca_full_torso_projection_max_height_m"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_standardized_ridge(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    z = (x_train - mean) / scale
    y_mean = float(y_train.mean())
    centered_y = y_train - y_mean
    penalty = np.eye(z.shape[1]) * alpha
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ centered_y)
    return mean, scale, beta, y_mean, z @ beta + y_mean


def predict_ridge(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray, beta: np.ndarray, y_mean: float
) -> np.ndarray:
    return ((x - mean) / scale) @ beta + y_mean


def deterministic_folds(n: int, fold_count: int = 5) -> list[np.ndarray]:
    return [np.arange(n)[index::fold_count] for index in range(fold_count)]


def select_alpha(x: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> float:
    folds = deterministic_folds(len(y), 5)
    scores = []
    for alpha in alphas:
        squared_errors = []
        for validation in folds:
            keep = np.ones(len(y), dtype=bool)
            keep[validation] = False
            mean, scale, beta, y_mean, _ = fit_standardized_ridge(
                x[keep], y[keep], float(alpha)
            )
            prediction = predict_ridge(x[validation], mean, scale, beta, y_mean)
            squared_errors.extend((prediction - y[validation]) ** 2)
        scores.append(float(np.mean(squared_errors)))
    return float(alphas[int(np.argmin(scores))])


def regression_metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    residual = prediction - actual
    sse = float(np.sum(residual**2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "r": float(np.corrcoef(prediction, actual)[0, 1]),
        "r_squared": 1 - sse / sst,
        "rmse_kg": float(np.sqrt(np.mean(residual**2))),
        "mae_kg": float(np.mean(np.abs(residual))),
    }


def loocv_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        design = np.column_stack((np.ones(np.count_nonzero(keep)), x[keep]))
        coefficients, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        predictions[index] = np.r_[1.0, x[index]] @ coefficients
    return predictions


def nested_loocv_ridge(
    x: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(len(y), dtype=float)
    selected = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        alpha = select_alpha(x[keep], y[keep], alphas)
        mean, scale, beta, y_mean, _ = fit_standardized_ridge(
            x[keep], y[keep], alpha
        )
        predictions[index] = predict_ridge(
            x[index : index + 1], mean, scale, beta, y_mean
        )[0]
        selected[index] = alpha
    return predictions, selected


def univariate_loocv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        predictions[index] = slope * x[index] + intercept
    return predictions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    chest_rows = sorted(read_csv(CHEST_CSV), key=lambda row: int(row["cow_id"]))
    if len(chest_rows) != 61:
        raise ValueError(f"expected 61 cattle, found {len(chest_rows)}")

    records: list[dict[str, object]] = []
    for chest in chest_rows:
        cow_id = int(chest["cow_id"])
        main = main_rows[cow_id]
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(chest["ground_truth_weight_kg"]),
                "topview_max_width_m": float(main["pca_full_torso_max_width_m"]),
                "virtual_chest_width_m": float(chest["virtual_chest_width_m"]),
                "torso_length_m": float(main["pca_full_torso_core_length_m"]),
                "rightview_chest_depth_m": float(chest["rightview_chest_depth_m"]),
                "topview_torso_height_m": float(main["pca_full_torso_projection_max_height_m"]),
            }
        )

    x = np.asarray([[float(row[field]) for field, _, _ in FEATURES] for row in records])
    y = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    design = np.column_stack((np.ones(len(y)), x))
    ols_coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    ols_fitted = design @ ols_coef
    ols_loo = loocv_ols(x, y)

    alphas = np.logspace(-4, 4, 41)
    ridge_loo, selected_alphas = nested_loocv_ridge(x, y, alphas)
    full_alpha = select_alpha(x, y, alphas)
    mean, scale, ridge_beta, y_mean, ridge_fitted = fit_standardized_ridge(
        x, y, full_alpha
    )

    for index, row in enumerate(records):
        row["ols_fitted_weight_kg"] = float(ols_fitted[index])
        row["ols_residual_kg"] = float(ols_fitted[index] - y[index])
        row["ols_loocv_predicted_weight_kg"] = float(ols_loo[index])
        row["ols_loocv_residual_kg"] = float(ols_loo[index] - y[index])
        row["ridge_fitted_weight_kg"] = float(ridge_fitted[index])
        row["ridge_nested_loocv_predicted_weight_kg"] = float(ridge_loo[index])
        row["ridge_nested_loocv_residual_kg"] = float(ridge_loo[index] - y[index])
        row["ridge_outer_selected_alpha"] = float(selected_alphas[index])
        row["method_version"] = METHOD_VERSION
    write_csv(RECORDS_CSV, records)

    summary_rows: list[dict[str, object]] = []
    for feature_index, (field, label, _) in enumerate(FEATURES):
        values = x[:, feature_index]
        prediction = univariate_loocv(values, y)
        in_sample_r = float(np.corrcoef(values, y)[0, 1])
        metrics = regression_metrics(prediction, y)
        summary_rows.append(
            {
                "model": label,
                "predictor_count": 1,
                "in_sample_r": in_sample_r,
                "in_sample_r_squared": in_sample_r**2,
                "in_sample_rmse_kg": float(
                    np.sqrt(np.mean((np.polyval(np.polyfit(values, y, 1), values) - y) ** 2))
                ),
                "validation_method": "LOOCV OLS",
                "validation_r": metrics["r"],
                "validation_r_squared": metrics["r_squared"],
                "validation_rmse_kg": metrics["rmse_kg"],
                "validation_mae_kg": metrics["mae_kg"],
                "ridge_alpha": "",
                "method_version": METHOD_VERSION,
            }
        )

    for label, fitted, validated, validation_method, alpha in (
        ("五指标OLS", ols_fitted, ols_loo, "LOOCV OLS", ""),
        ("五指标岭回归", ridge_fitted, ridge_loo, "Nested LOOCV Ridge", full_alpha),
    ):
        in_metrics = regression_metrics(fitted, y)
        validation_metrics = regression_metrics(validated, y)
        summary_rows.append(
            {
                "model": label,
                "predictor_count": 5,
                "in_sample_r": in_metrics["r"],
                "in_sample_r_squared": in_metrics["r_squared"],
                "in_sample_rmse_kg": in_metrics["rmse_kg"],
                "validation_method": validation_method,
                "validation_r": validation_metrics["r"],
                "validation_r_squared": validation_metrics["r_squared"],
                "validation_rmse_kg": validation_metrics["rmse_kg"],
                "validation_mae_kg": validation_metrics["mae_kg"],
                "ridge_alpha": alpha,
                "method_version": METHOD_VERSION,
            }
        )
    write_csv(SUMMARY_CSV, summary_rows)

    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    standardized_ols, *_ = np.linalg.lstsq(
        np.column_stack((np.ones(len(y)), z)),
        (y - y.mean()) / y.std(ddof=0),
        rcond=None,
    )
    coefficient_rows = []
    for index, (field, label, _) in enumerate(FEATURES):
        other = np.delete(z, index, axis=1)
        fitted_feature = np.column_stack((np.ones(len(y)), other)) @ np.linalg.lstsq(
            np.column_stack((np.ones(len(y)), other)), z[:, index], rcond=None
        )[0]
        feature_r2 = 1 - np.sum((z[:, index] - fitted_feature) ** 2) / np.sum(
            (z[:, index] - np.mean(z[:, index])) ** 2
        )
        coefficient_rows.append(
            {
                "feature_field": field,
                "feature_label_zh": label,
                "univariate_pearson_r": float(np.corrcoef(x[:, index], y)[0, 1]),
                "standardized_ols_beta": float(standardized_ols[index + 1]),
                "standardized_ridge_beta": float(ridge_beta[index] / y.std(ddof=0)),
                "vif": float(1 / (1 - feature_r2)),
                "method_version": METHOD_VERSION,
            }
        )
    write_csv(COEFFICIENTS_CSV, coefficient_rows)

    matrix_labels = [label for _, label, _ in FEATURES] + ["体重"]
    matrix_values = np.column_stack((x, y))
    matrix = np.corrcoef(matrix_values, rowvar=False)
    matrix_rows = []
    for i, label in enumerate(matrix_labels):
        matrix_rows.append(
            {"variable": label, **{matrix_labels[j]: float(matrix[i, j]) for j in range(len(matrix_labels))}}
        )
    write_csv(CORRELATION_MATRIX_CSV, matrix_rows)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "features": [
                    {"field": field, "label_zh": label, "source_field": source}
                    for field, label, source in FEATURES
                ],
                "model_results": summary_rows,
                "coefficients": coefficient_rows,
                "feature_correlation_matrix": matrix_rows,
                "full_sample_ridge_alpha": full_alpha,
                "outer_selected_alpha_median": float(np.median(selected_alphas)),
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
