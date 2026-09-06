#!/usr/bin/env python3
"""Analyze 11 width and 11 height stations without averaging them."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
DATA_DIR = BASE / "pca_two_torso_11_station_metrics_61"
MAIN_CSV = BASE / "long_axis_records.csv"
STATION_CSV = DATA_DIR / "pca_two_torso_11_station_records_61.csv"
CORRELATIONS_CSV = DATA_DIR / "continuous_profile_station_correlations_61.csv"
MODEL_SUMMARY_CSV = DATA_DIR / "continuous_profile_model_summary_61.csv"
PREDICTIONS_CSV = DATA_DIR / "continuous_profile_predictions_61.csv"
COEFFICIENTS_CSV = DATA_DIR / "continuous_profile_ridge_coefficients_61.csv"
SUMMARY_JSON = DATA_DIR / "continuous_profile_summary_61.json"

DEFINITIONS = {
    "pca_4548": "原45%/48%边界",
    "pca_rear70": "臀端0%至70%",
}
STATION_RATIOS = (0.05, 0.14, 0.23, 0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95)
ALPHAS = np.logspace(-4, 5, 46)
SEED = 20260906
METHOD_VERSION = "continuous_11_width_11_height_nested_loocv_v1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    residual = pred - actual
    sse = float(np.sum(residual**2))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r": float(np.corrcoef(pred, actual)[0, 1]),
        "r_squared": 1.0 - sse / sst,
        "rmse_kg": float(np.sqrt(np.mean(residual**2))),
        "mae_kg": float(np.mean(np.abs(residual))),
    }


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    clipped = min(max(r, -0.999999), 0.999999)
    z = np.arctanh(clipped)
    se = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def loocv_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        design = np.column_stack((np.ones(keep.sum()), x[keep]))
        beta, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        pred[i] = np.r_[1.0, x[i]] @ beta
    return pred


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    y_mean = float(y.mean())
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - y_mean))
    return mean, scale, beta, y_mean


def predict_ridge(x: np.ndarray, model) -> np.ndarray:
    mean, scale, beta, y_mean = model
    return ((x - mean) / scale) @ beta + y_mean


def inner_folds(n: int, fold_count: int = 5) -> list[np.ndarray]:
    order = np.random.default_rng(SEED + n).permutation(n)
    return [order[i::fold_count] for i in range(fold_count)]


def select_alpha(x: np.ndarray, y: np.ndarray) -> float:
    fold_indices = inner_folds(len(y))
    losses = []
    for alpha in ALPHAS:
        errors = []
        for val in fold_indices:
            keep = np.ones(len(y), dtype=bool)
            keep[val] = False
            pred = predict_ridge(x[val], fit_ridge(x[keep], y[keep], float(alpha)))
            errors.extend((pred - y[val]) ** 2)
        losses.append(float(np.mean(errors)))
    return float(ALPHAS[int(np.argmin(losses))])


def nested_loocv_ridge(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, selected = np.empty(len(y)), np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        selected[i] = select_alpha(x[keep], y[keep])
        pred[i] = predict_ridge(x[i : i + 1], fit_ridge(x[keep], y[keep], selected[i]))[0]
    return pred, selected


def fit_pls1(x: np.ndarray, y: np.ndarray, components: int):
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    x_scale[x_scale == 0] = 1.0
    y_mean, y_scale = float(y.mean()), float(y.std(ddof=0))
    work_x = (x - x_mean) / x_scale
    work_y = (y - y_mean) / y_scale
    weights, loadings, y_loadings = [], [], []
    for _ in range(components):
        w = work_x.T @ work_y
        norm = float(np.linalg.norm(w))
        if norm < 1e-12:
            break
        w = w / norm
        score = work_x @ w
        denom = float(score @ score)
        if denom < 1e-12:
            break
        p = work_x.T @ score / denom
        q = float(work_y @ score / denom)
        weights.append(w)
        loadings.append(p)
        y_loadings.append(q)
        work_x = work_x - np.outer(score, p)
        work_y = work_y - score * q
    w_mat = np.column_stack(weights)
    p_mat = np.column_stack(loadings)
    q_vec = np.asarray(y_loadings)
    coef_std = w_mat @ np.linalg.pinv(p_mat.T @ w_mat) @ q_vec
    return x_mean, x_scale, y_mean, y_scale, coef_std


def predict_pls1(x: np.ndarray, model) -> np.ndarray:
    x_mean, x_scale, y_mean, y_scale, coef_std = model
    return y_mean + y_scale * (((x - x_mean) / x_scale) @ coef_std)


def select_pls_components(x: np.ndarray, y: np.ndarray) -> int:
    maximum = min(10, x.shape[1], len(y) - 2)
    fold_indices = inner_folds(len(y))
    losses = []
    for components in range(1, maximum + 1):
        errors = []
        for val in fold_indices:
            keep = np.ones(len(y), dtype=bool)
            keep[val] = False
            pred = predict_pls1(x[val], fit_pls1(x[keep], y[keep], components))
            errors.extend((pred - y[val]) ** 2)
        losses.append(float(np.mean(errors)))
    return int(np.argmin(losses) + 1)


def nested_loocv_pls(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, selected = np.empty(len(y)), np.empty(len(y), dtype=int)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        selected[i] = select_pls_components(x[keep], y[keep])
        pred[i] = predict_pls1(x[i : i + 1], fit_pls1(x[keep], y[keep], int(selected[i])))[0]
    return pred, selected


def main() -> None:
    main_by_id = {int(r["cow_id"]): r for r in read_csv(MAIN_CSV)}
    station_rows = read_csv(STATION_CSV)
    cube: dict[str, dict[int, dict[int, tuple[float, float, float]]]] = {
        code: {} for code in DEFINITIONS
    }
    weights: dict[int, float] = {}
    for row in station_rows:
        if row["status"] != "ok":
            raise ValueError(f"Non-ok station: cow {row['cow_id']}")
        code, cow_id, station = row["torso_definition_code"], int(row["cow_id"]), int(row["station_index"])
        cube[code].setdefault(cow_id, {})[station] = (
            float(row["width_m"]), float(row["height_m"]), float(row["station_ratio_full_pca"])
        )
        weights[cow_id] = float(row["ground_truth_weight_kg"])
    cow_ids = sorted(weights)
    if len(cow_ids) != 61:
        raise ValueError(f"Expected 61 cattle, got {len(cow_ids)}")
    for code in DEFINITIONS:
        if any(len(cube[code].get(cow_id, {})) != 11 for cow_id in cow_ids):
            raise ValueError(f"Incomplete profile for {code}")

    y = np.asarray([weights[cow_id] for cow_id in cow_ids])
    correlation_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_rows = [{"cow_id": cow_id, "ground_truth_weight_kg": weights[cow_id]} for cow_id in cow_ids]
    summary_json: dict[str, object] = {"cattle_count": len(cow_ids), "definitions": {}, "method_version": METHOD_VERSION}

    for code, label in DEFINITIONS.items():
        width = np.asarray([[cube[code][cow][s][0] for s in range(1, 12)] for cow in cow_ids])
        height = np.asarray([[cube[code][cow][s][1] for s in range(1, 12)] for cow in cow_ids])
        full_ratios = np.asarray([[cube[code][cow][s][2] for s in range(1, 12)] for cow in cow_ids])
        main_rows = [main_by_id[cow] for cow in cow_ids]
        if code == "pca_4548":
            base = np.asarray([[float(r["pca_full_torso_projected_volume_m3"]), float(r["pca_full_torso_projected_area_m2"]), float(r["pca_full_torso_core_length_m"])] for r in main_rows])
        else:
            base = np.asarray([[float(r["pca_rear70_torso_projected_volume_m3"]), float(r["pca_rear70_torso_projected_area_m2"]), float(r["pca_rear70_torso_length_m"])] for r in main_rows])

        for kind, values, prefix in (("width", width, "W"), ("height", height, "H")):
            for j in range(11):
                r = float(np.corrcoef(values[:, j], y)[0, 1])
                low, high = fisher_ci(r, len(y))
                loo = loocv_ols(values[:, j : j + 1], y)
                loo_metrics = regression_metrics(loo, y)
                correlation_rows.append({
                    "torso_definition": code, "torso_definition_label_zh": label,
                    "feature_type": kind, "feature": f"{prefix}{j + 1}", "station_index": j + 1,
                    "station_ratio_within_torso": STATION_RATIOS[j],
                    "mean_station_ratio_full_pca": float(full_ratios[:, j].mean()),
                    "pearson_r": r, "fisher_95ci_low": low, "fisher_95ci_high": high,
                    "loocv_r": loo_metrics["r"], "loocv_r_squared": loo_metrics["r_squared"],
                    "loocv_rmse_kg": loo_metrics["rmse_kg"], "loocv_mae_kg": loo_metrics["mae_kg"],
                    "method_version": METHOD_VERSION,
                })

        groups = (
            ("base3", "体积+面积+长度（3项）", base, ["投影体积", "投影面积", "躯干长度"]),
            ("width11", "W1–W11（11项体宽）", width, [f"W{i}" for i in range(1, 12)]),
            ("height11", "H1–H11（11项高度）", height, [f"H{i}" for i in range(1, 12)]),
            ("profile22", "W1–W11+H1–H11（22项）", np.column_stack((width, height)), [f"W{i}" for i in range(1, 12)] + [f"H{i}" for i in range(1, 12)]),
            ("full25", "体积+面积+长度+W1–W11+H1–H11（25项）", np.column_stack((base, width, height)), ["投影体积", "投影面积", "躯干长度"] + [f"W{i}" for i in range(1, 12)] + [f"H{i}" for i in range(1, 12)]),
        )
        definition_results = []
        for group_code, group_label, x, feature_names in groups:
            design = np.column_stack((np.ones(len(y)), x))
            ols_beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            ols_fit = design @ ols_beta
            ols_loo = loocv_ols(x, y)
            alpha = select_alpha(x, y)
            full_ridge_model = fit_ridge(x, y, alpha)
            ridge_fit = predict_ridge(x, full_ridge_model)
            ridge_loo, outer_alpha = nested_loocv_ridge(x, y)
            pls_components = select_pls_components(x, y)
            pls_model = fit_pls1(x, y, pls_components)
            pls_fit = predict_pls1(x, pls_model)
            pls_loo, outer_components = nested_loocv_pls(x, y)
            for model_code, model_label, fitted, validated, validation_method, model_alpha, model_components, outer_median in (
                ("ols", "OLS", ols_fit, ols_loo, "LOOCV OLS", "", "", ""),
                ("ridge", "岭回归", ridge_fit, ridge_loo, "Nested LOOCV Ridge", alpha, "", float(np.median(outer_alpha))),
                ("pls", "PLS", pls_fit, pls_loo, "Nested LOOCV PLS", "", pls_components, float(np.median(outer_components))),
            ):
                in_m, val_m = regression_metrics(fitted, y), regression_metrics(validated, y)
                model_rows.append({
                    "torso_definition": code, "torso_definition_label_zh": label,
                    "feature_group": group_code, "feature_group_label_zh": group_label,
                    "model": model_code, "model_label_zh": model_label, "predictor_count": x.shape[1],
                    "in_sample_r": in_m["r"], "in_sample_r_squared": in_m["r_squared"],
                    "in_sample_rmse_kg": in_m["rmse_kg"], "validation_method": validation_method,
                    "validation_r": val_m["r"], "validation_r_squared": val_m["r_squared"],
                    "validation_rmse_kg": val_m["rmse_kg"], "validation_mae_kg": val_m["mae_kg"],
                    "full_sample_alpha": model_alpha,
                    "outer_alpha_median": outer_median if model_code == "ridge" else "",
                    "full_sample_pls_components": model_components,
                    "outer_pls_components_median": outer_median if model_code == "pls" else "",
                    "method_version": METHOD_VERSION,
                })
            zscale = x.std(axis=0, ddof=0)
            standardized_beta = full_ridge_model[2] / y.std(ddof=0)
            for feature_name, beta in zip(feature_names, standardized_beta):
                coefficient_rows.append({
                    "torso_definition": code, "torso_definition_label_zh": label,
                    "feature_group": group_code, "feature_group_label_zh": group_label,
                    "feature": feature_name, "standardized_ridge_beta": float(beta),
                    "full_sample_alpha": alpha, "method_version": METHOD_VERSION,
                })
            for i, record in enumerate(prediction_rows):
                record[f"{code}_{group_code}_ols_loocv_prediction_kg"] = float(ols_loo[i])
                record[f"{code}_{group_code}_ridge_nested_loocv_prediction_kg"] = float(ridge_loo[i])
                record[f"{code}_{group_code}_pls_nested_loocv_prediction_kg"] = float(pls_loo[i])
            definition_results.append({"feature_group": group_code, "ridge": regression_metrics(ridge_loo, y), "alpha": alpha})

        for i, record in enumerate(prediction_rows):
            for j in range(11):
                record[f"{code}_W{j + 1}_m"] = float(width[i, j])
                record[f"{code}_H{j + 1}_m"] = float(height[i, j])
        summary_json["definitions"][code] = {"label_zh": label, "ridge_results": definition_results}

    for row in prediction_rows:
        row["method_version"] = METHOD_VERSION
    write_csv(CORRELATIONS_CSV, correlation_rows)
    write_csv(MODEL_SUMMARY_CSV, model_rows)
    write_csv(PREDICTIONS_CSV, prediction_rows)
    write_csv(COEFFICIENTS_CSV, coefficient_rows)
    SUMMARY_JSON.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "station_best": {
            code: sorted([r for r in correlation_rows if r["torso_definition"] == code], key=lambda r: abs(float(r["pearson_r"])), reverse=True)[:6]
            for code in DEFINITIONS
        },
        "models": model_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
