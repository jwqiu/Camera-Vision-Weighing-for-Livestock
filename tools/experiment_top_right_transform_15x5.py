#!/usr/bin/env python3
"""Estimate a 15-cow preliminary transform, test on 5 cows, and record both boundary methods."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

import experiment_top_right_transform_3x3 as core


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
OUTPUT = BASE / "top_right_transform_experiment_15x5"
SEED = 20260906
TRAIN_COUNT = 15
VALIDATION_COUNT = 5
METHOD_VERSION = "top_right_fixed_rigid_endpoint_anchor_15train_5validation_v1.0"
PROPORTIONAL_VERSION = "rightview_full_body_endpoint_proportional_pca45_48_v1.0"


def matrix_record(result: dict[str, object]) -> dict[str, object]:
    return {
        "cow_id": result["cow_id"],
        "sample_role": result["split"],
        "right_source_ply": result["right_source_ply"],
        "rightview_rear_boundary_x_m": result["matrix_predicted_rear_x_m"],
        "rightview_head_boundary_x_m": result["matrix_predicted_head_x_m"],
        "rightview_core_length_m": result["matrix_predicted_core_length_m"],
        "top_rear_boundary_ratio": result["top_rear_boundary_ratio"],
        "top_head_boundary_ratio": result["top_head_boundary_ratio"],
        "method_version": METHOD_VERSION,
        "coordinate_system": "right_camera_x_m; head_at_lower_x; rear_at_higher_x",
        "status": "experimental_not_ground_truth_validated",
    }


def proportional_record(result: dict[str, object]) -> dict[str, object]:
    return {
        "cow_id": result["cow_id"],
        "sample_role": result["split"],
        "right_source_ply": result["right_source_ply"],
        "rightview_rear_boundary_x_m": result["reference_rear_x_m"],
        "rightview_head_boundary_x_m": result["reference_head_x_m"],
        "rightview_core_length_m": result["reference_core_length_m"],
        "top_rear_boundary_ratio": result["top_rear_boundary_ratio"],
        "top_head_boundary_ratio": result["top_head_boundary_ratio"],
        "right_robust_rear_endpoint_x_m": result["right_robust_rear_x_m"],
        "right_robust_head_endpoint_x_m": result["right_robust_head_x_m"],
        "method_version": PROPORTIONAL_VERSION,
        "coordinate_system": "right_camera_x_m; head_at_lower_x; rear_at_higher_x",
        "status": "experimental_proportional_reference_not_manual_ground_truth",
    }


def comparison_record(result: dict[str, object]) -> dict[str, object]:
    return {
        "cow_id": result["cow_id"],
        "sample_role": result["split"],
        "right_source_ply": result["right_source_ply"],
        "matrix_rear_boundary_x_m": result["matrix_predicted_rear_x_m"],
        "proportional_rear_boundary_x_m": result["reference_rear_x_m"],
        "rear_matrix_minus_proportional_m": result["rear_error_m"],
        "matrix_head_boundary_x_m": result["matrix_predicted_head_x_m"],
        "proportional_head_boundary_x_m": result["reference_head_x_m"],
        "head_matrix_minus_proportional_m": result["head_error_m"],
        "matrix_core_length_m": result["matrix_predicted_core_length_m"],
        "proportional_core_length_m": result["reference_core_length_m"],
        "core_length_matrix_minus_proportional_m": result["core_length_error_m"],
        "mean_absolute_boundary_difference_m": result["mean_absolute_boundary_error_m"],
        "matrix_method_version": METHOD_VERSION,
        "proportional_method_version": PROPORTIONAL_VERSION,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    core.METHOD_VERSION = METHOD_VERSION
    rows = core.strict_rows()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    train_rows = rows[:TRAIN_COUNT]
    validation_rows = rows[TRAIN_COUNT : TRAIN_COUNT + VALIDATION_COUNT]
    train_ids = {int(row["cow_id"]) for row in train_rows}
    validation_ids = {int(row["cow_id"]) for row in validation_rows}

    # Load all 61 once because both methods must be recorded for the complete strict set.
    observations = [(row, core.load_right_observation(row)) for row in rows]
    train = [(row, right) for row, right in observations if int(row["cow_id"]) in train_ids]
    matrix, diagnostics = core.estimate_transform(train)

    results = []
    qa_paths = []
    for row, right in observations:
        cow_id = int(row["cow_id"])
        if cow_id in train_ids:
            split = "train"
        elif cow_id in validation_ids:
            split = "validation"
        else:
            split = "application"
        result = core.evaluate(row, right, matrix, split)
        results.append(result)
        if split in {"train", "validation"}:
            qa_path = OUTPUT / "qa" / f"cow_{cow_id:03d}_{split}.png"
            core.render_qa(result, right, qa_path)
            qa_paths.append(qa_path)

    validation = [result for result in results if result["split"] == "validation"]
    endpoint_errors = np.array(
        [abs(float(result[key])) for result in validation for key in ("rear_error_m", "head_error_m")]
    )
    per_cow_errors = np.array([float(result["mean_absolute_boundary_error_m"]) for result in validation])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix_rows = [matrix_record(result) for result in sorted(results, key=lambda item: int(item["cow_id"]))]
    proportional_rows = [
        proportional_record(result) for result in sorted(results, key=lambda item: int(item["cow_id"]))
    ]
    comparison_rows = [
        comparison_record(result) for result in sorted(results, key=lambda item: int(item["cow_id"]))
    ]
    write_csv(OUTPUT / "rightview_core_boundaries_matrix_method_61.csv", matrix_rows)
    write_csv(OUTPUT / "rightview_core_boundaries_proportional_method_61.csv", proportional_rows)
    write_csv(OUTPUT / "rightview_core_boundaries_two_methods_comparison_61.csv", comparison_rows)

    validation_qa = [path for path in qa_paths if "_validation" in path.name]
    training_qa = [path for path in qa_paths if "_train" in path.name]
    core.combine_qa(training_qa, OUTPUT / "training_15_cows_combined.png")
    core.combine_qa(validation_qa, OUTPUT / "validation_5_cows_combined.png")
    np.savetxt(OUTPUT / "top_to_right_matrix_4x4.txt", matrix, fmt="%.9f")

    summary = {
        "experiment": METHOD_VERSION,
        "random_seed": SEED,
        "strict_cattle_count": len(rows),
        "training_cow_ids": [int(row["cow_id"]) for row in train_rows],
        "validation_cow_ids": [int(row["cow_id"]) for row in validation_rows],
        "top_to_right_matrix": matrix.tolist(),
        "matrix_scope": "preliminary fixed rigid transform; longitudinal boundary transfer is evaluated",
        "validation_comparison_target": "independent within-cow full-body proportional boundary method",
        "validation_mean_absolute_boundary_difference_m": float(np.mean(endpoint_errors)),
        "validation_median_absolute_boundary_difference_m": float(np.median(endpoint_errors)),
        "validation_max_absolute_boundary_difference_m": float(np.max(endpoint_errors)),
        "validation_per_cow_mean_absolute_difference_m": per_cow_errors.tolist(),
        "acceptance_rule": "exploratory pass only if mean <= 0.05 m and max <= 0.10 m",
        "preliminary_pass": bool(np.mean(endpoint_errors) <= 0.05 and np.max(endpoint_errors) <= 0.10),
        "diagnostics": diagnostics,
        "outputs": {
            "matrix_method_61": "rightview_core_boundaries_matrix_method_61.csv",
            "proportional_method_61": "rightview_core_boundaries_proportional_method_61.csv",
            "side_by_side_comparison_61": "rightview_core_boundaries_two_methods_comparison_61.csv",
            "validation_qa": "validation_5_cows_combined.png",
        },
        "limitations": [
            "No calibration target or author-supplied extrinsic matrix is available.",
            "Differences are between two estimated methods, not errors against manually surveyed ground truth.",
            "Neither method is promoted to the existing formal long_axis_records.csv in this experiment.",
        ],
    }
    with (OUTPUT / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "training_cow_ids": summary["training_cow_ids"],
        "validation_cow_ids": summary["validation_cow_ids"],
        "matrix": summary["top_to_right_matrix"],
        "validation_mean_difference_m": summary["validation_mean_absolute_boundary_difference_m"],
        "validation_median_difference_m": summary["validation_median_absolute_boundary_difference_m"],
        "validation_max_difference_m": summary["validation_max_absolute_boundary_difference_m"],
        "preliminary_pass": summary["preliminary_pass"],
        "matrix_record_count": len(matrix_rows),
        "proportional_record_count": len(proportional_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
