#!/usr/bin/env python3
"""Estimate a preliminary fixed Top->Right transform from 3 cows and test on 3 cows.

This is an explicitly exploratory experiment.  In the absence of calibration targets,
the transform is anchored with robust rear/head silhouette endpoints from synchronized
Top and Right observations.  Held-out validation compares matrix-projected PCA 45/48
boundaries with an independent within-animal endpoint-normalized reference.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compute_rightview_median_body_depth import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    fit_ground_plane,
    read_ply_xyz,
    segment_cow,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
RECORDS = BASE / "long_axis_records.csv"
OUTPUT = BASE / "top_right_transform_experiment_3x3"
SEED = 20260906
TRAIN_COUNT = 3
VALIDATION_COUNT = 3
METHOD_VERSION = "top_right_fixed_rigid_endpoint_anchor_3train_3validation_v1.0"


def strict_rows() -> list[dict[str, str]]:
    with RECORDS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible") == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]


def load_right_observation(row: dict[str, str]) -> dict[str, object]:
    cow_id = int(row["cow_id"])
    source = ROOT / row["rightview_source_ply"]
    xyz = read_ply_xyz(source).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    ground, fraction = fit_ground_plane(xyz, np.random.default_rng(SEED + cow_id))
    mask, height = segment_cow(xyz, ground)
    points = xyz[mask].astype(float)
    x = points[:, 0]
    return {
        "source": str(source.relative_to(ROOT)),
        "xyz": xyz,
        "mask": mask,
        "height": height,
        "points": points,
        "head_x_m": float(np.quantile(x, 0.005)),
        "rear_x_m": float(np.quantile(x, 0.995)),
        "median_z_m": float(np.median(points[:, 2])),
        "ground": ground,
        "ground_inlier_fraction": fraction,
    }


def endpoint_pairs(row: dict[str, str], right: dict[str, object]) -> list[tuple[np.ndarray, float, str]]:
    rear = np.array([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    head = np.array([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    return [
        (rear, float(right["rear_x_m"]), "rear"),
        (head, float(right["head_x_m"]), "head"),
    ]


def estimate_transform(train: list[tuple[dict[str, str], dict[str, object]]]) -> tuple[np.ndarray, dict[str, object]]:
    top_xy: list[np.ndarray] = []
    right_x: list[float] = []
    pair_details = []
    for row, right in train:
        for xy, rx, label in endpoint_pairs(row, right):
            top_xy.append(xy)
            right_x.append(rx)
            pair_details.append({"cow_id": int(row["cow_id"]), "anchor": label})

    x = np.asarray(top_xy)
    y = np.asarray(right_x)
    design = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    raw_scale = float(np.linalg.norm(beta[:2]))
    raw_free_direction = (beta[:2] / raw_scale).tolist()

    # The fixed rig and all inspected images establish that the two image-horizontal
    # axes follow the chute in opposite directions.  Three cows are not enough to
    # estimate yaw freely: endpoint/pose differences spuriously rotate the solution.
    # Hold that physical rotation fixed and estimate only the shared longitudinal
    # camera offset.  A cow-level midpoint cancels much of the view-dependent length
    # difference between nose and rump endpoints.
    direction = np.array([-1.0, 0.0])
    tx_by_cow = []
    for row, right in train:
        offsets = [rx - float(xy @ direction) for xy, rx, _ in endpoint_pairs(row, right)]
        tx_by_cow.append({"cow_id": int(row["cow_id"]), "offset_m": float(np.mean(offsets))})
    tx = float(np.median([item["offset_m"] for item in tx_by_cow]))

    row_x = np.array([direction[0], direction[1], 0.0])
    row_y = np.array([0.0, 0.0, -1.0])
    row_z = np.cross(row_x, row_y)
    rotation = np.vstack([row_x, row_y, row_z])

    ty_values = []
    tz_values = []
    for row, right in train:
        cx = float(row["centroid_x_m"])
        cy = float(row["centroid_y_m"])
        top_ground_z = (
            float(row["ground_plane_a"]) * cx
            + float(row["ground_plane_b"]) * cy
            + float(row["ground_plane_c_m"])
        )
        points = np.asarray(right["points"])
        mx = float(np.median(points[:, 0]))
        mz = float(np.median(points[:, 2]))
        ground = np.asarray(right["ground"])
        right_ground_y = float(ground[0] * mx + ground[1] * mz + ground[2])
        ty_values.append(right_ground_y + top_ground_z)
        tz_values.append(mz - (row_z[0] * cx + row_z[1] * cy))

    translation = np.array([tx, float(np.median(ty_values)), float(np.median(tz_values))])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation

    predicted = x @ direction + tx
    residuals = y - predicted
    diagnostics = {
        "unconstrained_longitudinal_scale": raw_scale,
        "unconstrained_longitudinal_direction_rejected": raw_free_direction,
        "longitudinal_direction_constraint": "right_x = -top_x + tx (fixed opposing chute axes)",
        "training_cow_midpoint_offset_samples": tx_by_cow,
        "training_anchor_rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "training_anchor_median_absolute_error_m": float(np.median(np.abs(residuals))),
        "training_anchor_residuals": [
            {**detail, "residual_m": float(residual)}
            for detail, residual in zip(pair_details, residuals)
        ],
        "vertical_translation_samples_m": ty_values,
        "lateral_translation_samples_m": tz_values,
    }
    return matrix, diagnostics


def boundary_ratio(row: dict[str, str], prefix: str) -> float:
    start = np.array([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    end = np.array([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    boundary = np.array(
        [
            float(row[f"pca_full_torso_{prefix}_boundary_x_m"]),
            float(row[f"pca_full_torso_{prefix}_boundary_y_m"]),
        ]
    )
    axis = end - start
    return float(np.dot(boundary - start, axis) / np.dot(axis, axis))


def predicted_boundary_x(row: dict[str, str], matrix: np.ndarray, prefix: str) -> float:
    point = np.array(
        [
            float(row[f"pca_full_torso_{prefix}_boundary_x_m"]),
            float(row[f"pca_full_torso_{prefix}_boundary_y_m"]),
            float(row["ground_plane_c_m"]),
            1.0,
        ]
    )
    return float((matrix @ point)[0])


def evaluate(
    row: dict[str, str], right: dict[str, object], matrix: np.ndarray, split: str
) -> dict[str, object]:
    rear_ratio = boundary_ratio(row, "rear")
    head_ratio = boundary_ratio(row, "head")
    right_rear = float(right["rear_x_m"])
    right_head = float(right["head_x_m"])
    reference_rear = right_rear + rear_ratio * (right_head - right_rear)
    reference_head = right_rear + head_ratio * (right_head - right_rear)
    predicted_rear = predicted_boundary_x(row, matrix, "rear")
    predicted_head = predicted_boundary_x(row, matrix, "head")
    rear_error = predicted_rear - reference_rear
    head_error = predicted_head - reference_head
    predicted_length = predicted_rear - predicted_head
    reference_length = reference_rear - reference_head
    return {
        "cow_id": int(row["cow_id"]),
        "split": split,
        "top_source_ply": row["source_top_ply"],
        "right_source_ply": right["source"],
        "top_rear_boundary_ratio": rear_ratio,
        "top_head_boundary_ratio": head_ratio,
        "right_robust_head_x_m": right_head,
        "right_robust_rear_x_m": right_rear,
        "matrix_predicted_rear_x_m": predicted_rear,
        "matrix_predicted_head_x_m": predicted_head,
        "reference_rear_x_m": reference_rear,
        "reference_head_x_m": reference_head,
        "rear_error_m": rear_error,
        "head_error_m": head_error,
        "mean_absolute_boundary_error_m": (abs(rear_error) + abs(head_error)) / 2,
        "matrix_predicted_core_length_m": predicted_length,
        "reference_core_length_m": reference_length,
        "core_length_error_m": predicted_length - reference_length,
        "right_ground_fit_inlier_fraction": right["ground_inlier_fraction"],
        "method_version": METHOD_VERSION,
    }


def render_qa(result: dict[str, object], right: dict[str, object], output: Path) -> None:
    points = np.asarray(right["points"])
    ground = np.asarray(right["ground"])
    x = points[:, 0]
    plane_y = ground[0] * points[:, 0] + ground[1] * points[:, 2] + ground[2]
    h = (points[:, 1] - plane_y) / np.sqrt(1 + ground[0] ** 2 + ground[1] ** 2)
    x0, x1 = np.quantile(x, [0.002, 0.998])
    h0, h1 = np.quantile(h, [0.01, 0.995])
    width, height, margin = 1100, 560, 55

    def px(value):
        return margin + (np.asarray(value) - x0) / (x1 - x0) * (width - 2 * margin)

    def py(value):
        return height - margin - (np.asarray(value) - h0) / (h1 - h0) * (height - 2 * margin)

    image = Image.new("RGB", (width, height), (8, 13, 21))
    draw = ImageDraw.Draw(image)
    sample = np.arange(0, len(x), max(1, len(x) // 50000))
    for xx, yy in zip(px(x[sample]).astype(int), py(h[sample]).astype(int)):
        if margin <= xx < width - margin and margin <= yy < height - margin:
            draw.point((int(xx), int(yy)), fill=(100, 118, 140))

    pred_rear = float(result["matrix_predicted_rear_x_m"])
    pred_head = float(result["matrix_predicted_head_x_m"])
    ref_rear = float(result["reference_rear_x_m"])
    ref_head = float(result["reference_head_x_m"])
    for value, color, line_width in [
        (pred_rear, (255, 88, 88), 5),
        (pred_head, (255, 205, 65), 5),
        (ref_rear, (80, 220, 255), 2),
        (ref_head, (80, 220, 255), 2),
    ]:
        xx = int(px(value))
        draw.line((xx, margin, xx, height - margin), fill=color, width=line_width)

    title = f"cow {int(result['cow_id']):03d}  {result['split']}  fixed-matrix Top->Right boundary test"
    errors = (
        f"rear error {float(result['rear_error_m']) * 100:+.1f} cm    "
        f"head error {float(result['head_error_m']) * 100:+.1f} cm    "
        f"mean abs {float(result['mean_absolute_boundary_error_m']) * 100:.1f} cm"
    )
    draw.text((margin, 12), title, fill=(240, 245, 250))
    draw.text((margin, 31), errors, fill=(190, 200, 215))
    draw.text((margin, height - 35), "thick red/yellow: matrix prediction    thin cyan: independent proportional reference", fill=(190, 200, 215))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def combine_qa(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    canvas = Image.new("RGB", (width, sum(image.height for image in images)), (8, 13, 21))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output)


def main() -> None:
    rows = strict_rows()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    train_rows = rows[:TRAIN_COUNT]
    validation_rows = rows[TRAIN_COUNT : TRAIN_COUNT + VALIDATION_COUNT]
    selected = [(row, load_right_observation(row)) for row in [*train_rows, *validation_rows]]
    train = selected[:TRAIN_COUNT]
    matrix, diagnostics = estimate_transform(train)

    results = []
    qa_paths: list[Path] = []
    for index, (row, right) in enumerate(selected):
        split = "train" if index < TRAIN_COUNT else "validation"
        result = evaluate(row, right, matrix, split)
        results.append(result)
        qa_path = OUTPUT / "qa" / f"cow_{int(row['cow_id']):03d}_{split}.png"
        render_qa(result, right, qa_path)
        qa_paths.append(qa_path)

    combine_qa(qa_paths[:TRAIN_COUNT], OUTPUT / "training_3_cows_combined.png")
    combine_qa(qa_paths[TRAIN_COUNT:], OUTPUT / "validation_3_cows_combined.png")

    validation_errors = np.array(
        [float(result["mean_absolute_boundary_error_m"]) for result in results if result["split"] == "validation"]
    )
    validation_endpoint_errors = np.array(
        [
            abs(float(result[key]))
            for result in results
            if result["split"] == "validation"
            for key in ("rear_error_m", "head_error_m")
        ]
    )
    summary = {
        "experiment": METHOD_VERSION,
        "random_seed": SEED,
        "strict_candidate_count": len(rows),
        "training_cow_ids": [int(row["cow_id"]) for row in train_rows],
        "validation_cow_ids": [int(row["cow_id"]) for row in validation_rows],
        "top_to_right_matrix": matrix.tolist(),
        "matrix_scope": "preliminary fixed rigid transform; longitudinal row is the tested component",
        "anchor_definition": "Top PCA robust rear/head endpoints paired with Right segmented silhouette 99.5%/0.5% x quantiles",
        "validation_reference": "held-out within-cow rear-to-head proportional mapping; not physical calibration ground truth",
        "diagnostics": diagnostics,
        "validation_mean_absolute_boundary_error_m": float(np.mean(validation_endpoint_errors)),
        "validation_median_absolute_boundary_error_m": float(np.median(validation_endpoint_errors)),
        "validation_max_absolute_boundary_error_m": float(np.max(validation_endpoint_errors)),
        "validation_per_cow_mean_absolute_error_m": validation_errors.tolist(),
        "acceptance_rule": "preliminary pass only if validation mean <= 0.05 m and max <= 0.10 m",
        "preliminary_pass": bool(np.mean(validation_endpoint_errors) <= 0.05 and np.max(validation_endpoint_errors) <= 0.10),
        "limitations": [
            "No calibration target or author-supplied extrinsic matrix is available.",
            "The complete 4x4 matrix vertical/lateral components are provisional; only longitudinal boundary transfer is evaluated.",
            "The validation reference is an independent proportional silhouette estimate, not surveyed 3-D ground truth.",
        ],
        "records": results,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "top_to_right_matrix.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (OUTPUT / "boundary_validation_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    np.savetxt(OUTPUT / "top_to_right_matrix_4x4.txt", matrix, fmt="%.9f")
    print(json.dumps({key: summary[key] for key in [
        "training_cow_ids", "validation_cow_ids", "top_to_right_matrix",
        "validation_mean_absolute_boundary_error_m", "validation_median_absolute_boundary_error_m",
        "validation_max_absolute_boundary_error_m", "preliminary_pass"
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
