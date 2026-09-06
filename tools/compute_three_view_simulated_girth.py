#!/usr/bin/env python3
"""Compute a three-view simulated girth from Top, Right, and Left PLY files."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
DATASET = ROOT / "dataset"
MAIN_CSV = BASE / "long_axis_records.csv"
VALLEY_CSV = (
    BASE
    / "pca_horizontal_60pct_upper_halfwidth_preview_61"
    / "pca_upper_halfwidth_records_61.csv"
)
TOP_ARC_CSV = (
    BASE
    / "dual_valley_topview_dorsal_arcs_61"
    / "dual_valley_dorsal_arc_records_61.csv"
)
OUTPUT_DIR = BASE / "three_view_simulated_girth_61"
RECORDS_CSV = OUTPUT_DIR / "three_view_simulated_girth_records_61.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "three_view_simulated_girth_correlations_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "three_view_simulated_girth_summary.json"
QA_DIR = OUTPUT_DIR / "qa_cross_sections"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 424
LONGITUDINAL_OFFSETS_M = np.arange(-0.05, 0.0501, 0.01)
SLAB_HALF_WIDTH_M = 0.01
VERTICAL_BIN_M = 0.01
VERTICAL_SMOOTHING_M = 0.03
METHOD_VERSION = "three_view_top_dorsal_plus_right_left_lower_arc_v2.0"
RECORDED_ON = "2026-09-06"

sys.path.insert(0, str(ROOT / "tools"))
from compute_rightview_median_body_depth import (  # noqa: E402
    fit_ground_plane,
    profile_bounds,
    read_ply_xyz,
    robust_line_fit,
    rolling_quantile,
    segment_cow,
)


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


def prepare_side(path: Path, cow_id: int, view: str) -> dict[str, object]:
    xyz_grid = read_ply_xyz(path).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    rng = np.random.default_rng(20260906 + cow_id + (0 if view == "right" else 10000))
    ground_coef, ground_fraction = fit_ground_plane(xyz_grid, rng)
    cow_mask, height_grid = segment_cow(xyz_grid, ground_coef)
    points = xyz_grid[cow_mask].astype(float)
    x = points[:, 0]
    h = height_grid[cow_mask].astype(float)

    ground_normal = np.asarray([-ground_coef[0], 1.0, -ground_coef[1]])
    ground_normal /= np.linalg.norm(ground_normal)
    origin = np.median(points, axis=0)
    relative = points - origin
    ground_projection = relative - np.outer(relative @ ground_normal, ground_normal)
    covariance = np.cov(ground_projection, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if long_axis[0] < 0:
        long_axis = -long_axis
    long_axis -= float(long_axis @ ground_normal) * ground_normal
    long_axis /= np.linalg.norm(long_axis)
    lateral_axis = np.cross(ground_normal, long_axis)
    lateral_axis /= np.linalg.norm(lateral_axis)
    u = relative @ long_axis
    v = h
    w = relative @ lateral_axis
    camera_direction = -origin
    outward_sign = 1.0 if float(camera_direction @ lateral_axis) >= 0 else -1.0

    centers, raw_back, raw_belly, counts = profile_bounds(u, v)
    back = rolling_quantile(raw_back, radius=3, quantile=0.5)
    belly = rolling_quantile(raw_belly, radius=20, quantile=0.90)
    belly = rolling_quantile(belly, radius=3, quantile=0.5)
    body_low, body_high = np.quantile(u, [0.005, 0.995])
    if view == "right":
        rear_u, head_u = float(body_high), float(body_low)
    else:
        rear_u, head_u = float(body_low), float(body_high)
    return {
        "path": path,
        "u": u,
        "v": v,
        "w": w,
        "outward_sign": outward_sign,
        "centers": centers,
        "back": back,
        "belly": belly,
        "counts": counts,
        "rear_u": rear_u,
        "head_u": head_u,
        "ground_fraction": ground_fraction,
        "long_axis_x_component": float(long_axis[0]),
    }


def vertical_surface_profile(
    side: dict[str, object], station_u: float
) -> tuple[np.ndarray, np.ndarray, float, float, float, int] | None:
    u = np.asarray(side["u"])
    v = np.asarray(side["v"])
    w = np.asarray(side["w"])
    outward_sign = float(side["outward_sign"])
    centers = np.asarray(side["centers"])
    back_curve = np.asarray(side["back"])
    belly_curve = np.asarray(side["belly"])
    selected = np.abs(u - station_u) < SLAB_HALF_WIDTH_M
    if np.count_nonzero(selected) < 80:
        return None
    back = float(np.interp(station_u, centers, back_curve))
    belly = float(np.interp(station_u, centers, belly_curve))
    if not np.isfinite(back) or not np.isfinite(belly) or back - belly < 0.30:
        return None
    selected &= (v >= belly) & (v <= back)
    vv = v[selected]
    ww = w[selected]
    if len(vv) < 60:
        return None

    edges = np.arange(
        math.floor(belly / VERTICAL_BIN_M) * VERTICAL_BIN_M,
        math.ceil(back / VERTICAL_BIN_M) * VERTICAL_BIN_M + VERTICAL_BIN_M * 0.5,
        VERTICAL_BIN_M,
    )
    heights = (edges[:-1] + edges[1:]) / 2
    indices = np.digitize(vv, edges) - 1
    depths = np.full(len(heights), np.nan)
    counts = np.zeros(len(heights), dtype=int)
    for index in range(len(heights)):
        cell = ww[indices == index]
        counts[index] = len(cell)
        if len(cell) >= 2:
            depths[index] = np.median(cell)
    valid = np.isfinite(depths)
    if np.count_nonzero(valid) < max(18, 0.55 * len(depths)):
        return None
    first, last = np.flatnonzero(valid)[[0, -1]]
    heights = heights[first : last + 1]
    depths = depths[first : last + 1]
    counts = counts[first : last + 1]
    valid = np.isfinite(depths)
    depths[~valid] = np.interp(
        np.flatnonzero(~valid), np.flatnonzero(valid), depths[valid]
    )
    width = max(3, int(round(VERTICAL_SMOOTHING_M / VERTICAL_BIN_M)))
    if width % 2 == 0:
        width += 1
    depths = smooth(depths, width)
    body_depth = float(heights[-1] - heights[0])
    # For an approximately elliptical section, the lateral widest point is at
    # half the back-to-belly height. This fixed geometric landmark is much more
    # reproducible across opposing cameras than the raw depth extremum, which
    # shifts with camera angle and partial surface visibility.
    midpoint_height = heights[0] + 0.50 * body_depth
    outward_index = int(np.argmin(np.abs(heights - midpoint_height)))
    if outward_index < 4:
        return None
    lower_h = heights[: outward_index + 1]
    lower_z = depths[: outward_index + 1]
    arc = float(np.sum(np.sqrt(np.diff(lower_h) ** 2 + np.diff(lower_z) ** 2)))
    full_arc = float(np.sum(np.sqrt(np.diff(heights) ** 2 + np.diff(depths) ** 2)))
    coverage = float(np.mean(counts > 0))
    return heights, depths, arc, full_arc, coverage, outward_index


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
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        predictions[index] = slope * x[index] + intercept
    residual = predictions - y
    return (
        float(np.corrcoef(predictions, y)[0, 1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(np.abs(residual))),
    )


def metric_summary(
    field: str, label: str, values: np.ndarray, outcome: np.ndarray, outcome_field: str
) -> dict[str, object]:
    pearson = float(np.corrcoef(values, outcome)[0, 1])
    spearman = float(
        np.corrcoef(average_ranks(values), average_ranks(outcome))[0, 1]
    )
    loo_r, loo_rmse, loo_mae = loocv(values, outcome)
    return {
        "metric_field": field,
        "metric_label_zh": label,
        "outcome_field": outcome_field,
        "n": len(values),
        "pearson_r": round(pearson, 6),
        "spearman_rho": round(spearman, 6),
        "r_squared": round(pearson**2, 6),
        "loocv_r": round(loo_r, 6),
        "loocv_rmse": round(loo_rmse, 6),
        "loocv_mae": round(loo_mae, 6),
        "method_version": METHOD_VERSION,
    }


def load_font(size: int):
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


QA_FONT = load_font(17)


def render_cross_section_qa(
    cow_id: int,
    right_profile: tuple[np.ndarray, np.ndarray, float, float, float, int],
    left_profile: tuple[np.ndarray, np.ndarray, float, float, float, int],
    output: Path,
) -> None:
    canvas = Image.new("RGB", (1000, 500), (8, 13, 21))
    draw = ImageDraw.Draw(canvas)
    colors = ((72, 211, 255), (255, 173, 68))
    for panel, (view, profile, color) in enumerate(
        (("Right", right_profile, colors[0]), ("Left", left_profile, colors[1]))
    ):
        heights, depths, arc, _, coverage, outward = profile
        x0 = 55 + panel * 500
        x1 = 445 + panel * 500
        y0, y1 = 65, 440
        d0, d1 = float(np.min(depths)), float(np.max(depths))
        h0, h1 = float(np.min(heights)), float(np.max(heights))

        def px(value):
            return x0 + (np.asarray(value) - d0) / max(d1 - d0, 1e-6) * (x1 - x0)

        def py(value):
            return y1 - (np.asarray(value) - h0) / max(h1 - h0, 1e-6) * (y1 - y0)

        points = list(zip(px(depths).tolist(), py(heights).tolist()))
        draw.line(points, fill=(116, 128, 148), width=3)
        lower_points = points[: outward + 1]
        draw.line(lower_points, fill=color, width=6)
        ox, oy = points[outward]
        draw.ellipse((ox - 6, oy - 6, ox + 6, oy + 6), fill=color)
        draw.text((x0, 20), f"牛 {cow_id:03d}  {view}", font=QA_FONT, fill=(235, 240, 247))
        draw.text(
            (x0, 455),
            f"彩色段={arc * 100:.1f} cm  覆盖={coverage:.0%}",
            font=QA_FONT,
            fill=color,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    valleys = sorted(read_csv(VALLEY_CSV), key=lambda row: int(row["cow_id"]))
    top_arcs = {int(row["cow_id"]): row for row in read_csv(TOP_ARC_CSV)}
    if len(valleys) != 61:
        raise ValueError(f"expected 61 cattle, found {len(valleys)}")

    records: list[dict[str, object]] = []
    for position, valley in enumerate(valleys, start=1):
        cow_id = int(valley["cow_id"])
        row = main_rows[cow_id]
        upper_ratio = float(valley["pca_upper_width_valley_station_ratio_full_axis"])
        lower_ratio = float(valley["pca_lower_width_valley_station_ratio_full_axis"])
        chest_ratio = (upper_ratio + lower_ratio) / 2
        right_paths = list((DATASET / str(cow_id)).glob("right-*.ply"))
        left_paths = list((DATASET / str(cow_id)).glob("left-*.ply"))
        if len(right_paths) != 1 or len(left_paths) != 1:
            raise ValueError(f"cow {cow_id}: missing or duplicate side PLY")
        right = prepare_side(right_paths[0], cow_id, "right")
        left = prepare_side(left_paths[0], cow_id, "left")

        side_results = {}
        central_profiles = {}
        for view, side in (("right", right), ("left", left)):
            rear_u = float(side["rear_u"])
            head_u = float(side["head_u"])
            center_u = rear_u + chest_ratio * (head_u - rear_u)
            lower_arcs = []
            full_arcs = []
            coverages = []
            profiles = []
            for offset in LONGITUDINAL_OFFSETS_M:
                profile = vertical_surface_profile(side, center_u + float(offset))
                if profile is None:
                    continue
                profiles.append((float(offset), profile))
                lower_arcs.append(profile[2])
                full_arcs.append(profile[3])
                coverages.append(profile[4])
            if len(lower_arcs) < 7:
                raise ValueError(
                    f"cow {cow_id} {view}: only {len(lower_arcs)} valid side profiles"
                )
            central = min(profiles, key=lambda item: abs(item[0]))[1]
            central_profiles[view] = central
            side_results[view] = {
                "center_u": center_u,
                "lower_arc": float(np.median(lower_arcs)),
                "full_arc": float(np.median(full_arcs)),
                "coverage": float(np.median(coverages)),
                "valid_count": len(lower_arcs),
            }

        top_arc = float(top_arcs[cow_id]["method2_midpoint_full_arc_m"])
        simulated = (
            top_arc
            + float(side_results["right"]["lower_arc"])
            + float(side_results["left"]["lower_arc"])
        )
        side_full = float(side_results["right"]["full_arc"]) + float(
            side_results["left"]["full_arc"]
        )
        records.append(
            {
                "cow_id": cow_id,
                "ground_truth_weight_kg": float(row["ground_truth_weight_kg"]),
                "ground_truth_heart_girth_cm": float(row["ground_truth_heart_girth_cm"]),
                "upper_valley_ratio_from_rear": upper_ratio,
                "lower_valley_ratio_from_rear": lower_ratio,
                "virtual_chest_ratio_from_rear": chest_ratio,
                "topview_dorsal_arc_m": top_arc,
                "rightview_lower_side_arc_m": side_results["right"]["lower_arc"],
                "leftview_lower_side_arc_m": side_results["left"]["lower_arc"],
                "three_view_simulated_girth_m": simulated,
                "rightview_full_visible_half_arc_m": side_results["right"]["full_arc"],
                "leftview_full_visible_half_arc_m": side_results["left"]["full_arc"],
                "right_plus_left_full_visible_arc_m": side_full,
                "rightview_chest_station_u_m": side_results["right"]["center_u"],
                "leftview_chest_station_u_m": side_results["left"]["center_u"],
                "rightview_valid_profile_count": side_results["right"]["valid_count"],
                "leftview_valid_profile_count": side_results["left"]["valid_count"],
                "rightview_median_profile_coverage": side_results["right"]["coverage"],
                "leftview_median_profile_coverage": side_results["left"]["coverage"],
                "rightview_ground_fit_inlier_fraction": right["ground_fraction"],
                "leftview_ground_fit_inlier_fraction": left["ground_fraction"],
                "rightview_source_ply": str(right_paths[0].relative_to(ROOT)),
                "leftview_source_ply": str(left_paths[0].relative_to(ROOT)),
                "simulated_girth_formula": "top dorsal arc + right lower-side arc + left lower-side arc",
                "side_body_correction": "rolling 90th percentile belly boundary removes legs and udder excursions; lower arc starts at 50% back-to-belly height",
                "method_version": METHOD_VERSION,
                "status": "ok_experimental_proxy_not_anatomical_girth",
            }
        )
        render_cross_section_qa(
            cow_id,
            central_profiles["right"],
            central_profiles["left"],
            QA_DIR / f"cow_{cow_id:03d}_side_arc_cross_sections.png",
        )
        print(
            f"[{position:02d}/61] cow {cow_id:03d}: simulated={simulated * 100:.1f} cm",
            flush=True,
        )

    write_csv(RECORDS_CSV, records)
    weights = np.asarray([float(row["ground_truth_weight_kg"]) for row in records])
    girths = np.asarray([float(row["ground_truth_heart_girth_cm"]) for row in records])
    metric_specs = (
        ("three_view_simulated_girth_m", "三视角模拟胸围"),
        ("right_plus_left_full_visible_arc_m", "左右侧完整可见半弧之和"),
        ("topview_dorsal_arc_m", "Topview胸区上背弧"),
        ("ground_truth_heart_girth_cm", "真实胸围"),
    )
    metrics: list[dict[str, object]] = []
    for field, label in metric_specs:
        values = np.asarray([float(row[field]) for row in records])
        metrics.append(metric_summary(field, label, values, weights, "ground_truth_weight_kg"))
        if field != "ground_truth_heart_girth_cm":
            metrics.append(
                metric_summary(
                    field, label, values, girths, "ground_truth_heart_girth_cm"
                )
            )
    write_csv(CORRELATIONS_CSV, metrics)

    simulated_values = np.asarray(
        [float(row["three_view_simulated_girth_m"]) for row in records]
    )
    right_arcs = np.asarray(
        [float(row["rightview_lower_side_arc_m"]) for row in records]
    )
    left_arcs = np.asarray(
        [float(row["leftview_lower_side_arc_m"]) for row in records]
    )
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "simulated_girth_mean_cm": float(np.mean(simulated_values) * 100),
                "simulated_girth_minimum_cm": float(np.min(simulated_values) * 100),
                "simulated_girth_maximum_cm": float(np.max(simulated_values) * 100),
                "right_left_lower_arc_pearson_r": float(np.corrcoef(right_arcs, left_arcs)[0, 1]),
                "mean_absolute_right_left_lower_arc_difference_cm": float(
                    np.mean(np.abs(right_arcs - left_arcs)) * 100
                ),
                "method_definition": "Top midpoint dorsal arc plus independently measured right and left lower visible surface arcs at the same normalized body station",
                "calibration_limit": "views are matched by normalized rear-to-head position; no cross-camera extrinsic matrix is available",
                "correlations": metrics,
                "records_file": str(RECORDS_CSV.relative_to(ROOT)),
                "correlations_file": str(CORRELATIONS_CSV.relative_to(ROOT)),
                "qa_directory": str(QA_DIR.relative_to(ROOT)),
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
