#!/usr/bin/env python3
"""Generate stable Topview back high-band centerlines for the accepted 61 cattle."""

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
MAIN_CSV = BASE / "long_axis_records.csv"
BOUNDARY_CSV = (
    BASE
    / "pca_horizontal_60pct_upper_halfwidth_preview_61"
    / "pca_upper_halfwidth_records_61.csv"
)
OUTPUT_DIR = BASE / "topview_back_high_band_centerline_61"
POINTS_CSV = OUTPUT_DIR / "back_high_band_centerline_points_61.csv"
SUMMARY_CSV = OUTPUT_DIR / "back_high_band_centerline_summary_61.csv"
METHOD_JSON = OUTPUT_DIR / "back_high_band_centerline_method.json"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
GROUP_DIR = OUTPUT_DIR / "groups_2cols_10cows"

LONGITUDINAL_STEP_M = 0.01
SLAB_HALF_WIDTH_M = 0.01
LATERAL_BIN_M = 0.01
HEIGHT_BAND_DROP_M = 0.01
LATERAL_SMOOTHING_M = 0.05
LONGITUDINAL_SMOOTHING_M = 0.09
METHOD_VERSION = "topview_back_high_band_centerline_v1.0"
RECORDED_ON = "2026-09-06"

PANEL_WIDTH = 820
PANEL_HEIGHT = 440
PLOT_LEFT = 28
PLOT_RIGHT = PANEL_WIDTH - 28
PLOT_TOP = 76
PLOT_BOTTOM = PANEL_HEIGHT - 48
BACKGROUND = (9, 14, 22)
TEXT = (232, 237, 245)
MUTED = (161, 172, 190)
AXIS = (255, 69, 205)
RIDGE = (52, 224, 255)
RIDGE_OUTLINE = (3, 7, 12)
CORE_MARKER = (245, 205, 65)
SEPARATOR = (35, 44, 58)

sys.path.insert(0, str(ROOT / "tools"))
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def load_font(size: int, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc"
        if bold
        else "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(24, True)
LABEL_FONT = load_font(17)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def height_color(normalized: float) -> tuple[int, int, int]:
    value = float(np.clip(normalized, 0, 1))
    stops = (
        (0.00, np.asarray([31, 147, 155], dtype=float)),
        (0.45, np.asarray([90, 190, 112], dtype=float)),
        (0.72, np.asarray([235, 216, 70], dtype=float)),
        (1.00, np.asarray([255, 154, 48], dtype=float)),
    )
    for (lv, lc), (rv, rc) in zip(stops[:-1], stops[1:]):
        if value <= rv:
            fraction = (value - lv) / (rv - lv)
            return tuple(np.rint(lc * (1 - fraction) + rc * fraction).astype(int))
    return tuple(stops[-1][1].astype(int))


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    return np.asarray(
        [
            np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)])
            for i in range(len(values))
        ]
    )


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def extract_profile(
    lateral: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    if len(lateral) < 80:
        return None
    low, high = np.quantile(lateral, [0.03, 0.97])
    edges = np.arange(
        math.floor(low / LATERAL_BIN_M) * LATERAL_BIN_M,
        math.ceil(high / LATERAL_BIN_M) * LATERAL_BIN_M + LATERAL_BIN_M * 0.5,
        LATERAL_BIN_M,
    )
    if len(edges) < 13:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.digitize(lateral, edges) - 1
    values = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        cell = height[indices == index]
        if len(cell) >= 2:
            values[index] = np.median(cell)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < max(12, 0.6 * len(values)):
        return None
    values = np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid])
    lateral_window = max(3, int(round(LATERAL_SMOOTHING_M / LATERAL_BIN_M)))
    if lateral_window % 2 == 0:
        lateral_window += 1
    values = smooth_profile(values, lateral_window)

    peak_index = int(np.argmax(values))
    peak_height = float(values[peak_index])
    in_high_band = values >= peak_height - HEIGHT_BAND_DROP_M
    left = peak_index
    right = peak_index
    while left > 0 and in_high_band[left - 1]:
        left -= 1
    while right + 1 < len(values) and in_high_band[right + 1]:
        right += 1
    band_center = float((centers[left] + centers[right]) / 2)
    band_width = float((right - left + 1) * LATERAL_BIN_M)
    return centers, values, band_center, band_width, peak_height


def calculate_centerline(row: dict[str, str]) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    cow_id = int(row["cow_id"])
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.asarray(
        [
            float(row["pca_full_torso_axis_centroid_x_m"]),
            float(row["pca_full_torso_axis_centroid_y_m"]),
        ]
    )
    axis = np.asarray(
        [float(row["pca_full_torso_axis_unit_x"]), float(row["pca_full_torso_axis_unit_y"])]
    )
    axis /= np.linalg.norm(axis)
    normal = np.asarray([-axis[1], axis[0]])
    relative = points_xy - centroid
    longitudinal = relative @ axis
    lateral = relative @ normal
    ground_z = (
        float(row["ground_plane_a"]) * vertices["x"]
        + float(row["ground_plane_b"]) * vertices["y"]
        + float(row["ground_plane_c_m"])
    )
    height = ground_z - vertices["z"]

    start = float(row["pca_full_torso_rear_boundary_axis_m"])
    end = float(row["pca_full_torso_head_boundary_axis_m"])
    stations = np.arange(start, end + LONGITUDINAL_STEP_M * 0.25, LONGITUDINAL_STEP_M)
    raw_centers = np.full(len(stations), np.nan)
    band_widths = np.full(len(stations), np.nan)
    peak_heights = np.full(len(stations), np.nan)
    profiles: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(stations)

    for index, station in enumerate(stations):
        selected = np.abs(longitudinal - station) < SLAB_HALF_WIDTH_M
        result = extract_profile(lateral[selected], height[selected])
        if result is None:
            continue
        centers, values, band_center, band_width, peak_height = result
        profiles[index] = (centers, values)
        raw_centers[index] = band_center
        band_widths[index] = band_width
        peak_heights[index] = peak_height

    valid = np.isfinite(raw_centers)
    if np.count_nonzero(valid) < 0.8 * len(stations):
        raise ValueError(f"cow {cow_id}: insufficient valid ridge stations")
    raw_centers = np.interp(np.arange(len(stations)), np.flatnonzero(valid), raw_centers[valid])
    smooth_window = max(3, int(round(LONGITUDINAL_SMOOTHING_M / LONGITUDINAL_STEP_M)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed_centers = rolling_median(raw_centers, smooth_window)
    smoothed_centers = smooth_profile(smoothed_centers, 3)

    surface_heights = np.full(len(stations), np.nan)
    for index, profile in enumerate(profiles):
        if profile is not None:
            centers, values = profile
            surface_heights[index] = np.interp(smoothed_centers[index], centers, values)
    height_valid = np.isfinite(surface_heights)
    surface_heights = np.interp(
        np.arange(len(stations)), np.flatnonzero(height_valid), surface_heights[height_valid]
    )

    xy = centroid + stations[:, None] * axis + smoothed_centers[:, None] * normal
    surface_z = (
        float(row["ground_plane_a"]) * xy[:, 0]
        + float(row["ground_plane_b"]) * xy[:, 1]
        + float(row["ground_plane_c_m"])
        - surface_heights
    )
    rear_xy = np.asarray([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    head_xy = np.asarray([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    full_length = float(np.linalg.norm(head_xy - rear_xy))
    from_rear = (xy - rear_xy) @ axis

    point_rows: list[dict[str, object]] = []
    for index in range(len(stations)):
        point_rows.append(
            {
                "cow_id": cow_id,
                "point_index": index,
                "station_axis_from_centroid_m": round(float(stations[index]), 6),
                "station_distance_from_rear_m": round(float(from_rear[index]), 6),
                "station_ratio_from_rear": round(float(from_rear[index] / full_length), 8),
                "raw_high_band_center_lateral_m": round(float(raw_centers[index]), 6),
                "smoothed_high_band_center_lateral_m": round(float(smoothed_centers[index]), 6),
                "center_x_m": round(float(xy[index, 0]), 6),
                "center_y_m": round(float(xy[index, 1]), 6),
                "surface_z_m": round(float(surface_z[index]), 6),
                "height_above_ground_m": round(float(surface_heights[index]), 6),
                "peak_height_above_ground_m": ""
                if not np.isfinite(peak_heights[index])
                else round(float(peak_heights[index]), 6),
                "high_band_width_m": ""
                if not np.isfinite(band_widths[index])
                else round(float(band_widths[index]), 6),
                "profile_status": "measured" if valid[index] else "interpolated",
                "method_version": METHOD_VERSION,
            }
        )

    line_steps = np.sqrt(np.sum(np.diff(np.column_stack([xy, surface_z]), axis=0) ** 2, axis=1))
    summary = {
        "cow_id": cow_id,
        "source_top_ply": row["extracted_ply"],
        "centerline_point_count": len(stations),
        "measured_point_count": int(np.count_nonzero(valid)),
        "interpolated_point_count": int(np.count_nonzero(~valid)),
        "measured_fraction": round(float(np.mean(valid)), 6),
        "centerline_start_ratio_from_rear": round(float(from_rear[0] / full_length), 6),
        "centerline_end_ratio_from_rear": round(float(from_rear[-1] / full_length), 6),
        "centerline_3d_length_m": round(float(np.sum(line_steps)), 6),
        "median_high_band_width_m": round(float(np.nanmedian(band_widths)), 6),
        "maximum_high_band_width_m": round(float(np.nanmax(band_widths)), 6),
        "median_absolute_lateral_offset_from_pca_m": round(
            float(np.median(np.abs(smoothed_centers))), 6
        ),
        "maximum_absolute_lateral_offset_from_pca_m": round(
            float(np.max(np.abs(smoothed_centers))), 6
        ),
        "status": "ok",
        "method_version": METHOD_VERSION,
        "recorded_on": RECORDED_ON,
    }
    render_data = {
        "longitudinal": longitudinal,
        "lateral": lateral,
        "height": height,
        "rear_xy": rear_xy,
        "head_xy": head_xy,
        "axis": axis,
        "ridge_xy": xy,
        "ridge_from_rear": from_rear,
        "ridge_lateral": smoothed_centers,
    }
    return point_rows, summary, render_data


def make_panel(row: dict[str, str], data: dict[str, np.ndarray], summary: dict[str, object]) -> Image.Image:
    longitudinal = data["longitudinal"]
    lateral = data["lateral"]
    height = data["height"]
    ridge_t = data["ridge_from_rear"]
    ridge_q = data["ridge_lateral"]
    rear_xy = data["rear_xy"]
    head_xy = data["head_xy"]
    full_length = float(np.linalg.norm(head_xy - rear_xy))
    # Convert point longitudinal coordinates from centroid-based to rear-based.
    points_xy = rear_xy + np.outer(np.zeros(len(longitudinal)), data["axis"])
    del points_xy
    centroid = np.asarray(
        [float(row["pca_full_torso_axis_centroid_x_m"]), float(row["pca_full_torso_axis_centroid_y_m"])]
    )
    rear_offset = float((centroid - rear_xy) @ data["axis"])
    t = longitudinal + rear_offset

    t_min, t_max = np.quantile(t, [0.001, 0.999])
    q_min, q_max = np.quantile(lateral, [0.001, 0.999])
    t_min = min(float(t_min), 0.0) - 0.04
    t_max = max(float(t_max), full_length) + 0.04
    q_min, q_max = float(q_min) - 0.04, float(q_max) + 0.04
    scale = min(
        (PLOT_RIGHT - PLOT_LEFT) / max(t_max - t_min, 1e-6),
        (PLOT_BOTTOM - PLOT_TOP) / max(q_max - q_min, 1e-6),
    )
    center_t = (t_min + t_max) / 2
    center_q = (q_min + q_max) / 2
    center_x = (PLOT_LEFT + PLOT_RIGHT) / 2
    center_y = (PLOT_TOP + PLOT_BOTTOM) / 2

    def px(value):
        return center_x + (np.asarray(value) - center_t) * scale

    def py(value):
        return center_y - (np.asarray(value) - center_q) * scale

    image = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    pixels = np.asarray(image).copy()
    h_low, h_high = np.quantile(height, [0.03, 0.97])
    normalized = np.clip((height - h_low) / max(float(h_high - h_low), 1e-6), 0, 1)
    xp = np.rint(px(t)).astype(int)
    yp = np.rint(py(lateral)).astype(int)
    for index in np.argsort(normalized):
        x, y = xp[index], yp[index]
        if PLOT_LEFT <= x <= PLOT_RIGHT and PLOT_TOP <= y <= PLOT_BOTTOM:
            color = height_color(float(normalized[index]))
            pixels[max(PLOT_TOP, y - 1) : min(PLOT_BOTTOM + 1, y + 2), max(PLOT_LEFT, x - 1) : min(PLOT_RIGHT + 1, x + 2)] = color
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    axis_y = float(py(0.0))
    draw.line((float(px(0)), axis_y, float(px(full_length)), axis_y), fill=RIDGE_OUTLINE, width=6)
    draw.line((float(px(0)), axis_y, float(px(full_length)), axis_y), fill=AXIS, width=2)
    ridge_pixels = list(zip(px(ridge_t).tolist(), py(ridge_q).tolist()))
    draw.line(ridge_pixels, fill=RIDGE_OUTLINE, width=8, joint="curve")
    draw.line(ridge_pixels, fill=RIDGE, width=4, joint="curve")
    for endpoint in (ridge_pixels[0], ridge_pixels[-1]):
        draw.ellipse((endpoint[0]-5, endpoint[1]-5, endpoint[0]+5, endpoint[1]+5), fill=CORE_MARKER, outline=RIDGE_OUTLINE, width=2)
    cow_id = int(row["cow_id"])
    draw.text((22, 16), f"牛 {cow_id:03d}  牛背高位带中心线", font=TITLE_FONT, fill=TEXT)
    draw.text((22, 49), "青线=高位带中心  粉线=PCA方向参考", font=LABEL_FONT, fill=MUTED)
    draw.text((PLOT_LEFT, PANEL_HEIGHT - 22), "臀部", font=LABEL_FONT, fill=RIDGE, anchor="ls")
    draw.text((PLOT_RIGHT, PANEL_HEIGHT - 22), "头部", font=LABEL_FONT, fill=(255, 161, 48), anchor="rs")
    draw.text(
        (PANEL_WIDTH / 2, PANEL_HEIGHT - 22),
        f"高位带中位宽度 {float(summary['median_high_band_width_m']) * 100:.1f} cm",
        font=LABEL_FONT,
        fill=MUTED,
        anchor="ms",
    )
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    main_by_cow = {int(row["cow_id"]): row for row in read_csv(MAIN_CSV)}
    cow_ids = sorted(int(row["cow_id"]) for row in read_csv(BOUNDARY_CSV))
    if len(cow_ids) != 61:
        raise ValueError(f"expected 61 cattle, found {len(cow_ids)}")

    all_points: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    panels: list[tuple[int, Image.Image]] = []
    for cow_id in cow_ids:
        row = main_by_cow[cow_id]
        points, summary, render_data = calculate_centerline(row)
        panel = make_panel(row, render_data, summary)
        panel.save(INDIVIDUAL_DIR / f"cow_{cow_id:03d}_back_high_band_centerline.png", optimize=True)
        all_points.extend(points)
        summaries.append(summary)
        panels.append((cow_id, panel))

    write_csv(POINTS_CSV, all_points)
    write_csv(SUMMARY_CSV, summaries)
    for group_number, start in enumerate(range(0, len(panels), 10), start=1):
        group = panels[start : start + 10]
        rows = math.ceil(len(group) / 2)
        canvas = Image.new("RGB", (PANEL_WIDTH * 2, PANEL_HEIGHT * rows), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        for index, (_, panel) in enumerate(group):
            x = (index % 2) * PANEL_WIDTH
            y = (index // 2) * PANEL_HEIGHT
            canvas.paste(panel, (x, y))
            if index % 2 == 0:
                draw.line((PANEL_WIDTH - 1, y + 8, PANEL_WIDTH - 1, y + PANEL_HEIGHT - 8), fill=SEPARATOR, width=2)
        first_id, last_id = group[0][0], group[-1][0]
        canvas.save(
            GROUP_DIR / f"back_high_band_centerline_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.png",
            optimize=True,
        )

    METHOD_JSON.write_text(
        json.dumps(
            {
                "method_version": METHOD_VERSION,
                "cattle_count": len(cow_ids),
                "scope": "existing PCA 45/48 core-torso longitudinal interval",
                "longitudinal_step_m": LONGITUDINAL_STEP_M,
                "cross_section_slab_total_width_m": SLAB_HALF_WIDTH_M * 2,
                "lateral_bin_m": LATERAL_BIN_M,
                "height_band_definition": "contiguous lateral profile within 0.01 m below the smoothed maximum",
                "height_band_center": "midpoint of the contiguous high-band segment containing the maximum",
                "lateral_profile_smoothing_m": LATERAL_SMOOTHING_M,
                "longitudinal_centerline_smoothing_m": LONGITUDINAL_SMOOTHING_M,
                "line_interpretation": "computed center of the Topview high dorsal band; not an anatomical spine annotation",
                "points_file": str(POINTS_CSV.relative_to(ROOT)),
                "summary_file": str(SUMMARY_CSV.relative_to(ROOT)),
                "recorded_on": RECORDED_ON,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"cattle={len(cow_ids)} points={len(all_points)}")
    print(POINTS_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
