#!/usr/bin/env python3
"""Create horizontal PCA-axis previews with a 60%-from-rear marker for 61 cattle."""

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
RECORDS_CSV = BASE / "long_axis_records.csv"
OUTPUT_DIR = BASE / "pca_horizontal_60pct_preview_61"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
GROUP_DIR = OUTPUT_DIR / "groups_2cols_10cows"
POSITION_CSV = OUTPUT_DIR / "pca_60pct_positions_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "pca_60pct_preview_summary.json"

PANEL_WIDTH = 820
PANEL_HEIGHT = 440
HEADER_HEIGHT = 64
PLOT_LEFT = 28
PLOT_RIGHT = PANEL_WIDTH - 28
PLOT_TOP = 78
PLOT_BOTTOM = PANEL_HEIGHT - 45
GROUP_SIZE = 10
GROUP_COLUMNS = 2

BACKGROUND = (9, 14, 22)
TEXT = (232, 237, 245)
MUTED = (161, 172, 190)
AXIS_OUTLINE = (3, 7, 12)
AXIS = (255, 69, 205)
REAR_POINT = (42, 215, 238)
HEAD_POINT = (255, 161, 48)
MARKER = (245, 247, 250)
SEPARATOR = (35, 44, 58)

sys.path.insert(0, str(ROOT / "tools"))
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def load_font(size: int, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(25, True)
LABEL_FONT = load_font(18)
SMALL_FONT = load_font(16)


def read_rows() -> list[dict[str, str]]:
    with RECORDS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    strict = [
        row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible", "").upper() == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(strict) != 61:
        raise ValueError(f"expected 61 strict cattle, found {len(strict)}")
    return sorted(strict, key=lambda row: int(row["cow_id"]))


def height_color(normalized: float) -> tuple[int, int, int]:
    value = float(np.clip(normalized, 0, 1))
    stops = (
        (0.00, np.asarray([31, 147, 155], dtype=float)),
        (0.45, np.asarray([90, 190, 112], dtype=float)),
        (0.72, np.asarray([235, 216, 70], dtype=float)),
        (1.00, np.asarray([255, 154, 48], dtype=float)),
    )
    for (left_value, left_color), (right_value, right_color) in zip(stops[:-1], stops[1:]):
        if value <= right_value:
            fraction = (value - left_value) / (right_value - left_value)
            color = left_color * (1 - fraction) + right_color * fraction
            return tuple(np.rint(color).astype(int))
    return tuple(stops[-1][1].astype(int))


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float) -> None:
    start = min(y0, y1)
    end = max(y0, y1)
    for y in np.arange(start, end, 18):
        draw.line((x, y, x, min(end, y + 10)), fill=MARKER, width=4)


def make_panel(
    row: dict[str, str], marker_ratios: tuple[float, ...] = (0.60,)
) -> tuple[Image.Image, dict[str, object]]:
    cow_id = int(row["cow_id"])
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    rear_xy = np.asarray([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    head_xy = np.asarray([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    axis = head_xy - rear_xy
    full_length = float(np.linalg.norm(axis))
    axis /= full_length
    normal = np.asarray([-axis[1], axis[0]])
    relative_to_rear = points_xy - rear_xy
    longitudinal = relative_to_rear @ axis
    lateral = relative_to_rear @ normal

    marker_distances = [ratio * full_length for ratio in marker_ratios]
    plane_z = (
        float(row["ground_plane_a"]) * vertices["x"]
        + float(row["ground_plane_b"]) * vertices["y"]
        + float(row["ground_plane_c_m"])
    )
    heights = plane_z - vertices["z"]
    height_low, height_high = np.quantile(heights, [0.03, 0.97])
    height_span = max(float(height_high - height_low), 1e-6)

    t_min, t_max = np.quantile(longitudinal, [0.001, 0.999])
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
    center_px_x = (PLOT_LEFT + PLOT_RIGHT) / 2
    center_px_y = (PLOT_TOP + PLOT_BOTTOM) / 2

    def px(t: float) -> float:
        return center_px_x + (t - center_t) * scale

    def py(q: float) -> float:
        return center_px_y - (q - center_q) * scale

    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    pixels = np.asarray(panel).copy()
    x_pixels = np.rint(center_px_x + (longitudinal - center_t) * scale).astype(int)
    y_pixels = np.rint(center_px_y - (lateral - center_q) * scale).astype(int)
    normalized_heights = np.clip((heights - height_low) / height_span, 0, 1)
    order = np.argsort(normalized_heights)
    for index in order:
        x, y = x_pixels[index], y_pixels[index]
        if PLOT_LEFT <= x <= PLOT_RIGHT and PLOT_TOP <= y <= PLOT_BOTTOM:
            color = height_color(float(normalized_heights[index]))
            pixels[max(PLOT_TOP, y - 1) : min(PLOT_BOTTOM + 1, y + 2), max(PLOT_LEFT, x - 1) : min(PLOT_RIGHT + 1, x + 2)] = color
    panel = Image.fromarray(pixels)
    draw = ImageDraw.Draw(panel)

    if len(marker_ratios) == 1:
        ratio = marker_ratios[0]
        title = (
            f"牛 {cow_id:03d}   PCA全长 {full_length:.3f} m   "
            f"{ratio * 100:.0f}%位置距臀 {ratio * full_length:.3f} m"
        )
    else:
        marker_text = " / ".join(
            f"{ratio * 100:.0f}%={ratio * full_length:.3f} m" for ratio in marker_ratios
        )
        title = f"牛 {cow_id:03d}   PCA全长 {full_length:.3f} m   {marker_text}"
    draw.text((22, 16), title, font=TITLE_FONT, fill=TEXT)
    axis_y = py(0.0)
    draw.line((px(0.0), axis_y, px(full_length), axis_y), fill=AXIS_OUTLINE, width=8)
    draw.line((px(0.0), axis_y, px(full_length), axis_y), fill=AXIS, width=4)
    endpoint_radius = 8
    draw.ellipse(
        (px(0.0) - endpoint_radius, axis_y - endpoint_radius, px(0.0) + endpoint_radius, axis_y + endpoint_radius),
        fill=REAR_POINT,
        outline=AXIS_OUTLINE,
        width=3,
    )
    draw.ellipse(
        (px(full_length) - endpoint_radius, axis_y - endpoint_radius, px(full_length) + endpoint_radius, axis_y + endpoint_radius),
        fill=HEAD_POINT,
        outline=AXIS_OUTLINE,
        width=3,
    )

    marker_colors = ((245, 247, 250), (255, 205, 65), (84, 210, 255))
    for marker_index, (ratio, marker_distance) in enumerate(zip(marker_ratios, marker_distances)):
        marker_x = px(marker_distance)
        nearby = np.abs(longitudinal - marker_distance) <= 0.025
        if np.count_nonzero(nearby) >= 5:
            local_low, local_high = np.quantile(lateral[nearby], [0.01, 0.99])
        else:
            local_low, local_high = q_min, q_max
        marker_top = py(float(local_high) + 0.055)
        marker_bottom = py(float(local_low) - 0.055)
        if abs(ratio - 0.70) < 1e-9:
            marker_color = (255, 205, 65)
        elif abs(ratio - 0.50) < 1e-9:
            marker_color = (245, 247, 250)
        else:
            marker_color = marker_colors[marker_index % len(marker_colors)]
        start = min(marker_top, marker_bottom)
        end = max(marker_top, marker_bottom)
        for y in np.arange(start, end, 18):
            draw.line(
                (marker_x, y, marker_x, min(end, y + 10)),
                fill=marker_color,
                width=4,
            )
        draw.text(
            (marker_x, max(HEADER_HEIGHT + 2, marker_top - 7)),
            f"{ratio * 100:.0f}%",
            font=LABEL_FONT,
            fill=marker_color,
            anchor="ms",
        )
    draw.text((PLOT_LEFT, PANEL_HEIGHT - 23), "屁股 0%", font=LABEL_FONT, fill=REAR_POINT, anchor="ls")
    draw.text((PLOT_RIGHT, PANEL_HEIGHT - 23), "头部 100%", font=LABEL_FONT, fill=HEAD_POINT, anchor="rs")
    if marker_ratios == (0.50, 0.70):
        middle_label = "50%线在左  │  70%线在右"
    else:
        middle_label = "  │  ".join(f"距臀 {ratio * 100:.0f}%" for ratio in marker_ratios)
    draw.text((PANEL_WIDTH / 2, PANEL_HEIGHT - 22), middle_label, font=SMALL_FONT, fill=MUTED, anchor="ms")

    record = {
        "cow_id": cow_id,
        "source_top_ply": row["source_top_ply"],
        "extracted_top_ply": row["extracted_ply"],
        "rear_axis_x_m": float(rear_xy[0]),
        "rear_axis_y_m": float(rear_xy[1]),
        "head_axis_x_m": float(head_xy[0]),
        "head_axis_y_m": float(head_xy[1]),
        "pca_full_length_m": full_length,
        "display_orientation": "rear_left_head_right_pca_axis_horizontal",
        "method_version": "pca_full_axis_ratio_markers_preview_v2.0",
    }
    for ratio, marker_distance in zip(marker_ratios, marker_distances):
        marker_xy = rear_xy + marker_distance * axis
        prefix = f"marker_{int(round(ratio * 100))}pct"
        record[f"{prefix}_ratio_from_rear"] = ratio
        record[f"{prefix}_distance_from_rear_m"] = marker_distance
        record[f"{prefix}_distance_to_head_m"] = (1 - ratio) * full_length
        record[f"{prefix}_x_m"] = float(marker_xy[0])
        record[f"{prefix}_y_m"] = float(marker_xy[1])
    return panel, record


def main() -> None:
    rows = read_rows()
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[int, Image.Image]] = []
    records: list[dict[str, object]] = []
    for row in rows:
        panel, record = make_panel(row)
        cow_id = int(record["cow_id"])
        individual_path = INDIVIDUAL_DIR / f"cow_{cow_id:03d}_pca_horizontal_60pct.png"
        panel.save(individual_path, optimize=True)
        panels.append((cow_id, panel))
        records.append(record)

    group_files: list[str] = []
    for group_number, start in enumerate(range(0, len(panels), GROUP_SIZE), start=1):
        group = panels[start : start + GROUP_SIZE]
        group_rows = math.ceil(len(group) / GROUP_COLUMNS)
        canvas = Image.new("RGB", (PANEL_WIDTH * GROUP_COLUMNS, PANEL_HEIGHT * group_rows), BACKGROUND)
        group_draw = ImageDraw.Draw(canvas)
        for index, (_, panel) in enumerate(group):
            x = (index % GROUP_COLUMNS) * PANEL_WIDTH
            y = (index // GROUP_COLUMNS) * PANEL_HEIGHT
            canvas.paste(panel, (x, y))
            if index % GROUP_COLUMNS == 0:
                group_draw.line((PANEL_WIDTH - 1, y + 8, PANEL_WIDTH - 1, y + PANEL_HEIGHT - 8), fill=SEPARATOR, width=2)
            if index < len(group) - GROUP_COLUMNS:
                group_draw.line((x + 10, y + PANEL_HEIGHT - 1, x + PANEL_WIDTH - 10, y + PANEL_HEIGHT - 1), fill=SEPARATOR, width=2)
        first_id, last_id = group[0][0], group[-1][0]
        group_path = GROUP_DIR / f"pca_horizontal_60pct_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.png"
        canvas.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(group_path, optimize=True)
        group_files.append(str(group_path.relative_to(ROOT)))
        for index in range(start, start + len(group)):
            records[index]["preview_group_number"] = group_number
            records[index]["preview_group_file"] = str(group_path.relative_to(ROOT))

    with POSITION_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "group_size": GROUP_SIZE,
                "columns_per_group": GROUP_COLUMNS,
                "group_count": len(group_files),
                "orientation": "PCA full long axis horizontal; rear left; head right",
                "marker": "60% of full PCA long-axis length measured from rear toward head",
                "group_files": group_files,
                "position_record": str(POSITION_CSV.relative_to(ROOT)),
                "method_version": "pca_full_axis_rear60pct_preview_v1.0",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"cattle: {len(records)}")
    print(f"groups: {len(group_files)}")
    for path in group_files:
        print(path)


if __name__ == "__main__":
    main()
