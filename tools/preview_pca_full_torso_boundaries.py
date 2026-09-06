#!/usr/bin/env python3
"""Create PCA-axis head/tail boundary previews for qualified cattle.

Preview only. No CSV, JSON, or formal measurement record is updated.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
OUTPUT_DIR = BASE / "pca_full_torso_boundary_preview"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
COMBINED_DIR = OUTPUT_DIR / "combined_vertical_2cols"

BIN_M = 0.01
SMOOTHING_BINS = 3
REAR_RATIO = 0.45
HEAD_RATIO = 0.48

PANEL_WIDTH = 900
PANEL_HEIGHT = 500
PLOT_LEFT = 50
PLOT_TOP = 92
PLOT_RIGHT = 850
PLOT_BOTTOM = 465
BACKGROUND = (12, 15, 22)
OUTSIDE_COLOR = (55, 63, 75)
INSIDE_COLOR = (73, 190, 104)
AXIS_COLOR = (31, 150, 222)
REAR_COLOR = (255, 166, 24)
HEAD_COLOR = (35, 201, 235)
TEXT_COLOR = (225, 230, 238)

PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]
)


def load_font(size: int):
    for path in (
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(28)
LABEL_FONT = load_font(22)
SMALL_FONT = load_font(18)


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"unexpected end of PLY header: {path}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count: {path}")
        return np.fromfile(handle, dtype=PLY_DTYPE, count=vertex_count)


def rolling_median(values: np.ndarray, window: int):
    half = window // 2
    result = np.empty_like(values)
    for index in range(len(values)):
        low = max(0, index - half)
        high = min(len(values), index + half + 1)
        result[index] = np.nanmedian(values[low:high])
    return result


def pca_width_profile(points_xy: np.ndarray, centroid: np.ndarray, axis: np.ndarray):
    normal = np.array([-axis[1], axis[0]], dtype=float)
    relative = points_xy - centroid
    longitudinal = relative @ axis
    lateral = relative @ normal
    start = math.floor(float(np.min(longitudinal)) / BIN_M) * BIN_M
    stop = math.ceil(float(np.max(longitudinal)) / BIN_M) * BIN_M
    stations = np.arange(start, stop + BIN_M * 0.5, BIN_M)
    indices = np.clip(np.floor((longitudinal - start) / BIN_M).astype(int), 0, len(stations) - 1)
    widths = np.full(len(stations), np.nan)
    for index in np.unique(indices):
        values = lateral[indices == index]
        if len(values) >= 2:
            widths[index] = np.percentile(values, 98) - np.percentile(values, 2)
    valid = np.isfinite(widths)
    widths[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), widths[valid])
    return longitudinal, lateral, stations, rolling_median(widths, SMOOTHING_BINS)


def detect_rear(widths: np.ndarray, max_index: int, maximum_width: float):
    for index in range(max_index - 1, -1, -1):
        if widths[index] < REAR_RATIO * maximum_width:
            return index, "首次低于45%"
    return 0, "后端兜底"


def detect_head(widths: np.ndarray, max_index: int, maximum_width: float):
    for index in range(max_index + 1, len(widths)):
        if widths[index] < HEAD_RATIO * maximum_width:
            return index, "首次低于48%"
    return len(widths) - 1, "头端兜底"


def create_transform(points_xy: np.ndarray, extra_points: np.ndarray):
    combined = np.vstack([points_xy, extra_points])
    mins = np.nanmin(combined, axis=0)
    maxs = np.nanmax(combined, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((PLOT_RIGHT - PLOT_LEFT) / span[0], (PLOT_BOTTOM - PLOT_TOP) / span[1])
    center = (mins + maxs) / 2

    def transform(array):
        output = np.empty_like(array, dtype=float)
        output[:, 0] = (array[:, 0] - center[0]) * scale + (PLOT_LEFT + PLOT_RIGHT) / 2
        output[:, 1] = -(array[:, 1] - center[1]) * scale + (PLOT_TOP + PLOT_BOTTOM) / 2
        return output

    return transform


def draw_points(draw: ImageDraw.ImageDraw, points_px: np.ndarray, color, radius=1):
    for x, y in np.rint(points_px).astype(int):
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def make_panel(row):
    cow_id = int(row["cow_id"])
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.array([float(row["centroid_x_m"]), float(row["centroid_y_m"])])
    axis = np.array([float(row["long_axis_unit_x"]), float(row["long_axis_unit_y"])])
    axis /= np.linalg.norm(axis)
    longitudinal, _, stations, widths = pca_width_profile(points_xy, centroid, axis)
    max_index = int(np.nanargmax(widths))
    maximum_width = float(widths[max_index])
    rear_index, rear_reason = detect_rear(widths, max_index, maximum_width)
    head_index, head_reason = detect_head(widths, max_index, maximum_width)
    if rear_index >= head_index:
        raise ValueError(f"invalid boundary order for cow {cow_id}")

    rear_station = float(stations[rear_index])
    head_station = float(stations[head_index])
    retained = (longitudinal >= rear_station) & (longitudinal <= head_station)
    normal = np.array([-axis[1], axis[0]])
    half_line = maximum_width * 0.80
    rear_center = centroid + rear_station * axis
    head_center = centroid + head_station * axis
    rear_segment = np.vstack([rear_center - half_line * normal, rear_center + half_line * normal])
    head_segment = np.vstack([head_center - half_line * normal, head_center + half_line * normal])
    axis_segment = np.vstack([centroid + stations[0] * axis, centroid + stations[-1] * axis])

    transform = create_transform(points_xy, np.vstack([rear_segment, head_segment]))
    points_px = transform(points_xy)
    rear_px = transform(rear_segment)
    head_px = transform(head_segment)
    axis_px = transform(axis_segment)

    image = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"牛 {cow_id:03d}  PCA中轴去头去尾预览", font=TITLE_FONT, fill=TEXT_COLOR)
    draw.text(
        (28, 54),
        f"最大体宽 {maximum_width:.3f} m    后侧：{rear_reason}    头侧：{head_reason}",
        font=SMALL_FONT,
        fill=(170, 179, 193),
    )
    draw_points(draw, points_px[~retained], OUTSIDE_COLOR, radius=1)
    draw_points(draw, points_px[retained], INSIDE_COLOR, radius=1)
    draw.line([tuple(point) for point in axis_px], fill=(7, 16, 24), width=7)
    draw.line([tuple(point) for point in axis_px], fill=AXIS_COLOR, width=3)
    draw.line([tuple(point) for point in rear_px], fill=(7, 16, 24), width=9)
    draw.line([tuple(point) for point in rear_px], fill=REAR_COLOR, width=5)
    draw.line([tuple(point) for point in head_px], fill=(7, 16, 24), width=9)
    draw.line([tuple(point) for point in head_px], fill=HEAD_COLOR, width=5)
    rear_label = (max(8, min(PANEL_WIDTH - 120, int(rear_px[:, 0].mean()) - 50)), max(PLOT_TOP, int(rear_px[:, 1].min()) - 30))
    head_label = (max(8, min(PANEL_WIDTH - 120, int(head_px[:, 0].mean()) - 50)), max(PLOT_TOP, int(head_px[:, 1].min()) - 30))
    draw.text(rear_label, "后侧边界", font=LABEL_FONT, fill=REAR_COLOR)
    draw.text(head_label, "头侧边界", font=LABEL_FONT, fill=HEAD_COLOR)
    return image, {
        "cow_id": cow_id,
        "rear_reason": rear_reason,
        "head_reason": head_reason,
        "rear_station_m": round(rear_station, 3),
        "head_station_m": round(head_station, 3),
    }


def main():
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for path in INDIVIDUAL_DIR.glob("cow_*_pca_full_torso_boundary_preview.png"):
        path.unlink()
    for path in COMBINED_DIR.glob("pca_full_torso_group_*.jpg"):
        path.unlink()
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row.get("analysis_include", "TRUE").upper() == "TRUE"]
    excluded = sorted(int(row["cow_id"]) for row in all_rows if row not in rows)

    previews = []
    metrics = []
    for row in rows:
        cow_id = int(row["cow_id"])
        panel, metric = make_panel(row)
        path = INDIVIDUAL_DIR / f"cow_{cow_id:03d}_pca_full_torso_boundary_preview.png"
        panel.save(path)
        previews.append((cow_id, path))
        metrics.append(metric)

    combined_paths = []
    for group_number, start in enumerate(range(0, len(previews), 10), start=1):
        group = previews[start : start + 10]
        row_count = math.ceil(len(group) / 2)
        canvas = Image.new("RGB", (PANEL_WIDTH * 2, PANEL_HEIGHT * row_count), (28, 28, 28))
        for index, (_, path) in enumerate(group):
            canvas.paste(Image.open(path).convert("RGB"), ((index % 2) * PANEL_WIDTH, (index // 2) * PANEL_HEIGHT))
        first_id, last_id = group[0][0], group[-1][0]
        output_path = COMBINED_DIR / f"pca_full_torso_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.jpg"
        canvas.save(output_path, quality=95, subsampling=0)
        combined_paths.append(str(output_path.relative_to(ROOT)))

    print(json.dumps({
        "included_cows": len(metrics),
        "excluded_unqualified": excluded,
        "combined_images": len(combined_paths),
        "layout": f"5 rows x 2 cows; last image contains {len(previews) % 10 or 10} cows",
        "rear_reason_counts": dict(Counter(item["rear_reason"] for item in metrics)),
        "head_reason_counts": dict(Counter(item["head_reason"] for item in metrics)),
        "rear_fallback_cows": [item["cow_id"] for item in metrics if item["rear_reason"] == "后端兜底"],
        "head_fallback_cows": [item["cow_id"] for item in metrics if item["head_reason"] == "头端兜底"],
        "combined_paths": combined_paths,
        "note": "preview images only; no CSV or JSON record was updated",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
