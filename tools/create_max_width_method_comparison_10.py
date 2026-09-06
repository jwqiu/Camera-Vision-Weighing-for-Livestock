#!/usr/bin/env python3
"""Draw three Topview width definitions for a reproducible random sample of 10 cattle."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
MAIN_CSV = BASE / "long_axis_records.csv"
SKELETON_CSV = BASE / "skeleton_max_width_61" / "skeleton_max_width_records_61.csv"
OUTPUT_DIR = BASE / "max_width_method_comparison_10"
OUTPUT_PNG = OUTPUT_DIR / "maximum_width_methods_random_10.png"
OUTPUT_CSV = OUTPUT_DIR / "maximum_width_methods_random_10.csv"
OUTPUT_JSON = OUTPUT_DIR / "maximum_width_methods_random_10.json"

SEED = 20260906
GRID_M = 0.01
PANEL_W = 680
PANEL_H = 300
MARGIN_X = 28
PLOT_TOP = 60
PLOT_BOTTOM = 270

sys.path.insert(0, str(ROOT / "tools"))
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def font(size: int, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = font(30, True)
LEGEND_FONT = font(21)
PANEL_TITLE_FONT = font(21, True)
LABEL_FONT = font(17)
SMALL_FONT = font(15)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < height and 0 <= xx < width and not mask[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                queue.append((yy, xx))
    return mask | (~mask & ~outside)


def maximum_inscribed_circle(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    rear: float,
    head: float,
) -> tuple[float, float, float]:
    selected = (longitudinal >= rear) & (longitudinal <= head)
    t = longitudinal[selected]
    q = lateral[selected]
    t0 = math.floor(float(t.min()) / GRID_M) * GRID_M - 2 * GRID_M
    q0 = math.floor(float(q.min()) / GRID_M) * GRID_M - 2 * GRID_M
    width = int(math.ceil((float(t.max()) - t0) / GRID_M)) + 3
    height = int(math.ceil((float(q.max()) - q0) / GRID_M)) + 3
    x = np.clip(np.floor((t - t0) / GRID_M).astype(int), 0, width - 1)
    y = np.clip(np.floor((q - q0) / GRID_M).astype(int), 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y, x] = 255
    closed = Image.fromarray(mask).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    body = fill_holes(np.asarray(closed) > 0)

    # Restrict to the largest connected body component.
    seen = np.zeros_like(body, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for yy, xx in zip(*np.where(body)):
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        queue = deque([(int(yy), int(xx))])
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and body[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(component)
    largest = max(components, key=len)
    body[:] = False
    yy = np.asarray([point[0] for point in largest])
    xx = np.asarray([point[1] for point in largest])
    body[yy, xx] = True

    padded = np.pad(body, 1, constant_values=False)
    neighbor_inside = np.zeros_like(padded)
    neighbor_inside[1:-1, 1:-1] = body
    outside = ~padded
    adjacent = outside & (
        np.roll(neighbor_inside, 1, axis=0)
        | np.roll(neighbor_inside, -1, axis=0)
        | np.roll(neighbor_inside, 1, axis=1)
        | np.roll(neighbor_inside, -1, axis=1)
        | np.roll(np.roll(neighbor_inside, 1, axis=0), 1, axis=1)
        | np.roll(np.roll(neighbor_inside, 1, axis=0), -1, axis=1)
        | np.roll(np.roll(neighbor_inside, -1, axis=0), 1, axis=1)
        | np.roll(np.roll(neighbor_inside, -1, axis=0), -1, axis=1)
    )
    interior = np.column_stack(np.where(padded))
    boundary = np.column_stack(np.where(adjacent))
    best_distance_sq = -1.0
    best_point = None
    for start in range(0, len(interior), 256):
        chunk = interior[start : start + 256]
        distance_sq = np.sum((chunk[:, None, :] - boundary[None, :, :]) ** 2, axis=2)
        nearest = np.min(distance_sq, axis=1)
        local_index = int(np.argmax(nearest))
        if float(nearest[local_index]) > best_distance_sq:
            best_distance_sq = float(nearest[local_index])
            best_point = chunk[local_index]
    if best_point is None:
        raise ValueError("maximum inscribed circle not found")
    py, px = best_point - 1
    center_t = t0 + (float(px) + 0.5) * GRID_M
    center_q = q0 + (float(py) + 0.5) * GRID_M
    radius = math.sqrt(best_distance_sq) * GRID_M
    return center_t, center_q, radius


def transform_xy(point_xy: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    normal = np.asarray([-axis[1], axis[0]])
    relative = point_xy - centroid
    return np.asarray([relative @ axis, relative @ normal])


def main() -> None:
    rows = read_csv(MAIN_CSV)
    strict = [
        row
        for row in rows
        if row.get("pca_full_torso_analysis_eligible", "").upper() == "TRUE"
        and row.get("pca_full_torso_rear_boundary_reason") == "first_below_45pct"
        and row.get("pca_full_torso_head_boundary_reason") == "first_below_48pct"
    ]
    if len(strict) != 61:
        raise ValueError(f"expected 61 cattle, found {len(strict)}")
    rng = np.random.default_rng(SEED)
    selected_ids = sorted(rng.choice([int(row["cow_id"]) for row in strict], size=10, replace=False).tolist())
    rows_by_id = {int(row["cow_id"]): row for row in strict}
    skeleton_by_id = {int(row["cow_id"]): row for row in read_csv(SKELETON_CSV)}

    canvas = Image.new("RGB", (PANEL_W * 2, 160 + PANEL_H * 5), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((PANEL_W, 20), "10头牛三种Topview宽度定义对比", font=TITLE_FONT, fill="#172033", anchor="ma")
    legend_y = 74
    legends = [
        ("#dc2626", "实线", "PCA最大体宽"),
        ("#2563eb", "虚线", "骨架局部最大体宽"),
        ("#16a34a", "点线＋圆", "核心躯干最大内接圆直径"),
    ]
    x_cursor = 130
    for color, style, label in legends:
        draw.line((x_cursor, legend_y, x_cursor + 55, legend_y), fill=color, width=5)
        if style == "虚线":
            draw.line((x_cursor + 13, legend_y, x_cursor + 25, legend_y), fill="#f8fafc", width=6)
            draw.line((x_cursor + 39, legend_y, x_cursor + 50, legend_y), fill="#f8fafc", width=6)
        draw.text((x_cursor + 66, legend_y), label, font=LEGEND_FONT, fill="#172033", anchor="lm")
        x_cursor += 390 if label != legends[-1][2] else 0

    output_rows: list[dict[str, object]] = []
    for panel_index, cow_id in enumerate(selected_ids):
        row = rows_by_id[cow_id]
        skeleton = skeleton_by_id[cow_id]
        vertices = read_ply_vertices(BASE / row["extracted_ply"])
        xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
        centroid = np.asarray(
            [float(row["pca_full_torso_axis_centroid_x_m"]), float(row["pca_full_torso_axis_centroid_y_m"])]
        )
        axis = np.asarray([float(row["pca_full_torso_axis_unit_x"]), float(row["pca_full_torso_axis_unit_y"])])
        axis /= np.linalg.norm(axis)
        normal = np.asarray([-axis[1], axis[0]])
        relative = xy - centroid
        longitudinal = relative @ axis
        lateral = relative @ normal
        rear = float(row["pca_full_torso_rear_boundary_axis_m"])
        head = float(row["pca_full_torso_head_boundary_axis_m"])
        circle_t, circle_q, circle_radius = maximum_inscribed_circle(longitudinal, lateral, rear, head)

        x0 = (panel_index % 2) * PANEL_W
        y0 = 130 + (panel_index // 2) * PANEL_H
        panel_draw = draw
        t_min, t_max = np.quantile(longitudinal, [0.002, 0.998])
        q_min, q_max = np.quantile(lateral, [0.002, 0.998])
        t_pad = max(0.04, (t_max - t_min) * 0.03)
        q_pad = max(0.03, (q_max - q_min) * 0.08)
        t_min -= t_pad
        t_max += t_pad
        q_min -= q_pad
        q_max += q_pad
        plot_w = PANEL_W - 2 * MARGIN_X
        plot_h = PLOT_BOTTOM - PLOT_TOP
        scale = min(plot_w / (t_max - t_min), plot_h / (q_max - q_min))
        center_px_x = x0 + PANEL_W / 2
        center_px_y = y0 + (PLOT_TOP + PLOT_BOTTOM) / 2
        center_t_plot = (t_min + t_max) / 2
        center_q_plot = (q_min + q_max) / 2

        def px(t_value: float) -> float:
            return center_px_x + (t_value - center_t_plot) * scale

        def py(q_value: float) -> float:
            return center_px_y - (q_value - center_q_plot) * scale

        # Actual Topview points, oriented rear-left and head-right.
        sample_step = max(1, len(longitudinal) // 18000)
        for t_value, q_value in zip(longitudinal[::sample_step], lateral[::sample_step]):
            xx, yy = int(round(px(float(t_value)))), int(round(py(float(q_value))))
            if x0 <= xx < x0 + PANEL_W and y0 + PLOT_TOP <= yy <= y0 + PLOT_BOTTOM:
                panel_draw.point((xx, yy), fill="#bcc7d5")

        # Accepted PCA core boundaries.
        for boundary in (rear, head):
            xx = px(boundary)
            panel_draw.line((xx, y0 + PLOT_TOP, xx, y0 + PLOT_BOTTOM), fill="#94a3b8", width=1)

        # PCA width: use the actual local lateral midpoint and recorded smoothed width.
        pca_station = float(row["pca_full_torso_max_width_axis_m"])
        local = np.abs(longitudinal - pca_station) <= 0.005
        low, high = np.quantile(lateral[local], [0.02, 0.98])
        midpoint = float((low + high) / 2)
        pca_width = float(row["pca_full_torso_max_width_m"])
        panel_draw.line(
            (px(pca_station), py(midpoint - pca_width / 2), px(pca_station), py(midpoint + pca_width / 2)),
            fill="#dc2626",
            width=5,
        )

        # Skeleton width: retain its local-normal angle but scale to the recorded smoothed width.
        skeleton_center_xy = np.asarray(
            [float(skeleton["skeleton_max_width_center_x_m"]), float(skeleton["skeleton_max_width_center_y_m"])]
        )
        tangent_xy = np.asarray(
            [float(skeleton["skeleton_local_tangent_x"]), float(skeleton["skeleton_local_tangent_y"])]
        )
        local_normal_xy = np.asarray([-tangent_xy[1], tangent_xy[0]])
        local_normal_xy /= np.linalg.norm(local_normal_xy)
        skeleton_width = float(skeleton["skeleton_max_width_m"])
        endpoint_1 = skeleton_center_xy - local_normal_xy * skeleton_width / 2
        endpoint_2 = skeleton_center_xy + local_normal_xy * skeleton_width / 2
        endpoint_1_tq = transform_xy(endpoint_1, centroid, axis)
        endpoint_2_tq = transform_xy(endpoint_2, centroid, axis)
        points = (
            px(float(endpoint_1_tq[0])), py(float(endpoint_1_tq[1])),
            px(float(endpoint_2_tq[0])), py(float(endpoint_2_tq[1])),
        )
        # Dashed blue line.
        total = math.hypot(points[2] - points[0], points[3] - points[1])
        for start in np.arange(0, total, 16):
            end = min(total, start + 9)
            f0, f1 = start / total, end / total
            segment = (
                points[0] + (points[2] - points[0]) * f0,
                points[1] + (points[3] - points[1]) * f0,
                points[0] + (points[2] - points[0]) * f1,
                points[1] + (points[3] - points[1]) * f1,
            )
            panel_draw.line(segment, fill="#2563eb", width=5)

        # Maximum inscribed circle and its transverse diameter.
        radius_px = circle_radius * scale
        cx, cy = px(circle_t), py(circle_q)
        panel_draw.ellipse((cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px), outline="#16a34a", width=3)
        diameter_y0, diameter_y1 = cy - radius_px, cy + radius_px
        for start in np.arange(diameter_y0, diameter_y1, 12):
            panel_draw.line((cx, start, cx, min(diameter_y1, start + 6)), fill="#16a34a", width=5)

        circle_diameter = 2 * circle_radius
        panel_draw.text((x0 + 18, y0 + 12), f"牛 {cow_id:03d}", font=PANEL_TITLE_FONT, fill="#172033")
        panel_draw.text(
            (x0 + 104, y0 + 15),
            f"PCA {pca_width*100:.1f}  骨架 {skeleton_width*100:.1f}  内接圆 {circle_diameter*100:.1f} cm",
            font=LABEL_FONT,
            fill="#172033",
        )
        panel_draw.text((x0 + PANEL_W - 16, y0 + PANEL_H - 20), "臀部 ←　→ 头部", font=SMALL_FONT, fill="#475569", anchor="ra")
        if panel_index % 2 == 0:
            panel_draw.line((x0 + PANEL_W - 1, y0 + 8, x0 + PANEL_W - 1, y0 + PANEL_H - 8), fill="#d5dce5", width=1)
        if panel_index < 8:
            panel_draw.line((x0 + 12, y0 + PANEL_H - 1, x0 + PANEL_W - 12, y0 + PANEL_H - 1), fill="#d5dce5", width=1)

        circle_center_xy = centroid + circle_t * axis + circle_q * normal
        output_rows.append(
            {
                "random_seed": SEED,
                "cow_id": cow_id,
                "pca_max_width_m": pca_width,
                "pca_max_width_station_m": pca_station,
                "skeleton_max_width_m": skeleton_width,
                "skeleton_max_width_station_from_rear_m": float(skeleton["skeleton_max_width_station_from_rear_m"]),
                "max_inscribed_circle_diameter_m": circle_diameter,
                "max_inscribed_circle_radius_m": circle_radius,
                "max_inscribed_circle_center_x_m": float(circle_center_xy[0]),
                "max_inscribed_circle_center_y_m": float(circle_center_xy[1]),
                "max_inscribed_circle_grid_m": GRID_M,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reduced = canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    reduced.save(OUTPUT_PNG, optimize=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "random_seed": SEED,
                "sampled_cow_ids": selected_ids,
                "methods": {
                    "pca": "PCA 45/48 core maximum-width station; 2nd-to-98th percentile local width; 3cm smoothing",
                    "skeleton": "local normal to rear-extended curved skeleton; contiguous 1st-to-99th percentile width; 3cm smoothing",
                    "inscribed_circle": "largest 1cm-grid circle inside the PCA 45/48 core Topview occupancy mask after 1-cell closing and hole filling",
                },
                "records": output_rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("sample", selected_ids)
    print("png", OUTPUT_PNG, OUTPUT_PNG.stat().st_size)
    print("csv", OUTPUT_CSV)


if __name__ == "__main__":
    main()
