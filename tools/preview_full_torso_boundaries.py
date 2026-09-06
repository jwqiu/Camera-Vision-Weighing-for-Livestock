#!/usr/bin/env python3
"""Preview head/tail removal rules along the extended geometric centerline.

Image output only: this script deliberately does not update CSV or JSON records.
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
AXIS_DIR = BASE / "skeleton_axis_comparison"
MAIN_CSV = BASE / "long_axis_records.csv"
PROFILE_CSV = AXIS_DIR / "extended_geometric_width_profiles.csv"
WIDTH_CSV = AXIS_DIR / "extended_geometric_max_width_records.csv"
OUTPUT_DIR = BASE / "full_torso_boundary_preview"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
COMBINED_DIR = OUTPUT_DIR / "combined_vertical_2cols"

REAR_RATIO = 0.175
REAR_CONTINUOUS_M = 0.03
REAR_ABRUPT_HIGH_RATIO = 0.40
REAR_ABRUPT_LOW_RATIO = 0.20
HEAD_RATIO = 0.40
HEAD_CONTINUOUS_M = 0.03
HEAD_IMMEDIATE_RATIO = 0.25
PROFILE_STEP_M = 0.01

PANEL_WIDTH = 900
PANEL_HEIGHT = 500
PLOT_LEFT = 50
PLOT_TOP = 92
PLOT_RIGHT = 850
PLOT_BOTTOM = 465
BACKGROUND = (12, 15, 22)
OUTSIDE_COLOR = (55, 63, 75)
INSIDE_COLOR = (73, 190, 104)
CENTERLINE_COLOR = (28, 147, 220)
REAR_COLOR = (255, 166, 24)
HEAD_COLOR = (35, 201, 235)
TEXT_COLOR = (225, 230, 238)

PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]
)


def load_font(size: int):
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
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


def detect_rear_boundary(widths: np.ndarray, max_index: int):
    required = max(1, int(math.ceil(REAR_CONTINUOUS_M / PROFILE_STEP_M)))
    run = []
    scanned = []
    for index in range(max_index - 1, -1, -1):
        width_ratio = widths[index]
        if not np.isfinite(width_ratio):
            run = []
            scanned.append((index, width_ratio))
            continue
        recent_toward_body = [ratio for _, ratio in scanned[-3:] if np.isfinite(ratio)]
        if width_ratio < REAR_ABRUPT_LOW_RATIO and any(ratio > REAR_ABRUPT_HIGH_RATIO for ratio in recent_toward_body):
            return index, "骤降40%→20%"
        if width_ratio < REAR_RATIO:
            run.append(index)
            if len(run) >= required:
                return run[0], "连续低于17.5%"
        else:
            run = []
        scanned.append((index, width_ratio))
    return 0, "后端兜底"


def detect_head_boundary(widths: np.ndarray, max_index: int):
    required = max(1, int(math.ceil(HEAD_CONTINUOUS_M / PROFILE_STEP_M)))
    run = []
    for index in range(max_index + 1, len(widths)):
        width_ratio = widths[index]
        if not np.isfinite(width_ratio):
            run = []
            continue
        if width_ratio < HEAD_IMMEDIATE_RATIO:
            return index, "立即低于25%"
        if width_ratio < HEAD_RATIO:
            run.append(index)
            if len(run) >= required:
                return run[0], "连续低于40%"
        else:
            run = []
    return len(widths) - 1, "头端兜底"


def nearest_station(points_xy: np.ndarray, centers: np.ndarray, stations: np.ndarray):
    assigned = np.empty(len(points_xy), dtype=float)
    for start in range(0, len(points_xy), 3000):
        batch = points_xy[start : start + 3000]
        distances = ((batch[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned[start : start + len(batch)] = stations[np.argmin(distances, axis=1)]
    return assigned


def create_transform(points_xy: np.ndarray, line_points: np.ndarray):
    combined = np.vstack([points_xy, line_points])
    mins = np.nanmin(combined, axis=0)
    maxs = np.nanmax(combined, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((PLOT_RIGHT - PLOT_LEFT) / span[0], (PLOT_BOTTOM - PLOT_TOP) / span[1])
    center = (mins + maxs) / 2

    def transform(array):
        out = np.empty_like(array, dtype=float)
        out[:, 0] = (array[:, 0] - center[0]) * scale + (PLOT_LEFT + PLOT_RIGHT) / 2
        out[:, 1] = -(array[:, 1] - center[1]) * scale + (PLOT_TOP + PLOT_BOTTOM) / 2
        return out

    return transform


def draw_points(draw: ImageDraw.ImageDraw, points_px: np.ndarray, color, radius=1):
    for x, y in np.rint(points_px).astype(int):
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def boundary_segment(center, tangent, left_offset, right_offset):
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return np.vstack([center + left_offset * normal, center + right_offset * normal])


def full_cut_segment(center, tangent, maximum_width):
    """Extend a cut beyond both silhouette edges to prevent paths around it."""
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    half_length = 0.80 * maximum_width
    return np.vstack([center - half_length * normal, center + half_length * normal])


def stable_tangent(centers: np.ndarray, index: int, radius: int = 4):
    """Estimate a stable rear-to-head tangent around an endpoint or boundary."""
    low = max(0, index - radius)
    high = min(len(centers) - 1, index + radius)
    vector = centers[high] - centers[low]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return vector / norm


def body_side_boundary_tangent(centers: np.ndarray, index: int, side: str, radius: int = 6):
    """Use only torso-side points so a hooked head/tail cannot rotate the cut."""
    if side == "rear":
        vector = centers[min(len(centers) - 1, index + radius)] - centers[index]
    else:
        vector = centers[index] - centers[max(0, index - radius)]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return stable_tangent(centers, index, radius=radius)
    return vector / norm


def points_inside_polygon(points: np.ndarray, polygon: np.ndarray):
    """Vectorized ray casting for a closed polygon in metric XY coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        crosses = (yi > y) != (yj > y)
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        inside ^= crosses & (x < x_intersection)
        xj, yj = xi, yi
    return inside


def connected_torso_after_cuts(points: np.ndarray, seed_xy: np.ndarray, cut_segments):
    """Cut the 1 cm occupancy grid and retain the component containing mid-body."""
    cells = np.floor(points / 0.01).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    cell_centers = (unique_cells.astype(float) + 0.5) * 0.01
    cut = np.zeros(len(unique_cells), dtype=bool)
    for segment in cut_segments:
        start, end = segment
        vector = end - start
        denominator = float(np.dot(vector, vector))
        if denominator < 1e-12:
            continue
        t = np.clip(((cell_centers - start) @ vector) / denominator, 0.0, 1.0)
        closest = start + t[:, None] * vector
        cut |= np.linalg.norm(cell_centers - closest, axis=1) <= 0.015

    active_indices = np.flatnonzero(~cut)
    seed_index = int(active_indices[np.argmin(np.sum((cell_centers[active_indices] - seed_xy) ** 2, axis=1))])
    lookup = {tuple(cell): index for index, cell in enumerate(unique_cells)}
    visited = np.zeros(len(unique_cells), dtype=bool)
    visited[seed_index] = True
    stack = [seed_index]
    while stack:
        index = stack.pop()
        x, y = unique_cells[index]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = lookup.get((x + dx, y + dy))
                if neighbor is not None and not cut[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return visited[inverse]


def make_panel(cow_id, main_row, profile, width_record):
    stations = np.asarray([float(row["station_from_rear_m"]) for row in profile])
    centers = np.asarray([[float(row["center_x_m"]), float(row["center_y_m"])] for row in profile])
    tangents = np.asarray([[float(row["tangent_x"]), float(row["tangent_y"])] for row in profile])
    widths_m = np.asarray([float(row["smoothed_width_m"]) if row["smoothed_width_m"] else np.nan for row in profile])
    left_offsets = np.asarray([float(row["left_offset_m"]) if row["left_offset_m"] else np.nan for row in profile])
    right_offsets = np.asarray([float(row["right_offset_m"]) if row["right_offset_m"] else np.nan for row in profile])
    station_indices = np.arange(len(profile))
    for offsets in (left_offsets, right_offsets):
        valid_offsets = np.isfinite(offsets)
        offsets[~valid_offsets] = np.interp(
            station_indices[~valid_offsets], station_indices[valid_offsets], offsets[valid_offsets]
        )
    maximum_width = float(width_record["extended_geometric_max_width_m"])
    ratios = widths_m / maximum_width
    max_station = float(width_record["max_width_station_m"])
    max_index = int(np.argmin(np.abs(stations - max_station)))
    rear_index, rear_reason = detect_rear_boundary(ratios, max_index)
    head_index, head_reason = detect_head_boundary(ratios, max_index)
    if rear_index >= head_index:
        raise ValueError(f"invalid boundary order for cow {cow_id}")

    vertices = read_ply_vertices(BASE / main_row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    rear_tangent = body_side_boundary_tangent(centers, rear_index, "rear")
    head_tangent = body_side_boundary_tangent(centers, head_index, "head")
    # Build a closed ribbon from the measured left/right body boundaries.
    # This avoids nearest-centerline ambiguity when the head bends back toward
    # the torso, as in cow 026.
    segment_centers = centers[rear_index : head_index + 1]
    segment_tangents = np.asarray([
        stable_tangent(centers, index, radius=3) for index in range(rear_index, head_index + 1)
    ])
    # Use exactly the same endpoint directions for the closed ribbon and the
    # visible boundary lines, so the color transition cannot appear to cross
    # a differently oriented line.
    segment_tangents[0] = rear_tangent
    segment_tangents[-1] = head_tangent
    segment_normals = np.column_stack([-segment_tangents[:, 1], segment_tangents[:, 0]])
    left_edge = segment_centers + (left_offsets[rear_index : head_index + 1] - 0.01)[:, None] * segment_normals
    right_edge = segment_centers + (right_offsets[rear_index : head_index + 1] + 0.01)[:, None] * segment_normals
    rear_segment = full_cut_segment(centers[rear_index], rear_tangent, maximum_width)
    head_segment = full_cut_segment(centers[head_index], head_tangent, maximum_width)
    # Cutting the occupancy grid and retaining the component attached to the
    # widest torso is robust when a hooked head bends back across the geometric
    # side of the boundary line.
    retained = connected_torso_after_cuts(points_xy, centers[max_index], [rear_segment, head_segment])
    transform = create_transform(points_xy, np.vstack([rear_segment, head_segment]))
    points_px = transform(points_xy)
    centers_px = transform(centers)
    rear_px = transform(rear_segment)
    head_px = transform(head_segment)

    image = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"牛 {cow_id:03d}  去头去尾预览", font=TITLE_FONT, fill=TEXT_COLOR)
    draw.text(
        (28, 54),
        f"后侧17.5%/3 cm：{rear_reason}    头侧40%/3 cm：{head_reason}",
        font=SMALL_FONT,
        fill=(170, 179, 193),
    )
    draw_points(draw, points_px[~retained], OUTSIDE_COLOR, radius=1)
    draw_points(draw, points_px[retained], INSIDE_COLOR, radius=1)
    draw.line([tuple(point) for point in centers_px], fill=(7, 16, 24), width=7, joint="curve")
    draw.line([tuple(point) for point in centers_px], fill=CENTERLINE_COLOR, width=3, joint="curve")
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
        "rear_station_m": round(float(stations[rear_index]), 3),
        "head_station_m": round(float(stations[head_index]), 3),
    }


def main():
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        main_rows = {int(row["cow_id"]): row for row in csv.DictReader(handle)}
    with WIDTH_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        width_rows = {int(row["cow_id"]): row for row in csv.DictReader(handle)}
    profiles = {}
    with PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            profiles.setdefault(int(row["cow_id"]), []).append(row)

    previews = []
    metrics = []
    for cow_id in sorted(width_rows):
        panel, metric = make_panel(cow_id, main_rows[cow_id], profiles[cow_id], width_rows[cow_id])
        individual_path = INDIVIDUAL_DIR / f"cow_{cow_id:03d}_full_torso_boundary_preview.png"
        panel.save(individual_path)
        previews.append((cow_id, individual_path))
        metrics.append(metric)

    combined_paths = []
    for group_number, start in enumerate(range(0, len(previews), 10), start=1):
        group = previews[start : start + 10]
        rows = math.ceil(len(group) / 2)
        canvas = Image.new("RGB", (PANEL_WIDTH * 2, PANEL_HEIGHT * rows), (28, 28, 28))
        for index, (_, path) in enumerate(group):
            panel = Image.open(path).convert("RGB")
            canvas.paste(panel, ((index % 2) * PANEL_WIDTH, (index // 2) * PANEL_HEIGHT))
        first_id, last_id = group[0][0], group[-1][0]
        output_path = COMBINED_DIR / f"full_torso_preview_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.jpg"
        canvas.save(output_path, quality=95, subsampling=0)
        combined_paths.append(str(output_path.relative_to(ROOT)))

    print(json.dumps({
        "cows": len(metrics),
        "combined_images": len(combined_paths),
        "layout": "5 rows x 2 cows (last image: 3 rows x 2 cows)",
        "rear_reason_counts": Counter(item["rear_reason"] for item in metrics),
        "head_reason_counts": Counter(item["head_reason"] for item in metrics),
        "rear_fallback_cows": [item["cow_id"] for item in metrics if item["rear_reason"] == "后端兜底"],
        "head_fallback_cows": [item["cow_id"] for item in metrics if item["head_reason"] == "头端兜底"],
        "combined_paths": combined_paths,
        "note": "preview images only; no CSV or JSON records were updated",
    }, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
