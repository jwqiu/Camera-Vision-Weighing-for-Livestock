#!/usr/bin/env python3
"""Create the three compact PNG figures displayed in the project README."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preview_pca_full_torso_boundaries import BASE, MAIN_CSV, read_ply_vertices


ROOT = BASE.parent
OUTPUT = ROOT / "docs" / "images"
SLICE_CSV = BASE / "elliptical_slice_volumes_two_right_boundaries" / "elliptical_core_torso_slice_details_two_methods_61.csv"
PREDICTIONS_CSV = ROOT / "outputs" / "01a074cb-14e9-7673-9f3d-8a2022f79eec" / "elliptical_volume_height_virtual_chest_width_predictions_61.csv"
BEST_PREDICTION = "predicted_weight_kg__ellipse_volume_plus_rightview_median_depth_original"
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size: int):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = font(30)
LABEL = font(21)
SMALL = font(17)
COLORS = {"blue": "#2563eb", "orange": "#f59e0b", "cyan": "#06b6d4", "green": "#16a34a", "gray": "#94a3b8", "dark": "#172033", "grid": "#d8dee9"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canvas(width: int, height: int, title: str):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 22), title, font=TITLE, fill=COLORS["dark"], anchor="ma")
    return image, draw


def scale(values: np.ndarray, low: float, high: float, start: float, end: float) -> np.ndarray:
    span = max(high - low, 1e-9)
    return start + (values - low) / span * (end - start)


def save_core_torso() -> None:
    row = next(item for item in rows(MAIN_CSV) if item["cow_id"] == "1")
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    centroid = np.array([float(row["pca_full_torso_axis_centroid_x_m"]), float(row["pca_full_torso_axis_centroid_y_m"])])
    axis = np.array([float(row["pca_full_torso_axis_unit_x"]), float(row["pca_full_torso_axis_unit_y"])])
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    relative = xy - centroid
    longitudinal = relative @ axis
    lateral = relative @ normal
    rear = float(row["pca_full_torso_rear_boundary_axis_m"])
    head = float(row["pca_full_torso_head_boundary_axis_m"])
    core = (longitudinal >= rear) & (longitudinal <= head)

    image, draw = canvas(1200, 590, "Core torso localization — Cow 1")
    left, top, right, bottom = 90, 85, 1155, 500
    xmin, xmax = np.percentile(longitudinal, [0.2, 99.8])
    ymin, ymax = np.percentile(lateral, [0.2, 99.8])
    px = scale(longitudinal, xmin, xmax, left, right)
    py = scale(lateral, ymin, ymax, bottom, top)
    indices = np.arange(0, len(px), max(1, len(px) // 28000))
    for index in indices:
        color = "#4fc36b" if core[index] else COLORS["gray"]
        draw.ellipse((px[index] - 1, py[index] - 1, px[index] + 1, py[index] + 1), fill=color)
    axis_y = float(scale(np.array([0.0]), ymin, ymax, bottom, top)[0])
    draw.line((left, axis_y, right, axis_y), fill=COLORS["blue"], width=3)
    for value, color, label in [(rear, COLORS["orange"], "Rear boundary (45%)"), (head, COLORS["cyan"], "Front boundary (48%)")]:
        x = float(scale(np.array([value]), xmin, xmax, left, right)[0])
        draw.line((x, top, x, bottom), fill=color, width=5)
        draw.text((x, top - 10), label, font=SMALL, fill=color, anchor="ms")
    draw.text((left, 545), "Position along PCA body axis", font=LABEL, fill=COLORS["dark"])
    image.save(OUTPUT / "core-torso-localization.png", optimize=True)


def plot_line(draw, x, y, bounds, color, width=4):
    left, top, right, bottom = bounds
    px = scale(x, float(x.min()), float(x.max()), left, right)
    py = scale(y, float(y.min()), float(y.max()), bottom, top)
    draw.line([(float(a), float(b)) for a, b in zip(px, py)], fill=color, width=width)


def save_elliptical_slices() -> None:
    data = [item for item in rows(SLICE_CSV) if item["cow_id"] == "1" and item["boundary_method"] == "proportional"]
    position = np.array([float(item["relative_core_position"]) * 100 for item in data])
    width = np.array([float(item["top_width_m"]) for item in data])
    depth = np.array([float(item["corrected_right_depth_m"]) for item in data])
    area = np.array([float(item["ellipse_area_m2"]) for item in data])

    image, draw = canvas(1100, 720, "Multi-view elliptical torso slices — Cow 1")
    bounds1 = (105, 100, 1050, 365)
    bounds2 = (105, 430, 1050, 640)
    for bounds in (bounds1, bounds2):
        draw.rectangle(bounds, outline=COLORS["grid"], width=2)
    common_min, common_max = float(min(width.min(), depth.min())), float(max(width.max(), depth.max()))
    px = scale(position, float(position.min()), float(position.max()), bounds1[0], bounds1[2])
    for values, color in [(width, COLORS["blue"]), (depth, "#f97316")]:
        py = scale(values, common_min, common_max, bounds1[3], bounds1[1])
        draw.line([(float(a), float(b)) for a, b in zip(px, py)], fill=color, width=4)
    plot_line(draw, position, area, bounds2, COLORS["green"])
    draw.text((120, 115), "Topview width", font=SMALL, fill=COLORS["blue"])
    draw.text((300, 115), "Rightview depth", font=SMALL, fill="#f97316")
    draw.text((120, 445), "Estimated ellipse area", font=SMALL, fill=COLORS["green"])
    draw.text((520, 680), "Relative position within core torso (%)", font=LABEL, fill=COLORS["dark"], anchor="ma")
    draw.text((18, 235), "Distance (m)", font=LABEL, fill=COLORS["dark"])
    draw.text((18, 535), "Area (m²)", font=LABEL, fill=COLORS["dark"])
    image.save(OUTPUT / "elliptical-torso-volume.png", optimize=True)


def save_prediction_plot() -> None:
    data = rows(PREDICTIONS_CSV)
    actual = np.array([float(item["ground_truth_weight_kg"]) for item in data])
    predicted = np.array([float(item[BEST_PREDICTION]) for item in data])
    low = float(min(actual.min(), predicted.min()) - 10)
    high = float(max(actual.max(), predicted.max()) + 10)
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    image, draw = canvas(820, 760, "Actual versus predicted live weight")
    left, top, right, bottom = 105, 95, 770, 675
    draw.rectangle((left, top, right, bottom), outline=COLORS["grid"], width=2)
    draw.line((left, bottom, right, top), fill="#64748b", width=3)
    px = scale(actual, low, high, left, right)
    py = scale(predicted, low, high, bottom, top)
    for x, y in zip(px, py):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS["blue"])
    draw.text((130, 120), f"LOOCV MAE = {mae:.1f} kg\nLOOCV RMSE = {rmse:.1f} kg", font=LABEL, fill=COLORS["dark"], spacing=8)
    draw.text((440, 715), "Actual weight (kg)", font=LABEL, fill=COLORS["dark"], anchor="ma")
    draw.text((18, 350), "Predicted weight (kg)", font=LABEL, fill=COLORS["dark"])
    image.save(OUTPUT / "actual-vs-predicted-weight.png", optimize=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_core_torso()
    save_elliptical_slices()
    save_prediction_plot()


if __name__ == "__main__":
    main()
