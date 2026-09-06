#!/usr/bin/env python3
"""Rotate existing PCA-overlay previews so the already-drawn PCA line is horizontal."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
RECORDS = BASE / "long_axis_records.csv"
SOURCE_DIR = BASE / "qa_individual"
OUTPUT_ROOT = BASE / "qa_pca_horizontal"
INDIVIDUAL_DIR = OUTPUT_ROOT / "individual"
COMBINED_DIR = OUTPUT_ROOT / "combined"

PANEL_WIDTH = 512
PANEL_HEIGHT = 474
HEADER_HEIGHT = 50
PLOT_BACKGROUND = (10, 15, 23)


def rotate_plot_region(source: Image.Image, angle_degrees: float) -> Image.Image:
    """Keep the title fixed and rotate only the plotted cow/PCA overlay."""
    source = source.convert("RGB")
    header = source.crop((0, 0, PANEL_WIDTH, HEADER_HEIGHT))
    plot = source.crop((0, HEADER_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT))

    rotated = plot.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=PLOT_BACKGROUND,
    )
    scale = min(PANEL_WIDTH / rotated.width, (PANEL_HEIGHT - HEADER_HEIGHT) / rotated.height)
    target_size = (max(1, round(rotated.width * scale)), max(1, round(rotated.height * scale)))
    rotated = rotated.resize(target_size, Image.Resampling.LANCZOS)

    output = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), PLOT_BACKGROUND)
    output.paste(header, (0, 0))
    x = (PANEL_WIDTH - rotated.width) // 2
    y = HEADER_HEIGHT + (PANEL_HEIGHT - HEADER_HEIGHT - rotated.height) // 2
    output.paste(rotated, (x, y))
    return output


def existing_line_angle(source: Image.Image) -> float:
    """Measure the magenta PCA line already present in the raster preview."""
    pixels = np.asarray(source.convert("RGB"))[HEADER_HEIGHT:]
    mask = (
        (pixels[:, :, 0] > 120)
        & (pixels[:, :, 2] > 80)
        & (pixels[:, :, 1] < 100)
        & ((pixels[:, :, 0].astype(int) - pixels[:, :, 1].astype(int)) > 60)
    )
    y, x = np.nonzero(mask)
    if len(x) < 20:
        raise RuntimeError("Unable to detect existing PCA line")
    slope = np.polyfit(x, y, 1)[0]
    return math.degrees(math.atan(slope))


def main() -> None:
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    with RECORDS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("analysis_include", "TRUE").upper() == "TRUE"
        ]

    previews: list[tuple[int, Path]] = []
    angles: list[float] = []
    for row in rows:
        cow_id = int(row["cow_id"])
        source_path = SOURCE_DIR / f"cow_{cow_id:03d}_segmentation_axis.png"
        source = Image.open(source_path)
        angle = existing_line_angle(source)
        output = rotate_plot_region(source, angle)
        output_path = INDIVIDUAL_DIR / f"cow_{cow_id:03d}_pca_horizontal.png"
        output.save(output_path)
        previews.append((cow_id, output_path))
        angles.append(angle)

    combined_paths: list[Path] = []
    for group_number, start in enumerate(range(0, len(previews), 10), start=1):
        group = previews[start : start + 10]
        columns = min(5, len(group))
        rows_count = math.ceil(len(group) / 5)
        canvas = Image.new(
            "RGB",
            (PANEL_WIDTH * columns, PANEL_HEIGHT * rows_count),
            (28, 28, 28),
        )
        for index, (_, path) in enumerate(group):
            canvas.paste(Image.open(path).convert("RGB"), ((index % 5) * PANEL_WIDTH, (index // 5) * PANEL_HEIGHT))
        first_id, last_id = group[0][0], group[-1][0]
        output_path = COMBINED_DIR / (
            f"qa_pca_horizontal_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.jpg"
        )
        canvas.save(output_path, quality=95, subsampling=0)
        combined_paths.append(output_path)

    print(f"qualified cattle: {len(previews)}")
    print("excluded cattle: 19, 28, 50, 98, 147")
    print(f"combined images: {len(combined_paths)}")
    print(f"maximum absolute rotation: {max(abs(value) for value in angles):.2f} degrees")
    print(f"output: {OUTPUT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
