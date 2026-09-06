#!/usr/bin/env python3
"""Create a 2x2 comparison of Cow 1 RGB frames and saved point-cloud views."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "cow_001_rgb_pointcloud_2x2.png"
PANEL_SIZE = (1920, 1080)
BACKGROUND = (8, 17, 31)

SOURCES = [
    ROOT / "dataset/1/raw/top/rgb-12.09.07.765.png",
    ROOT / "Codex Image 7 Sept 2026, 02_21_23.png",
    ROOT / "dataset/1/raw/right/rgb-12.09.07.778.png",
    ROOT / "Codex Image 7 Sept 2026, 02_38_05.png",
]


def fit_panel(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", PANEL_SIZE, BACKGROUND)
        x = (PANEL_SIZE[0] - image.width) // 2
        y = (PANEL_SIZE[1] - image.height) // 2
        panel.paste(image, (x, y))
        return panel


def main() -> None:
    panels = [fit_panel(path) for path in SOURCES]
    canvas = Image.new("RGB", (PANEL_SIZE[0] * 2, PANEL_SIZE[1] * 2), BACKGROUND)
    positions = [(0, 0), (PANEL_SIZE[0], 0), (0, PANEL_SIZE[1]), PANEL_SIZE]
    for panel, position in zip(panels, positions):
        canvas.paste(panel, position)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
