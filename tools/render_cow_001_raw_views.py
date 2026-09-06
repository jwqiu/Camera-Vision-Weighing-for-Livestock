#!/usr/bin/env python3
"""Save Cow 1 Topview RGB and static 3D renders of the raw top/right PLY files."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "1"
OUTPUT = ROOT / "outputs" / "cow_001_raw_views"
PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
    ]
)


def load_font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected end of PLY header: {path}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count: {path}")
        vertices = np.fromfile(handle, dtype=PLY_DTYPE, count=vertex_count)
    valid = (
        np.isfinite(vertices["x"])
        & np.isfinite(vertices["y"])
        & np.isfinite(vertices["z"])
        & (vertices["z"] > 0)
    )
    xyz = np.column_stack([vertices["x"][valid], vertices["y"][valid], vertices["z"][valid]]).astype(float)
    rgb = np.column_stack([vertices["red"][valid], vertices["green"][valid], vertices["blue"][valid]])
    return xyz, rgb


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    return rx @ ry


def render_ply(source: Path, destination: Path, title: str, yaw: float, pitch: float) -> int:
    xyz, rgb = read_ply(source)
    center = np.median(xyz, axis=0)
    centered = xyz - center
    rotated = centered @ rotation(yaw, pitch).T
    radius = float(np.max(np.ptp(rotated, axis=0)))
    camera_distance = radius * 2.8
    depth = rotated[:, 2] + camera_distance
    projected_x = rotated[:, 0] / depth
    projected_y = rotated[:, 1] / depth

    x_low, x_high = np.percentile(projected_x, [0.2, 99.8])
    y_low, y_high = np.percentile(projected_y, [0.2, 99.8])
    width, height, header = 1500, 950, 76
    margin = 40
    scale = min((width - 2 * margin) / (x_high - x_low), (height - header - 2 * margin) / (y_high - y_low))
    px = ((projected_x - (x_low + x_high) / 2) * scale + width / 2).astype(int)
    py = (-(projected_y - (y_low + y_high) / 2) * scale + (height + header) / 2).astype(int)
    visible = (px >= 1) & (px < width - 1) & (py >= header + 1) & (py < height - 1)
    px, py, depth, rgb = px[visible], py[visible], depth[visible], rgb[visible]

    order = np.argsort(depth)[::-1]
    px, py, rgb = px[order], py[order], rgb[order].astype(float)
    rgb = np.clip(rgb * 0.88 + 30, 0, 255).astype(np.uint8)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:] = np.array([8, 17, 31], dtype=np.uint8)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        pixels[py + dy, px + dx] = rgb

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill="#eef4ff", font=load_font(30))
    draw.text((width - 30, 27), f"{len(xyz):,} valid raw points", fill="#b9c6d9", font=load_font(20), anchor="ra")
    image.save(destination, optimize=True)
    return len(xyz)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "raw" / "top" / "rgb-12.09.07.765.png", OUTPUT / "cow_001_topview_rgb.png")
    top_count = render_ply(
        SOURCE / "top-12.09.07.765.ply",
        OUTPUT / "cow_001_topview_raw_3d.png",
        "Cow 1 — Raw Topview point cloud",
        yaw=25,
        pitch=-18,
    )
    right_count = render_ply(
        SOURCE / "right-12.09.07.778.ply",
        OUTPUT / "cow_001_rightview_raw_3d.png",
        "Cow 1 — Raw Rightview point cloud",
        yaw=-25,
        pitch=-12,
    )
    print(f"Saved {OUTPUT}")
    print(f"Topview valid points: {top_count:,}")
    print(f"Rightview valid points: {right_count:,}")


if __name__ == "__main__":
    main()
