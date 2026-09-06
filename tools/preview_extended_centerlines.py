#!/usr/bin/env python3
"""Create image-only previews with the rear geometric centerline extended.

This script intentionally does not update any CSV/JSON centerline records.
"""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
AXIS_DIR = BASE / "skeleton_axis_comparison"
POINTS_JSON = AXIS_DIR / "geometric_centerline_records.json"
PREVIEW_DIR = AXIS_DIR / "extended_preview"
PREVIEW_INDIVIDUAL = PREVIEW_DIR / "skeleton_individual"
PREVIEW_COMPARISONS = PREVIEW_DIR / "individual_comparisons"
COMBINED_DIR = AXIS_DIR / "combined_vertical"
MAIN_CSV = BASE / "long_axis_records.csv"

PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("r", "u1"),
        ("g", "u1"),
        ("b", "u1"),
        ("a", "u1"),
    ]
)


def shifted(mask: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    out = np.full_like(mask, fill)
    h, w = mask.shape
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def disk_offsets(radius: int):
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for dy, dx in disk_offsets(radius):
        out &= shifted(mask, dy, dx, fill=False)
    return out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for dy, dx in disk_offsets(radius):
        out |= shifted(mask, dy, dx, fill=False)
    return out


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for sy, sx in zip(*np.where(mask)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < h
                        and 0 <= xx < w
                        and mask[yy, xx]
                        and not seen[yy, xx]
                    ):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        result.append(comp)
    return sorted(result, key=len, reverse=True)


def largest_component(mask: np.ndarray) -> np.ndarray:
    comps = components(mask)
    out = np.zeros_like(mask)
    if comps:
        y, x = zip(*comps[0])
        out[np.asarray(y), np.asarray(x)] = True
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros_like(mask)
    stack = []
    for x in range(w):
        if not mask[0, x]:
            stack.append((0, x))
        if not mask[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if not mask[y, 0]:
            stack.append((y, 0))
        if not mask[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if outside[y, x] or mask[y, x]:
            continue
        outside[y, x] = True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not outside[yy, xx] and not mask[yy, xx]:
                stack.append((yy, xx))
    return ~outside


def body_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = np.max(rgb, axis=2) >= 55
    white_text = (r >= 175) & (g >= 175) & (b >= 175)
    magenta_overlay = (r >= 105) & (g <= 115) & (b >= 75)
    # Keep the cyan centerline pixels in the provisional body mask. Excluding
    # them creates a dark/cyan channel through the cow and can make an interior
    # endpoint appear to be outside the body (notably cow 002). Narrow overlay
    # remnants are removed later by the tail-suppression opening.
    candidate = bright & ~white_text & ~magenta_overlay
    candidate[:70, :] = False
    # Join the color surface across the dark outline surrounding the rendered
    # centerline before selecting the largest component.
    body = erode(largest_component(dilate(candidate, 5)), 5)
    body = fill_holes(dilate(erode(dilate(body, 2), 2), 1))
    return largest_component(body)


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count: {path}")
        return np.fromfile(handle, dtype=PLY_DTYPE, count=vertex_count)


def exact_cattle_mask_and_height(row):
    source = read_ply_vertices(ROOT / row["source_top_ply"])
    extracted = read_ply_vertices(BASE / row["extracted_ply"])
    if len(source) != 512 * 424:
        raise ValueError(f"unexpected organized cloud size for cow {row['cow_id']}: {len(source)}")
    source_xyz = np.column_stack([source["x"], source["y"], source["z"]]).astype("<f4")
    extracted_xyz = np.column_stack([extracted["x"], extracted["y"], extracted["z"]]).astype("<f4")
    source_keys = np.ascontiguousarray(source_xyz).view("V12").ravel()
    extracted_keys = np.ascontiguousarray(extracted_xyz).view("V12").ravel()
    mask_sensor = np.isin(source_keys, extracted_keys).reshape(424, 512)
    plane = (
        float(row["ground_plane_a"]) * source["x"]
        + float(row["ground_plane_b"]) * source["y"]
        + float(row["ground_plane_c_m"])
    )
    height_sensor = (plane - source["z"]).reshape(424, 512)
    mask = np.zeros((474, 512), dtype=bool)
    height = np.full((474, 512), np.nan, dtype=float)
    mask[50:, :] = mask_sensor
    height[50:, :] = height_sensor
    return mask, height


def clean_height_map(old_image: Image.Image, pca_image: Image.Image, mask: np.ndarray, height: np.ndarray):
    """Rebuild a clean map from the exact point-cloud mask and learned height colors."""
    pca = np.asarray(pca_image.convert("RGB"))
    r, g, b = pca[..., 0], pca[..., 1], pca[..., 2]
    magenta = (r >= 105) & (g <= 125) & (b >= 70) & (r >= g + 25) & (b >= g + 15)
    cyan = (r <= 80) & (g >= 120) & (b >= 160)
    yellow_marker = (r >= 245) & (g >= 120) & (g <= 205) & (b <= 55)
    overlay = dilate(magenta | cyan | yellow_marker, 4)
    valid = mask & np.isfinite(height) & (np.max(pca, axis=2) > 70) & ~overlay
    valid_h = height[valid]
    valid_rgb = pca[valid]
    if len(valid_h) < 100:
        raise ValueError("insufficient clean pixels to reconstruct the height map")

    millimeters = np.rint(valid_h * 1000).astype(int)
    unique_mm = np.unique(millimeters)
    color_table = np.empty((len(unique_mm), 3), dtype=float)
    for index, value in enumerate(unique_mm):
        color_table[index] = np.median(valid_rgb[millimeters == value], axis=0)

    out = np.full_like(pca, (12, 15, 22))
    out[:50, :] = np.asarray(old_image.convert("RGB"))[:50, :]
    target_h = height[mask]
    for channel in range(3):
        out[..., channel][mask] = np.clip(
            np.interp(target_h * 1000, unique_mm, color_table[:, channel]), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(out)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([-1.0, 0.0])
    return vector / norm


def cross_section(mask: np.ndarray, point: np.ndarray, tangent: np.ndarray, max_half_width=100):
    h, w = mask.shape
    normal = np.array([-tangent[1], tangent[0]])
    ts = np.arange(-max_half_width, max_half_width + 1, dtype=float)
    xy = point[None, :] + ts[:, None] * normal[None, :]
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside = np.zeros(len(ts), dtype=bool)
    inside[valid] = mask[yi[valid], xi[valid]]
    runs = []
    start = None
    for i, value in enumerate(inside):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(inside) - 1):
            end = i if value and i == len(inside) - 1 else i - 1
            runs.append((start, end))
            start = None
    if not runs:
        return None
    center_index = max_half_width
    run = min(
        runs,
        key=lambda ab: (
            0 if ab[0] <= center_index <= ab[1] else min(abs(ab[0] - center_index), abs(ab[1] - center_index)),
            -(ab[1] - ab[0]),
        ),
    )
    lo, hi = run
    width = float(ts[hi] - ts[lo] + 1)
    midpoint = point + ((ts[lo] + ts[hi]) / 2.0) * normal
    return midpoint, width


def extend_rear(points: np.ndarray, main_body: np.ndarray):
    rear = points[0].copy()
    anchor_index = min(8, len(points) - 1)
    direction = normalize(rear - points[anchor_index])
    normal = np.array([-direction[1], direction[0]])
    extension = [rear]
    invalid_steps = 0
    lateral_offset = 0.0
    widths = []
    for step_index in range(1, 71):
        predicted = rear + direction * (2.0 * step_index) + normal * lateral_offset
        section = cross_section(main_body, predicted, direction)
        if section is None:
            invalid_steps += 1
            if invalid_steps >= 2:
                break
            continue
        section_midpoint, width = section
        lateral_correction = float(np.dot(section_midpoint - predicted, normal))
        # Keep the final part aligned with the stable tangent of the old curve.
        # Only a gradual lateral shift is allowed, so an asymmetric rump cannot
        # bend the extension back into a loop.
        lateral_offset += float(np.clip(0.35 * lateral_correction, -0.75, 0.75))
        midpoint = rear + direction * (2.0 * step_index) + normal * lateral_offset
        x, y = int(round(midpoint[0])), int(round(midpoint[1]))
        if not (0 <= y < main_body.shape[0] and 0 <= x < main_body.shape[1] and main_body[y, x]):
            break
        extension.append(midpoint.copy())
        widths.append(width)
        invalid_steps = 0
        if width <= 2.5:
            break
    if len(extension) == 1:
        return np.asarray(extension), widths
    return np.asarray(extension), widths


def tail_suppressed_body(mask: np.ndarray) -> np.ndarray:
    # About 3 cm at the current rendering scale. This removes narrow tail-like
    # protrusions while preserving the broad rear body used for endpoint tracing.
    opened = dilate(erode(mask, 6), 6)
    opened = largest_component(opened)
    return fill_holes(dilate(erode(dilate(opened, 2), 2), 1))


def draw_centerline(image: Image.Image, centerline: np.ndarray) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    coords = [tuple(map(float, point)) for point in centerline]
    if len(coords) > 1:
        draw.line(coords, fill=(7, 16, 24), width=9, joint="curve")
        draw.line(coords, fill=(25, 195, 255), width=5, joint="curve")
    for x, y in (coords[0], coords[-1]):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(7, 16, 24))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(25, 195, 255))
    return out


def main():
    PREVIEW_INDIVIDUAL.mkdir(parents=True, exist_ok=True)
    PREVIEW_COMPARISONS.mkdir(parents=True, exist_ok=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(POINTS_JSON.read_text(encoding="utf-8"))
    records = payload["records"]
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows_by_cow = {int(row["cow_id"]): row for row in csv.DictReader(handle)}
    comparison_paths = []
    metrics = []
    for record in records:
        cow_id = int(record["cow_id"])
        old_path = AXIS_DIR / "skeleton_individual" / f"cow_{cow_id:03d}_skeleton_centerline.png"
        pca_path = BASE / "qa_individual" / f"cow_{cow_id:03d}_segmentation_axis.png"
        old_image = Image.open(old_path).convert("RGB")
        pca_image = Image.open(pca_path).convert("RGB")
        points = np.asarray([[p["x_px"], p["y_px"]] for p in record["points"]], dtype=float)
        mask, height = exact_cattle_mask_and_height(rows_by_cow[cow_id])
        main_body = tail_suppressed_body(mask)
        extension, widths = extend_rear(points, main_body)
        preview = clean_height_map(old_image, pca_image, mask, height)
        full_centerline = np.vstack([extension[::-1], points[1:]])
        preview = draw_centerline(preview, full_centerline)
        preview_path = PREVIEW_INDIVIDUAL / f"cow_{cow_id:03d}_extended_centerline.png"
        preview.save(preview_path)

        comparison = Image.new("RGB", (pca_image.width + preview.width, max(pca_image.height, preview.height)), (12, 15, 22))
        comparison.paste(pca_image, (0, 0))
        comparison.paste(preview, (pca_image.width, 0))
        comparison_path = PREVIEW_COMPARISONS / f"cow_{cow_id:03d}_pca_vs_extended.png"
        comparison.save(comparison_path)
        comparison_paths.append((cow_id, comparison_path))

        length = float(np.linalg.norm(np.diff(extension, axis=0), axis=1).sum()) if len(extension) > 1 else 0.0
        metrics.append({
            "cow_id": cow_id,
            "extension_length_px": round(length, 1),
            "extension_points": len(extension),
            "final_section_width_px": round(widths[-1], 1) if widths else None,
        })

    for group_index, start in enumerate(range(0, len(comparison_paths), 10), start=1):
        group = comparison_paths[start : start + 10]
        images = [Image.open(path).convert("RGB") for _, path in group]
        canvas = Image.new("RGB", (max(im.width for im in images), sum(im.height for im in images)), (12, 15, 22))
        y = 0
        for im in images:
            canvas.paste(im, (0, y))
            y += im.height
        first_id, last_id = group[0][0], group[-1][0]
        out_path = COMBINED_DIR / f"comparison_group_{group_index:02d}_cows_{first_id:03d}_{last_id:03d}.jpg"
        canvas.save(out_path, quality=95, subsampling=0)

    lengths = [m["extension_length_px"] for m in metrics]
    print(json.dumps({
        "cows": len(metrics),
        "combined_images": math.ceil(len(metrics) / 10),
        "extension_length_px_min": min(lengths),
        "extension_length_px_median": round(float(np.median(lengths)), 1),
        "extension_length_px_max": max(lengths),
        "longest": sorted(metrics, key=lambda x: x["extension_length_px"], reverse=True)[:8],
        "not_extended": [m["cow_id"] for m in metrics if m["extension_length_px"] < 0.5],
        "note": "image preview only; no CSV or JSON record was updated",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
