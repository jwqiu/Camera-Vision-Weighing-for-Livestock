#!/usr/bin/env python3
"""Compute a robust median body-depth measurement from right-view cattle PLY files."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 424


def fit_ground_plane(xyz: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Fit y = a*x + b*z + c to the visible floor using bottom image rows."""
    height, width, _ = xyz.shape
    yy, xx = np.mgrid[:height, :width]
    valid = np.isfinite(xyz).all(axis=2) & (np.linalg.norm(xyz, axis=2) > 0)
    candidate = valid & (yy >= int(height * 0.76)) & (xyz[:, :, 2] < 6.5)
    points = xyz[candidate]
    if len(points) < 300:
        raise ValueError("Too few candidate ground points")
    if len(points) > 15000:
        points = points[rng.choice(len(points), 15000, replace=False)]
    design = np.column_stack((points[:, 0], points[:, 2], np.ones(len(points))))
    best_coef = None
    best_count = -1
    for _ in range(500):
        sample = rng.choice(len(points), 3, replace=False)
        try:
            coef = np.linalg.solve(design[sample], points[sample, 1])
        except np.linalg.LinAlgError:
            continue
        if abs(coef[0]) > 0.4 or abs(coef[1]) > 1.2:
            continue
        residual = np.abs(points[:, 1] - design @ coef) / math.sqrt(1 + coef[0] ** 2 + coef[1] ** 2)
        count = int(np.count_nonzero(residual < 0.025))
        if count > best_count:
            best_count = count
            best_coef = coef
    if best_coef is None:
        raise ValueError("Unable to fit ground plane")
    residual = np.abs(points[:, 1] - design @ best_coef) / math.sqrt(1 + best_coef[0] ** 2 + best_coef[1] ** 2)
    inliers = residual < 0.035
    best_coef, *_ = np.linalg.lstsq(design[inliers], points[inliers, 1], rcond=None)
    inlier_fraction = float(np.mean(inliers))
    return best_coef, inlier_fraction


def label_components(mask: np.ndarray, xyz: np.ndarray, threshold: float = 0.075) -> list[np.ndarray]:
    height, width = mask.shape
    parent = np.arange(height * width, dtype=np.int32)
    size = np.ones(height * width, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            here = y * width + x
            if x and mask[y, x - 1] and np.linalg.norm(xyz[y, x] - xyz[y, x - 1]) < threshold:
                union(here, here - 1)
            if y and mask[y - 1, x] and np.linalg.norm(xyz[y, x] - xyz[y - 1, x]) < threshold:
                union(here, here - width)
    groups: dict[int, list[int]] = {}
    for flat in np.flatnonzero(mask):
        root = find(int(flat))
        groups.setdefault(root, []).append(int(flat))
    return [np.asarray(group, dtype=np.int32) for group in sorted(groups.values(), key=len, reverse=True)]


def segment_cow(xyz: np.ndarray, ground_coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(xyz).all(axis=2) & (np.linalg.norm(xyz, axis=2) > 0)
    plane_y = ground_coef[0] * xyz[:, :, 0] + ground_coef[1] * xyz[:, :, 2] + ground_coef[2]
    height_above_ground = (xyz[:, :, 1] - plane_y) / math.sqrt(1 + ground_coef[0] ** 2 + ground_coef[1] ** 2)
    candidate = valid & (height_above_ground > 0.10) & (xyz[:, :, 2] < 4.5)
    components = label_components(candidate, xyz)
    plausible = []
    for component in components[:30]:
        rows, cols = np.unravel_index(component, candidate.shape)
        if len(component) < 1500:
            continue
        bbox_w = cols.max() - cols.min() + 1
        bbox_h = rows.max() - rows.min() + 1
        med_z = float(np.median(xyz[rows, cols, 2]))
        if bbox_w >= 120 and bbox_h >= 60 and med_z < 3.5:
            plausible.append((len(component) * bbox_w / max(bbox_h, 1), component))
    if not plausible:
        raise ValueError("No plausible cow component")
    component = max(plausible, key=lambda item: item[0])[1]
    cow_mask = np.zeros(candidate.shape, dtype=bool)
    cow_mask.flat[component] = True
    return cow_mask, height_above_ground


def rolling_quantile(values: np.ndarray, radius: int, quantile: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        window = values[max(0, index - radius): min(len(values), index + radius + 1)]
        window = window[np.isfinite(window)]
        if len(window):
            result[index] = np.quantile(window, quantile)
    return result


def robust_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(5):
        coef = np.polyfit(x[keep], y[keep], 1)
        residual = y - np.polyval(coef, x)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        if mad < 1e-6:
            break
        updated = keep & (np.abs(residual - center) < 3.5 * 1.4826 * mad)
        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 10:
            break
        keep = updated
    return float(coef[0]), float(coef[1])


def profile_bounds(u: np.ndarray, v: np.ndarray, bin_size: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low, high = np.quantile(u, [0.005, 0.995])
    edges = np.arange(math.floor(low / bin_size) * bin_size,
                      math.ceil(high / bin_size) * bin_size + bin_size * 1.01,
                      bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.clip(np.digitize(u, edges) - 1, 0, len(centers) - 1)
    back = np.full(len(centers), np.nan)
    belly = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for index in range(len(centers)):
        column = v[indices == index]
        counts[index] = len(column)
        if len(column) >= 12:
            back[index] = np.quantile(column, 0.98)
            belly[index] = np.quantile(column, 0.02)
    return centers, back, belly, counts


def compute_body_depth(path: Path) -> dict[str, object]:
    xyz = read_ply_xyz(path).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    rng = np.random.default_rng(20260906 + int(path.parent.name))
    ground_coef, ground_fraction = fit_ground_plane(xyz, rng)
    cow_mask, height = segment_cow(xyz, ground_coef)
    x = xyz[:, :, 0][cow_mask].astype(float)
    h = height[cow_mask].astype(float)

    centers0, back0, _, _ = profile_bounds(x, h)
    q0, q1 = np.quantile(x, [0.15, 0.85])
    fit_mask = np.isfinite(back0) & (centers0 >= q0) & (centers0 <= q1)
    slope, intercept = robust_line_fit(centers0[fit_mask], back0[fit_mask])
    angle = math.atan(slope)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = cos_a * x + sin_a * h
    v = -sin_a * x + cos_a * h

    centers, raw_back, raw_belly, counts = profile_bounds(u, v)
    smooth_back = rolling_quantile(raw_back, radius=3, quantile=0.5)
    # A high rolling quantile follows the broad abdominal contour while lifting
    # narrow downward excursions caused by legs and the udder.
    smooth_belly = rolling_quantile(raw_belly, radius=20, quantile=0.90)
    smooth_belly = rolling_quantile(smooth_belly, radius=3, quantile=0.5)
    body_low, body_high = np.quantile(u, [0.005, 0.995])
    central_low = body_low + 0.40 * (body_high - body_low)
    central_high = body_low + 0.60 * (body_high - body_low)
    valid = (
        np.isfinite(smooth_back) & np.isfinite(smooth_belly)
        & (centers >= central_low) & (centers <= central_high)
        & (counts >= 12)
    )
    depths = smooth_back[valid] - smooth_belly[valid]
    depths = depths[(depths > 0.25) & (depths < 1.25)]
    if len(depths) < 12:
        raise ValueError(f"Too few valid depth slices ({len(depths)})")
    result: dict[str, object] = {
        "rightview_median_body_depth_m": float(np.median(depths)),
        "rightview_body_depth_q25_m": float(np.quantile(depths, 0.25)),
        "rightview_body_depth_q75_m": float(np.quantile(depths, 0.75)),
        "rightview_valid_slice_count": int(len(depths)),
        "rightview_central_region_start_ratio": 0.40,
        "rightview_central_region_end_ratio": 0.60,
        "rightview_profile_bin_m": 0.01,
        "rightview_belly_correction_window_m": 0.41,
        "rightview_belly_correction_quantile": 0.90,
        "rightview_horizontal_rotation_deg": float(math.degrees(angle)),
        "rightview_ground_fit_inlier_fraction": ground_fraction,
        "rightview_source_ply": str(path),
        "rightview_body_depth_method_version": "rightview_central40_60_robust_back_belly_v1.0",
        "rightview_body_depth_status": "ok",
        "rightview_body_depth_recorded_on": "2026-09-06",
        "_u": u,
        "_v": v,
        "_x": x,
        "_height_above_ground": h,
        "_centers": centers,
        "_back": smooth_back,
        "_belly": smooth_belly,
        "_counts": counts,
        "_central_low": central_low,
        "_central_high": central_high,
    }
    return result


def render_depth_qa(path: Path, result: dict[str, object], output: Path) -> None:
    u = np.asarray(result["_u"]); v = np.asarray(result["_v"])
    centers = np.asarray(result["_centers"])
    back = np.asarray(result["_back"]); belly = np.asarray(result["_belly"])
    central_low = float(result["_central_low"]); central_high = float(result["_central_high"])
    width, height = 1000, 520
    margin = 45
    u0, u1 = np.quantile(u, [0.002, 0.998]); v0, v1 = np.quantile(v, [0.002, 0.998])
    def px(uu: np.ndarray | float) -> np.ndarray:
        return margin + (np.asarray(uu) - u0) / (u1 - u0) * (width - 2 * margin)
    def py(vv: np.ndarray | float) -> np.ndarray:
        return height - margin - (np.asarray(vv) - v0) / (v1 - v0) * (height - 2 * margin)
    canvas = Image.new("RGB", (width, height), (8, 13, 21))
    draw = ImageDraw.Draw(canvas)
    sample = np.arange(0, len(u), max(1, len(u) // 45000))
    for xx, yy in zip(px(u[sample]).astype(int), py(v[sample]).astype(int)):
        if 0 <= xx < width and 0 <= yy < height:
            draw.point((int(xx), int(yy)), fill=(75, 90, 110))
    draw.rectangle((int(px(central_low)), margin, int(px(central_high)), height - margin), outline=(255, 205, 65), width=3)
    valid = np.isfinite(back)
    draw.line(list(zip(px(centers[valid]).astype(int), py(back[valid]).astype(int))), fill=(65, 210, 255), width=4)
    valid = np.isfinite(belly)
    draw.line(list(zip(px(centers[valid]).astype(int), py(belly[valid]).astype(int))), fill=(255, 120, 90), width=4)
    draw.text((margin, 12), f"median body depth = {float(result['rightview_median_body_depth_m']):.3f} m", fill=(240, 245, 250))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def inspect_segmentation(path: Path, output: Path) -> None:
    xyz = read_ply_xyz(path).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    coef, fraction = fit_ground_plane(xyz, np.random.default_rng(20260906))
    cow_mask, height = segment_cow(xyz, coef)
    rows, cols = np.where(cow_mask)
    print("ground_coef", coef, "inlier_fraction", fraction)
    print("cow", len(rows), "bbox", (cols.min(), rows.min(), cols.max(), rows.max()),
          "height_quantiles", np.quantile(height[cow_mask], [0, .05, .5, .95, 1]),
          "z_median", np.median(xyz[:, :, 2][cow_mask]))
    canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (8, 13, 21))
    pixels = np.asarray(canvas).copy()
    normalized = np.clip((height - 0.1) / 1.5, 0, 1)
    pixels[cow_mask, 0] = (50 + 205 * normalized[cow_mask]).astype(np.uint8)
    pixels[cow_mask, 1] = (190 - 80 * normalized[cow_mask]).astype(np.uint8)
    pixels[cow_mask, 2] = (245 - 160 * normalized[cow_mask]).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)


def inspect_depth(path: Path, output: Path) -> None:
    result = compute_body_depth(path)
    for key, value in result.items():
        if not key.startswith("_"):
            print(key, value)
    render_depth_qa(path, result, output)


OUTPUT_FIELDS = [
    "rightview_median_body_depth_m",
    "rightview_body_depth_q25_m",
    "rightview_body_depth_q75_m",
    "rightview_valid_slice_count",
    "rightview_central_region_start_ratio",
    "rightview_central_region_end_ratio",
    "rightview_profile_bin_m",
    "rightview_belly_correction_window_m",
    "rightview_belly_correction_quantile",
    "rightview_horizontal_rotation_deg",
    "rightview_ground_fit_inlier_fraction",
    "rightview_source_ply",
    "rightview_body_depth_method_version",
    "rightview_body_depth_status",
    "rightview_body_depth_recorded_on",
]


def batch_compute(records_csv: Path, dataset_root: Path, output_csv: Path, qa_dir: Path | None) -> None:
    with records_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    target_ids = [
        int(row["cow_id"])
        for row in records
        if row.get("analysis_data_quality_code") == "qualified"
        and row.get("pca_full_torso_analysis_eligible", "").upper() == "TRUE"
    ]
    rows = []
    for position, cow_id in enumerate(target_ids, 1):
        sources = sorted((dataset_root / str(cow_id)).glob("right-*.ply"))
        if len(sources) != 1:
            raise ValueError(f"Cow {cow_id}: expected one right PLY, found {len(sources)}")
        result = compute_body_depth(sources[0])
        row = {"cow_id": cow_id}
        row.update({field: result[field] for field in OUTPUT_FIELDS})
        rows.append(row)
        if qa_dir is not None:
            render_depth_qa(sources[0], result, qa_dir / f"cow_{cow_id:03d}_rightview_body_depth.png")
        print(f"[{position:02d}/{len(target_ids)}] cow {cow_id:03d}: {float(result['rightview_median_body_depth_m']):.3f} m", flush=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cow_id", *OUTPUT_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    values = np.array([float(row["rightview_median_body_depth_m"]) for row in rows])
    print("summary", len(rows), np.quantile(values, [0, .25, .5, .75, 1]), flush=True)


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing end_header in {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded.startswith("element vertex "):
                vertex_count = int(decoded.split()[-1])
            if decoded == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")
        dtype = np.dtype(
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
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))


def inspect(path: Path) -> None:
    xyz = read_ply_xyz(path)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0)
    xyz = xyz[valid]
    print(path)
    print("valid", len(xyz))
    for i, axis in enumerate("xyz"):
        print(axis, np.quantile(xyz[:, i], [0, .01, .1, .25, .5, .75, .9, .99, 1]))


def inspect_components(path: Path, output: Path, width: int = 512, height: int = 424) -> None:
    xyz = read_ply_xyz(path).reshape(height, width, 3)
    valid = np.isfinite(xyz).all(axis=2) & (np.linalg.norm(xyz, axis=2) > 0)
    parent = np.arange(height * width, dtype=np.int32)
    size = np.ones(height * width, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    threshold = 0.055
    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                continue
            here = y * width + x
            if x and valid[y, x - 1] and np.linalg.norm(xyz[y, x] - xyz[y, x - 1]) < threshold:
                union(here, here - 1)
            if y and valid[y - 1, x] and np.linalg.norm(xyz[y, x] - xyz[y - 1, x]) < threshold:
                union(here, here - width)

    groups: dict[int, list[tuple[int, int]]] = {}
    for y, x in zip(*np.where(valid)):
        root = find(int(y * width + x))
        groups.setdefault(root, []).append((int(y), int(x)))
    ranked = sorted(groups.values(), key=len, reverse=True)
    canvas = Image.new("RGB", (width, height), (8, 13, 21))
    pix = canvas.load()
    palette = [(80, 200, 120), (240, 170, 50), (70, 170, 245), (225, 90, 120), (180, 120, 245)]
    for rank, points in enumerate(ranked[:5]):
        ys = np.array([p[0] for p in points]); xs = np.array([p[1] for p in points])
        values = xyz[ys, xs]
        print(rank + 1, len(points), "bbox", (xs.min(), ys.min(), xs.max(), ys.max()),
              "xyz_med", np.median(values, axis=0))
        color = palette[rank]
        for yy, xx in points:
            pix[xx, yy] = color
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--inspect-components", type=Path)
    parser.add_argument("--inspect-segmentation", type=Path)
    parser.add_argument("--inspect-depth", type=Path)
    parser.add_argument("--records-csv", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--batch-output", type=Path)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/right_components.png"))
    args = parser.parse_args()
    if args.inspect:
        inspect(args.inspect)
    if args.inspect_components:
        inspect_components(args.inspect_components, args.output)
    if args.inspect_segmentation:
        inspect_segmentation(args.inspect_segmentation, args.output)
    if args.inspect_depth:
        inspect_depth(args.inspect_depth, args.output)
    if args.batch_output:
        if not args.records_csv:
            parser.error("--records-csv is required with --batch-output")
        batch_compute(args.records_csv, args.dataset_root, args.batch_output, args.qa_dir)


if __name__ == "__main__":
    main()
