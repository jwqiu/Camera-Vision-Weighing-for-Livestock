#!/usr/bin/env python3
"""Record the already-rendered geometric centerlines without altering PCA data."""

from __future__ import annotations

import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
SKELETON_DIR = BASE / "skeleton_axis_comparison"
METHOD_JSON = SKELETON_DIR / "skeleton_method.json"
SUMMARY_CSV = SKELETON_DIR / "geometric_centerline_records.csv"
POINTS_CSV = SKELETON_DIR / "geometric_centerline_points.csv"
RECORDS_JSON = SKELETON_DIR / "geometric_centerline_records.json"
MAIN_CSV = BASE / "long_axis_records.csv"

METHOD_VERSION = "qa_rendered_geometric_centerline_record_v1.0"
COORDINATE_SYSTEM = "overlay_image_pixel_top_left_origin_x_right_y_down"


def zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a one-pixel skeleton."""
    img = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(img, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            s = (
                (p2 == 0) & (p3 == 1)
                | (p3 == 0) & (p4 == 1)
                | (p4 == 0) & (p5 == 1)
                | (p5 == 0) & (p6 == 1)
                | (p6 == 0) & (p7 == 1)
                | (p7 == 0) & (p8 == 1)
                | (p8 == 0) & (p9 == 1)
                | (p9 == 0) & (p2 == 1)
            ).astype(np.uint8)
            if phase == 0:
                structural = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                structural = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (n >= 2) & (n <= 6) & (s == 1) & structural
            if np.any(remove):
                img[remove] = 0
                changed = True
    return img.astype(bool)


def graph_from_skeleton(skeleton: np.ndarray):
    points = [tuple(p) for p in np.argwhere(skeleton)]
    point_set = set(points)
    graph = {p: [] for p in points}
    for y, x in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (y + dy, x + dx)
                if q in point_set:
                    graph[(y, x)].append((q, math.hypot(dx, dy)))
    return graph


def connected_components(mask: np.ndarray):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for seed_y, seed_x in zip(*np.where(mask)):
        if visited[seed_y, seed_x]:
            continue
        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < height
                        and 0 <= xx < width
                        and mask[yy, xx]
                        and not visited[yy, xx]
                    ):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
        components.append(component)
    return sorted(components, key=len, reverse=True)


def dijkstra_farthest(graph, start):
    dist = {start: 0.0}
    prev = {}
    queue = [(0.0, start)]
    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist != dist.get(node):
            continue
        for neighbor, cost in graph[node]:
            candidate = current_dist + cost
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                prev[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    farthest = max(dist, key=dist.get)
    return farthest, dist, prev


def diameter_path(graph):
    if not graph:
        raise ValueError("empty skeleton graph")
    seed = next(iter(graph))
    end_a, _, _ = dijkstra_farthest(graph, seed)
    end_b, distances, previous = dijkstra_farthest(graph, end_a)
    path = [end_b]
    while path[-1] != end_a:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[end_b]


def resample_path(path, count):
    xy = np.array([(x, y) for y, x in path], dtype=float)
    if xy[0, 0] > xy[-1, 0]:
        xy = xy[::-1]
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] == 0:
        return np.repeat(xy[:1], count, axis=0)
    target = np.linspace(0.0, cumulative[-1], count)
    out = np.column_stack(
        [np.interp(target, cumulative, xy[:, axis]) for axis in range(2)]
    )
    return out


def extract_rendered_centerline(image_path: Path, target_count: int):
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    # The existing overlay uses a solid cyan fill (25, 195, 255). Include only
    # its tightly antialiased edge colors so the colored height map is excluded.
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cyan = (r <= 35) & (g >= 175) & (g <= 205) & (b >= 230)
    components = connected_components(cyan)
    if not components:
        raise ValueError(f"no rendered centerline found in {image_path}")
    line_mask = np.zeros_like(cyan)
    line_y, line_x = zip(*components[0])
    line_mask[np.array(line_y), np.array(line_x)] = True
    # Trace the already-rendered strip itself. Its shortest geodesic between the
    # two farthest ends follows the middle of the narrow cyan stroke and avoids
    # thinning artifacts at sharp turns.
    graph = graph_from_skeleton(line_mask)
    path, _ = diameter_path(graph)
    path_xy = np.array([(x, y) for y, x in path], dtype=float)

    # Endpoint markers are separated from the cyan line by a dark outline.
    # Add their centroids when they sit next to either end of the main line.
    marker_centers = []
    for component in components[1:]:
        if not 20 <= len(component) <= 200:
            continue
        comp = np.asarray(component, dtype=float)
        y_span = comp[:, 0].max() - comp[:, 0].min()
        x_span = comp[:, 1].max() - comp[:, 1].min()
        if x_span <= 20 and y_span <= 20:
            marker_centers.append(np.array([comp[:, 1].mean(), comp[:, 0].mean()]))
    for center in marker_centers:
        distances = [np.linalg.norm(center - path_xy[0]), np.linalg.norm(center - path_xy[-1])]
        endpoint = int(np.argmin(distances))
        if distances[endpoint] <= 20:
            if endpoint == 0:
                path_xy = np.vstack([center, path_xy])
            else:
                path_xy = np.vstack([path_xy, center])

    path = [(int(round(y)), int(round(x))) for x, y in path_xy]
    sampled = resample_path(path, target_count)
    polyline_length = float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum())
    direct_distance = float(np.linalg.norm(sampled[-1] - sampled[0]))
    height, width = rgb.shape[:2]
    return sampled, width, height, polyline_length, direct_distance


def write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def update_main_csv(summary_by_cow):
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    added_headers = [
        "pca_axis_record_type",
        "pca_axis_method_version",
        "geometric_centerline_record_type",
        "geometric_centerline_method_version",
        "geometric_centerline_coordinate_system",
        "geometric_centerline_point_count",
        "geometric_centerline_polyline_length_px",
        "geometric_centerline_direct_distance_px",
        "geometric_centerline_curvature_ratio",
        "geometric_centerline_points_file",
    ]
    for header in added_headers:
        if header not in headers:
            headers.append(header)

    for row in rows:
        cow_id = int(row["cow_id"])
        summary = summary_by_cow.get(cow_id)
        row["pca_axis_record_type"] = "straight_pca_axis"
        row["pca_axis_method_version"] = row.get("method_version", "")
        if summary:
            row["geometric_centerline_record_type"] = "curved_geometric_centerline"
            row["geometric_centerline_method_version"] = summary["method_version"]
            row["geometric_centerline_coordinate_system"] = summary["coordinate_system"]
            row["geometric_centerline_point_count"] = summary["centerline_point_count"]
            row["geometric_centerline_polyline_length_px"] = summary["polyline_length_px"]
            row["geometric_centerline_direct_distance_px"] = summary["direct_distance_px"]
            row["geometric_centerline_curvature_ratio"] = summary["curvature_ratio"]
            row["geometric_centerline_points_file"] = str(
                POINTS_CSV.relative_to(ROOT)
            )

    write_csv(MAIN_CSV, headers, rows)


def main():
    with METHOD_JSON.open("r", encoding="utf-8") as handle:
        method = json.load(handle)

    summary_rows = []
    point_rows = []
    json_records = []

    for record in method["records"]:
        cow_id = int(record["cow_id"])
        image_path = ROOT / record["skeleton_image"]
        target_count = int(record["centerline_points"])
        sampled, width, height, length, direct = extract_rendered_centerline(
            image_path, target_count
        )
        curvature = length / direct if direct else None
        summary = {
            "cow_id": cow_id,
            "record_type": "curved_geometric_centerline",
            "method_version": METHOD_VERSION,
            "source_centerline_method_version": method["method_version"],
            "coordinate_system": COORDINATE_SYSTEM,
            "source_image_width_px": width,
            "source_image_height_px": height,
            "centerline_point_count": target_count,
            "start_x_px": round(float(sampled[0, 0]), 3),
            "start_y_px": round(float(sampled[0, 1]), 3),
            "end_x_px": round(float(sampled[-1, 0]), 3),
            "end_y_px": round(float(sampled[-1, 1]), 3),
            "polyline_length_px": round(length, 3),
            "direct_distance_px": round(direct, 3),
            "curvature_ratio": round(curvature, 6) if curvature is not None else "",
            "source_skeleton_image": record["skeleton_image"],
            "points_file": str(POINTS_CSV.relative_to(ROOT)),
        }
        summary_rows.append(summary)

        points = []
        for index, (x, y) in enumerate(sampled):
            point = {
                "cow_id": cow_id,
                "centerline_type": "geometric_skeleton",
                "method_version": METHOD_VERSION,
                "point_index": index,
                "x_px": round(float(x), 3),
                "y_px": round(float(y), 3),
                "x_normalized": round(float(x / (width - 1)), 8),
                "y_normalized": round(float(y / (height - 1)), 8),
                "source_image_width_px": width,
                "source_image_height_px": height,
            }
            point_rows.append(point)
            points.append({"point_index": index, "x_px": point["x_px"], "y_px": point["y_px"]})
        json_records.append({**summary, "points": points})

    summary_headers = list(summary_rows[0].keys())
    point_headers = list(point_rows[0].keys())
    write_csv(SUMMARY_CSV, summary_headers, summary_rows)
    write_csv(POINTS_CSV, point_headers, point_rows)

    payload = {
        "record_version": METHOD_VERSION,
        "description": "Coordinates recovered from the already-generated geometric centerline overlays; PCA axis records remain separate.",
        "coordinate_system": COORDINATE_SYSTEM,
        "orientation": "point order is image-left to image-right; it does not assert anatomical head/tail direction",
        "records": json_records,
    }
    with RECORDS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    method["coordinate_records"] = {
        "record_version": METHOD_VERSION,
        "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
        "points_csv": str(POINTS_CSV.relative_to(ROOT)),
        "records_json": str(RECORDS_JSON.relative_to(ROOT)),
        "coordinate_system": COORDINATE_SYSTEM,
        "pca_records_policy": "preserved separately in cattle_3d_extraction_66/long_axis_records.csv",
    }
    for record in method["records"]:
        record["centerline_coordinates_recorded"] = True
        record["centerline_points_file"] = str(POINTS_CSV.relative_to(ROOT))
    with METHOD_JSON.open("w", encoding="utf-8") as handle:
        json.dump(method, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    update_main_csv({int(row["cow_id"]): row for row in summary_rows})
    print(
        json.dumps(
            {
                "cows": len(summary_rows),
                "points": len(point_rows),
                "summary_csv": str(SUMMARY_CSV),
                "points_csv": str(POINTS_CSV),
                "records_json": str(RECORDS_JSON),
                "main_csv": str(MAIN_CSV),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
