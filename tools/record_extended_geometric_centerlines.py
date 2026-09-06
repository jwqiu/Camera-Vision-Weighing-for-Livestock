#!/usr/bin/env python3
"""Record the user-approved rear-extended geometric centerline for 66 cows."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
AXIS_DIR = BASE / "skeleton_axis_comparison"
RAW_RECORDS_JSON = AXIS_DIR / "geometric_centerline_records.json"
MAIN_CSV = BASE / "long_axis_records.csv"
SUMMARY_CSV = AXIS_DIR / "extended_geometric_centerline_records.csv"
POINTS_CSV = AXIS_DIR / "extended_geometric_centerline_points.csv"
RECORDS_JSON = AXIS_DIR / "extended_geometric_centerline_records.json"
METHOD_JSON = AXIS_DIR / "skeleton_method.json"
PREVIEW_SCRIPT = ROOT / "tools" / "preview_extended_centerlines.py"

METHOD_VERSION = "geometric_centerline_rear_extended_v1.0"
IMAGE_COORDINATE_SYSTEM = "overlay_image_pixel_top_left_origin_x_right_y_down"
PHYSICAL_COORDINATE_SYSTEM = "camera_point_cloud_xyz_m"


def load_preview_module():
    spec = importlib.util.spec_from_file_location("extended_preview", PREVIEW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def organized_geometry(module, row):
    source = module.read_ply_vertices(ROOT / row["source_top_ply"])
    extracted = module.read_ply_vertices(BASE / row["extracted_ply"])
    if len(source) != 512 * 424:
        raise ValueError(f"cow {row['cow_id']} does not have a 512x424 organized point cloud")
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
    mask[50:, :] = mask_sensor
    return source.reshape(424, 512), height_sensor, mask


def nearest_cattle_pixel(mask: np.ndarray, x: float, y: float):
    height, width = mask.shape
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cy < height and 0 <= cx < width and mask[cy, cx]:
        return cx, cy, 0.0
    for radius in range(1, 31):
        x0, x1 = max(0, cx - radius), min(width - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(height - 1, cy + radius)
        candidates = []
        for xx in range(x0, x1 + 1):
            candidates.append((xx, y0))
            candidates.append((xx, y1))
        for yy in range(y0 + 1, y1):
            candidates.append((x0, yy))
            candidates.append((x1, yy))
        valid = [(xx, yy) for xx, yy in candidates if mask[yy, xx]]
        if valid:
            xx, yy = min(valid, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)
            return xx, yy, math.hypot(xx - x, yy - y)
    raise ValueError(f"no cattle pixel found near ({x:.2f}, {y:.2f})")


def update_main_csv(summary_by_cow):
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    added = [
        "extended_geometric_centerline_record_type",
        "extended_geometric_centerline_method_version",
        "extended_geometric_centerline_orientation",
        "extended_geometric_centerline_point_count",
        "extended_geometric_centerline_length_m",
        "extended_geometric_centerline_direct_distance_m",
        "extended_geometric_centerline_curvature_ratio",
        "extended_geometric_centerline_rear_x_m",
        "extended_geometric_centerline_rear_y_m",
        "extended_geometric_centerline_head_x_m",
        "extended_geometric_centerline_head_y_m",
        "extended_geometric_centerline_points_file",
        "extended_geometric_centerline_preview_image",
        "extended_geometric_centerline_status",
    ]
    for header in added:
        if header not in headers:
            headers.append(header)
    for row in rows:
        summary = summary_by_cow[int(row["cow_id"])]
        row.update(
            {
                "extended_geometric_centerline_record_type": "rear_extended_curved_geometric_centerline",
                "extended_geometric_centerline_method_version": METHOD_VERSION,
                "extended_geometric_centerline_orientation": "rear_to_head",
                "extended_geometric_centerline_point_count": summary["centerline_point_count"],
                "extended_geometric_centerline_length_m": summary["polyline_length_xy_m"],
                "extended_geometric_centerline_direct_distance_m": summary["direct_distance_xy_m"],
                "extended_geometric_centerline_curvature_ratio": summary["curvature_ratio_xy"],
                "extended_geometric_centerline_rear_x_m": summary["rear_x_m"],
                "extended_geometric_centerline_rear_y_m": summary["rear_y_m"],
                "extended_geometric_centerline_head_x_m": summary["head_x_m"],
                "extended_geometric_centerline_head_y_m": summary["head_y_m"],
                "extended_geometric_centerline_points_file": str(POINTS_CSV.relative_to(ROOT)),
                "extended_geometric_centerline_preview_image": summary["preview_image"],
                "extended_geometric_centerline_status": summary["status"],
            }
        )
    write_csv(MAIN_CSV, headers, rows)


def main():
    module = load_preview_module()
    raw_payload = json.loads(RAW_RECORDS_JSON.read_text(encoding="utf-8"))
    with MAIN_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    rows_by_cow = {int(row["cow_id"]): row for row in main_rows}

    summary_rows = []
    point_rows = []
    json_records = []
    for raw_record in raw_payload["records"]:
        cow_id = int(raw_record["cow_id"])
        row = rows_by_cow[cow_id]
        source_grid, height_sensor, mask = organized_geometry(module, row)
        raw_points = np.asarray(
            [[point["x_px"], point["y_px"]] for point in raw_record["points"]], dtype=float
        )
        extension, _ = module.extend_rear(raw_points, module.tail_suppressed_body(mask))
        full_points = np.vstack([extension[::-1], raw_points[1:]])

        mapped = []
        for point_index, (image_x, image_y) in enumerate(full_points):
            pixel_x, pixel_y, mapping_distance = nearest_cattle_pixel(mask, image_x, image_y)
            sensor_y = pixel_y - 50
            vertex = source_grid[sensor_y, pixel_x]
            ground_height = float(height_sensor[sensor_y, pixel_x])
            mapped.append(
                {
                    "cow_id": cow_id,
                    "centerline_type": "rear_extended_geometric_centerline",
                    "method_version": METHOD_VERSION,
                    "orientation": "rear_to_head",
                    "point_index": point_index,
                    "image_x_px": round(float(image_x), 3),
                    "image_y_px": round(float(image_y), 3),
                    "sensor_u_px": pixel_x,
                    "sensor_v_px": sensor_y,
                    "pixel_mapping_distance_px": round(mapping_distance, 3),
                    "camera_x_m": round(float(vertex["x"]), 6),
                    "camera_y_m": round(float(vertex["y"]), 6),
                    "camera_z_m": round(float(vertex["z"]), 6),
                    "height_above_ground_m": round(ground_height, 6),
                }
            )

        xy = np.asarray([[point["camera_x_m"], point["camera_y_m"]] for point in mapped])
        segments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        for point, arc_length in zip(mapped, cumulative):
            point["cumulative_xy_length_m"] = round(float(arc_length), 6)
            point_rows.append(point)
        polyline_length = float(cumulative[-1])
        direct_distance = float(np.linalg.norm(xy[-1] - xy[0]))
        mapping_distances = [point["pixel_mapping_distance_px"] for point in mapped]
        summary = {
            "cow_id": cow_id,
            "record_type": "rear_extended_curved_geometric_centerline",
            "method_version": METHOD_VERSION,
            "source_raw_centerline_method_version": raw_payload["record_version"],
            "orientation": "rear_to_head",
            "image_coordinate_system": IMAGE_COORDINATE_SYSTEM,
            "physical_coordinate_system": PHYSICAL_COORDINATE_SYSTEM,
            "centerline_point_count": len(mapped),
            "added_rear_extension_point_count": max(0, len(extension) - 1),
            "polyline_length_xy_m": round(polyline_length, 6),
            "direct_distance_xy_m": round(direct_distance, 6),
            "curvature_ratio_xy": round(polyline_length / direct_distance, 6) if direct_distance else "",
            "rear_x_m": mapped[0]["camera_x_m"],
            "rear_y_m": mapped[0]["camera_y_m"],
            "head_x_m": mapped[-1]["camera_x_m"],
            "head_y_m": mapped[-1]["camera_y_m"],
            "maximum_pixel_mapping_distance_px": round(max(mapping_distances), 3),
            "mean_pixel_mapping_distance_px": round(float(np.mean(mapping_distances)), 3),
            "points_file": str(POINTS_CSV.relative_to(ROOT)),
            "preview_image": str(
                (
                    AXIS_DIR
                    / "extended_preview"
                    / "skeleton_individual"
                    / f"cow_{cow_id:03d}_extended_centerline.png"
                ).relative_to(ROOT)
            ),
            "status": "ok" if max(mapping_distances) <= 5 else "review_required",
        }
        summary_rows.append(summary)
        json_records.append({**summary, "points": mapped})

    write_csv(SUMMARY_CSV, list(summary_rows[0].keys()), summary_rows)
    write_csv(POINTS_CSV, list(point_rows[0].keys()), point_rows)
    payload = {
        "record_version": METHOD_VERSION,
        "description": "User-approved geometric centerline with the rear extended to the tail-suppressed main-body contour.",
        "source_raw_centerline_records": str(RAW_RECORDS_JSON.relative_to(ROOT)),
        "pca_axis_records": str(MAIN_CSV.relative_to(ROOT)),
        "image_coordinate_system": IMAGE_COORDINATE_SYSTEM,
        "physical_coordinate_system": PHYSICAL_COORDINATE_SYSTEM,
        "orientation": "rear_to_head",
        "records": json_records,
    }
    RECORDS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_main_csv({int(row["cow_id"]): row for row in summary_rows})

    method = json.loads(METHOD_JSON.read_text(encoding="utf-8"))
    method["extended_coordinate_records"] = {
        "record_version": METHOD_VERSION,
        "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
        "points_csv": str(POINTS_CSV.relative_to(ROOT)),
        "records_json": str(RECORDS_JSON.relative_to(ROOT)),
        "image_coordinate_system": IMAGE_COORDINATE_SYSTEM,
        "physical_coordinate_system": PHYSICAL_COORDINATE_SYSTEM,
        "orientation": "rear_to_head",
        "record_policy": "PCA axis and original unextended skeleton centerline records are preserved separately.",
    }
    METHOD_JSON.write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "cows": len(summary_rows),
                "points": len(point_rows),
                "ok": sum(row["status"] == "ok" for row in summary_rows),
                "review_required": [row["cow_id"] for row in summary_rows if row["status"] != "ok"],
                "summary_csv": str(SUMMARY_CSV),
                "points_csv": str(POINTS_CSV),
                "records_json": str(RECORDS_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
