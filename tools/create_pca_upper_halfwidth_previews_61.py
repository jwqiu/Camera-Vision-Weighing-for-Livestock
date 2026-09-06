#!/usr/bin/env python3
"""Locate upper- and lower-contour boundaries with three ordered rules."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cattle_3d_extraction_66"
OUTPUT_DIR = BASE / "pca_horizontal_60pct_upper_halfwidth_preview_61"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
GROUP_DIR = OUTPUT_DIR / "groups_2cols_10cows"
RECORDS_CSV = OUTPUT_DIR / "pca_upper_halfwidth_records_61.csv"
SUMMARY_JSON = OUTPUT_DIR / "pca_upper_halfwidth_preview_summary.json"

PROFILE_BIN_M = 0.01
PROFILE_SMOOTHING_BINS = 9
UPPER_EDGE_QUANTILE = 0.98
SEARCH_LEFT_RATIO = 0.45
SEARCH_RIGHT_RATIO = 0.70
VALLEY_LOCAL_RADIUS_BINS = 4
VALLEY_PLATEAU_RADIUS_BINS = 6
VALLEY_PLATEAU_TOLERANCE_M = 0.0025
VALLEY_SHOULDER_OFFSET_BINS = 4
VALLEY_SHOULDER_WIDTH_BINS = 9
VALLEY_MIN_PROMINENCE_M = 0.003
GROUP_SIZE = 10
GROUP_COLUMNS = 2
HALFWIDTH_COLOR = (255, 74, 40)
HALFWIDTH_OUTLINE = (40, 8, 4)
SLOPE_CHANGE_COLOR = (75, 235, 120)
FAST_TO_FLAT_COLOR = (206, 112, 255)
NO_MATCH_COLOR = (185, 190, 198)
RULE2_MIN_SEGMENT_BINS = 6
RULE2_MIN_SLOPE_INCREASE = 0.10
RULE2_MIN_SLOPE_RATIO = 1.8
RULE2_MIN_FIT_IMPROVEMENT = 0.25
RULE3_SUPPORT_LEFT_RATIO = 0.35
RULE3_MIN_PEAK_TO_VALLEY_DROP_M = 0.005
RULE3_MIN_LEFT_CONFIRMATION_RISE_M = 0.002
PLATEAU_MIN_SEGMENT_BINS = 4
PLATEAU_MAX_ABS_SLOPE = 0.06
PLATEAU_MIN_SIDE_SLOPE = 0.10
PLATEAU_MIN_FIT_IMPROVEMENT = 0.40

sys.path.insert(0, str(ROOT / "tools"))
from create_pca_horizontal_60pct_previews_61 import (  # noqa: E402
    BACKGROUND,
    LABEL_FONT,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    PLOT_BOTTOM,
    PLOT_LEFT,
    PLOT_RIGHT,
    PLOT_TOP,
    SEPARATOR,
    make_panel,
    read_rows,
)
from preview_pca_full_torso_boundaries import read_ply_vertices  # noqa: E402


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        local = values[max(0, index - half) : min(len(values), index + half + 1)]
        finite = local[np.isfinite(local)]
        if len(finite):
            result[index] = float(np.median(finite))
    return result


def add_three_rule_boundary(
    row: dict[str, str],
    panel: Image.Image,
    base_record: dict[str, object],
    side: str = "upper",
) -> dict[str, object]:
    if side not in {"upper", "lower"}:
        raise ValueError(f"unsupported side: {side}")
    vertices = read_ply_vertices(BASE / row["extracted_ply"])
    points_xy = np.column_stack([vertices["x"], vertices["y"]]).astype(float)
    rear_xy = np.asarray([float(row["long_axis_start_x_m"]), float(row["long_axis_start_y_m"])])
    head_xy = np.asarray([float(row["long_axis_end_x_m"]), float(row["long_axis_end_y_m"])])
    axis = head_xy - rear_xy
    full_length = float(np.linalg.norm(axis))
    axis /= full_length
    normal = np.asarray([-axis[1], axis[0]])
    relative = points_xy - rear_xy
    longitudinal = relative @ axis
    lateral = relative @ normal
    side_sign = 1.0 if side == "upper" else -1.0
    side_lateral = side_sign * lateral

    first_station = math.ceil(SEARCH_LEFT_RATIO * full_length / PROFILE_BIN_M) * PROFILE_BIN_M
    last_station = math.floor(SEARCH_RIGHT_RATIO * full_length / PROFILE_BIN_M) * PROFILE_BIN_M
    stations = np.arange(first_station, last_station + PROFILE_BIN_M * 0.5, PROFILE_BIN_M)
    raw_upper = np.full(len(stations), np.nan)
    point_counts = np.zeros(len(stations), dtype=int)
    for index, station in enumerate(stations):
        selected = np.abs(longitudinal - station) <= PROFILE_BIN_M / 2
        values = side_lateral[selected]
        point_counts[index] = len(values)
        if len(values) >= 8:
            edge = float(np.quantile(values, UPPER_EDGE_QUANTILE))
            if edge > 0:
                raw_upper[index] = edge
    smooth_upper = rolling_median(raw_upper, PROFILE_SMOOTHING_BINS)
    if not np.any(np.isfinite(smooth_upper)):
        raise ValueError(f"cow {row['cow_id']}: no valid upper width profile")

    valley_index: int | None = None
    valley_prominence = 0.0
    shoulder_span = VALLEY_SHOULDER_OFFSET_BINS + VALLEY_SHOULDER_WIDTH_BINS
    # Iterate from right to left. Wider shoulder windows recognize broad, flat
    # valleys instead of requiring a sharp single-point local minimum.
    for index in range(
        len(stations) - VALLEY_LOCAL_RADIUS_BINS - 1,
        VALLEY_LOCAL_RADIUS_BINS,
        -1,
    ):
        local = smooth_upper[
            index - VALLEY_LOCAL_RADIUS_BINS : index + VALLEY_LOCAL_RADIUS_BINS + 1
        ]
        left_shoulder = smooth_upper[
            max(0, index - shoulder_span) : index - VALLEY_SHOULDER_OFFSET_BINS
        ]
        right_shoulder = smooth_upper[
            index + VALLEY_SHOULDER_OFFSET_BINS : min(len(smooth_upper), index + shoulder_span)
        ]
        if not (
            np.isfinite(smooth_upper[index])
            and np.all(np.isfinite(local))
            and np.count_nonzero(np.isfinite(left_shoulder)) >= 3
            and np.count_nonzero(np.isfinite(right_shoulder)) >= 3
        ):
            continue
        width = float(smooth_upper[index])
        prominence = min(
            float(np.nanmedian(left_shoulder)) - width,
            float(np.nanmedian(right_shoulder)) - width,
        )
        if (
            width <= float(np.nanmin(local)) + VALLEY_PLATEAU_TOLERANCE_M
            and prominence >= VALLEY_MIN_PROMINENCE_M
        ):
            basin_start = max(0, index - VALLEY_PLATEAU_RADIUS_BINS)
            basin_end = min(len(smooth_upper), index + VALLEY_PLATEAU_RADIUS_BINS + 1)
            basin = smooth_upper[basin_start:basin_end]
            basin_minimum = float(np.nanmin(basin))
            plateau = np.flatnonzero(
                np.isfinite(basin)
                & (basin <= basin_minimum + VALLEY_PLATEAU_TOLERANCE_M)
            )
            plateau_center = int(round(float(np.mean(plateau)))) + basin_start
            valley_index = plateau_center
            valley_prominence = prominence
            break

    piecewise_near70_slope = None
    piecewise_left_slope = None
    piecewise_fit_improvement = None
    rule3_peak_ratio = None
    rule3_peak_to_valley_drop_m = None
    rule3_left_confirmation_rise_m = None
    plateau_start_ratio = None
    plateau_end_ratio = None
    plateau_near70_slope = None
    plateau_middle_slope = None
    plateau_left_slope = None
    plateau_fit_improvement = None
    if valley_index is not None:
        detection_status = "first_broad_valley_from_70pct_leftward"
        selected_rule = "rule1_first_u_shaped_valley"
    else:
        # When the direct valley detector fails, fit two straight trends. The
        # same fit supports a broad-U supplement to rule 1 and rules 2 and 3.
        original_indices = np.flatnonzero(np.isfinite(smooth_upper))[::-1]
        x_from_70 = stations[-1] - stations[original_indices]
        widths_from_70 = smooth_upper[original_indices]
        single_fit = np.polyfit(x_from_70, widths_from_70, 1)
        single_sse = float(
            np.sum((widths_from_70 - np.polyval(single_fit, x_from_70)) ** 2)
        )
        piecewise_candidates: list[tuple[float, int, float, float, float]] = []
        for split in range(
            RULE2_MIN_SEGMENT_BINS,
            len(x_from_70) - RULE2_MIN_SEGMENT_BINS,
        ):
            near70_fit = np.polyfit(
                x_from_70[: split + 1], widths_from_70[: split + 1], 1
            )
            left_fit = np.polyfit(x_from_70[split:], widths_from_70[split:], 1)
            near70_prediction = np.polyval(near70_fit, x_from_70[: split + 1])
            left_prediction = np.polyval(left_fit, x_from_70[split:])
            two_sse = float(
                np.sum((widths_from_70[: split + 1] - near70_prediction) ** 2)
                + np.sum((widths_from_70[split:] - left_prediction) ** 2)
            )
            near70_slope = float(near70_fit[0])
            left_slope = float(left_fit[0])
            improvement = (single_sse - two_sse) / max(single_sse, 1e-12)
            piecewise_candidates.append(
                (two_sse, split, near70_slope, left_slope, improvement)
            )

        # Rule 1 supplement: a very broad U can have shoulders too far from its
        # flat bottom for the local prominence test. Opposite-signed fitted
        # slopes confirm the U, then the center of the low plateau is selected.
        broad_u_candidates = [
            candidate
            for candidate in piecewise_candidates
            if candidate[2] < 0.0
            and candidate[3] > 0.0
            and candidate[3] - candidate[2] >= RULE2_MIN_SLOPE_INCREASE
            and candidate[4] >= RULE2_MIN_FIT_IMPROVEMENT
        ]
        if broad_u_candidates:
            (
                _,
                _,
                piecewise_near70_slope,
                piecewise_left_slope,
                piecewise_fit_improvement,
            ) = min(broad_u_candidates)
            finite_indices = np.flatnonzero(np.isfinite(smooth_upper))
            profile_minimum = float(np.nanmin(smooth_upper))
            plateau_indices = finite_indices[
                smooth_upper[finite_indices]
                <= profile_minimum + VALLEY_PLATEAU_TOLERANCE_M
            ]
            valley_index = int(round(float(np.mean(plateau_indices))))
            detection_status = "broad_u_valley_piecewise_slope_supplement"
            selected_rule = "rule1_first_u_shaped_valley"
        else:
            # Rule 2: slow widening from 70% becomes much faster farther left.
            rule2_candidates = [
                candidate
                for candidate in piecewise_candidates
                if candidate[2] >= 0.0
                and candidate[3] > 0.0
                and candidate[3] - candidate[2] >= RULE2_MIN_SLOPE_INCREASE
                and candidate[3] / max(candidate[2], 0.02) >= RULE2_MIN_SLOPE_RATIO
                and candidate[4] >= RULE2_MIN_FIT_IMPROVEMENT
            ]
        if valley_index is None and rule2_candidates:
            (
                _,
                split,
                piecewise_near70_slope,
                piecewise_left_slope,
                piecewise_fit_improvement,
            ) = min(rule2_candidates)
            valley_index = int(original_indices[split])
            detection_status = "slope_acceleration_boundary_from_70pct_leftward"
            selected_rule = "rule2_slow_to_fast_width_increase"
        elif valley_index is None:
            # Rule 3: after rapid widening, continue left past the first peak
            # and select the following valley. Stations from 35%-45% are used
            # only to confirm that the contour rises again on the valley's left;
            # the selected boundary must still remain within 45%-70%.
            rule3_candidates = [
                candidate
                for candidate in piecewise_candidates
                if candidate[2] > 0.0
                and candidate[3] <= 0.02
                and candidate[2] - candidate[3] >= RULE2_MIN_SLOPE_INCREASE
                and candidate[4] >= RULE2_MIN_FIT_IMPROVEMENT
            ]
            if rule3_candidates:
                (
                    _,
                    split,
                    piecewise_near70_slope,
                    piecewise_left_slope,
                    piecewise_fit_improvement,
                ) = min(rule3_candidates)
                support_first_station = (
                    math.ceil(RULE3_SUPPORT_LEFT_RATIO * full_length / PROFILE_BIN_M)
                    * PROFILE_BIN_M
                )
                support_stations = np.arange(
                    support_first_station,
                    last_station + PROFILE_BIN_M * 0.5,
                    PROFILE_BIN_M,
                )
                support_raw = np.full(len(support_stations), np.nan)
                for support_index, support_station in enumerate(support_stations):
                    support_values = side_lateral[
                        np.abs(longitudinal - support_station) <= PROFILE_BIN_M / 2
                    ]
                    if len(support_values) >= 8:
                        edge = float(np.quantile(support_values, UPPER_EDGE_QUANTILE))
                        if edge > 0:
                            support_raw[support_index] = edge
                support_smooth = rolling_median(support_raw, PROFILE_SMOOTHING_BINS)
                main_support_indices = np.flatnonzero(
                    np.isfinite(support_smooth)
                    & (support_stations >= SEARCH_LEFT_RATIO * full_length)
                    & (support_stations <= SEARCH_RIGHT_RATIO * full_length)
                )
                peak_index = int(
                    main_support_indices[
                        np.argmax(support_smooth[main_support_indices])
                    ]
                )
                left_of_peak = main_support_indices[
                    support_stations[main_support_indices]
                    <= support_stations[peak_index] - 0.03
                ]
                post_peak_valley_index = int(
                    left_of_peak[np.argmin(support_smooth[left_of_peak])]
                )
                left_confirmation = np.flatnonzero(
                    np.isfinite(support_smooth)
                    & (support_stations >= support_stations[post_peak_valley_index] - 0.12)
                    & (support_stations <= support_stations[post_peak_valley_index] - 0.04)
                )
                peak_drop = float(
                    support_smooth[peak_index]
                    - support_smooth[post_peak_valley_index]
                )
                left_rise = float(
                    np.nanmedian(support_smooth[left_confirmation])
                    - support_smooth[post_peak_valley_index]
                )
                if (
                    len(left_confirmation) >= 3
                    and peak_drop >= RULE3_MIN_PEAK_TO_VALLEY_DROP_M
                    and left_rise >= RULE3_MIN_LEFT_CONFIRMATION_RISE_M
                ):
                    selected_station = float(support_stations[post_peak_valley_index])
                    valley_index = int(np.argmin(np.abs(stations - selected_station)))
                    rule3_peak_ratio = float(support_stations[peak_index] / full_length)
                    rule3_peak_to_valley_drop_m = peak_drop
                    rule3_left_confirmation_rise_m = left_rise
                    detection_status = "post_peak_valley_with_left_support_confirmation"
                    selected_rule = "rule3_post_peak_valley"
                else:
                    valley_index = None
                    detection_status = "no_rule1_rule2_or_rule3_match"
                    selected_rule = "none"
            else:
                valley_index = None
                detection_status = "no_rule1_rule2_or_rule3_match"
                selected_rule = "none"

        # Rule 2 supplement: a monotonic contour can contain a short flat band
        # between two widening stages. Select the center of that band. This is
        # evaluated only after all preceding detectors fail.
        if valley_index is None:
            plateau_candidates: list[
                tuple[float, int, int, float, float, float, float]
            ] = []
            for first_split in range(
                PLATEAU_MIN_SEGMENT_BINS,
                len(x_from_70) - 2 * PLATEAU_MIN_SEGMENT_BINS,
            ):
                for second_split in range(
                    first_split + PLATEAU_MIN_SEGMENT_BINS,
                    len(x_from_70) - PLATEAU_MIN_SEGMENT_BINS,
                ):
                    near70_fit = np.polyfit(
                        x_from_70[: first_split + 1],
                        widths_from_70[: first_split + 1],
                        1,
                    )
                    middle_fit = np.polyfit(
                        x_from_70[first_split : second_split + 1],
                        widths_from_70[first_split : second_split + 1],
                        1,
                    )
                    left_fit = np.polyfit(
                        x_from_70[second_split:], widths_from_70[second_split:], 1
                    )
                    three_sse = float(
                        np.sum(
                            (
                                widths_from_70[: first_split + 1]
                                - np.polyval(
                                    near70_fit, x_from_70[: first_split + 1]
                                )
                            )
                            ** 2
                        )
                        + np.sum(
                            (
                                widths_from_70[first_split : second_split + 1]
                                - np.polyval(
                                    middle_fit,
                                    x_from_70[first_split : second_split + 1],
                                )
                            )
                            ** 2
                        )
                        + np.sum(
                            (
                                widths_from_70[second_split:]
                                - np.polyval(left_fit, x_from_70[second_split:])
                            )
                            ** 2
                        )
                    )
                    slopes = (
                        float(near70_fit[0]),
                        float(middle_fit[0]),
                        float(left_fit[0]),
                    )
                    improvement = (single_sse - three_sse) / max(single_sse, 1e-12)
                    if (
                        slopes[0] >= PLATEAU_MIN_SIDE_SLOPE
                        and abs(slopes[1]) <= PLATEAU_MAX_ABS_SLOPE
                        and slopes[2] >= PLATEAU_MIN_SIDE_SLOPE
                        and improvement >= PLATEAU_MIN_FIT_IMPROVEMENT
                    ):
                        plateau_candidates.append(
                            (
                                three_sse,
                                first_split,
                                second_split,
                                slopes[0],
                                slopes[1],
                                slopes[2],
                                improvement,
                            )
                        )
            if plateau_candidates:
                (
                    _,
                    first_split,
                    second_split,
                    plateau_near70_slope,
                    plateau_middle_slope,
                    plateau_left_slope,
                    plateau_fit_improvement,
                ) = min(plateau_candidates)
                center_from_70 = float(
                    (x_from_70[first_split] + x_from_70[second_split]) / 2
                )
                selected_station = float(stations[-1] - center_from_70)
                valley_index = int(np.argmin(np.abs(stations - selected_station)))
                plateau_start_ratio = float(
                    stations[original_indices[first_split]] / full_length
                )
                plateau_end_ratio = float(
                    stations[original_indices[second_split]] / full_length
                )
                detection_status = "widening_plateau_between_two_increase_stages"
                selected_rule = "rule2_widening_plateau_supplement"

    if valley_index is not None:
        valley_station = float(stations[valley_index])
        valley_halfwidth = float(smooth_upper[valley_index])
        contour_xy = (
            rear_xy
            + valley_station * axis
            + side_sign * valley_halfwidth * normal
        )
        axis_xy = rear_xy + valley_station * axis
    else:
        valley_station = None
        valley_halfwidth = None
        contour_xy = None
        axis_xy = None

    # Reproduce the exact panel transform used by the base preview.
    t_min, t_max = np.quantile(longitudinal, [0.001, 0.999])
    q_min, q_max = np.quantile(lateral, [0.001, 0.999])
    t_min = min(float(t_min), 0.0) - 0.04
    t_max = max(float(t_max), full_length) + 0.04
    q_min, q_max = float(q_min) - 0.04, float(q_max) + 0.04
    scale = min(
        (PLOT_RIGHT - PLOT_LEFT) / max(t_max - t_min, 1e-6),
        (PLOT_BOTTOM - PLOT_TOP) / max(q_max - q_min, 1e-6),
    )
    center_t = (t_min + t_max) / 2
    center_q = (q_min + q_max) / 2
    center_px_x = (PLOT_LEFT + PLOT_RIGHT) / 2
    center_px_y = (PLOT_TOP + PLOT_BOTTOM) / 2

    def px(t: float) -> float:
        return center_px_x + (t - center_t) * scale

    def py(q: float) -> float:
        return center_px_y - (q - center_q) * scale

    draw = ImageDraw.Draw(panel)
    if valley_index is not None:
        x = px(valley_station)
        y_axis = py(0.0)
        y_contour = py(side_sign * valley_halfwidth)
        if selected_rule == "rule1_first_u_shaped_valley":
            line_color = HALFWIDTH_COLOR
        elif selected_rule.startswith("rule2_"):
            line_color = SLOPE_CHANGE_COLOR
        else:
            line_color = FAST_TO_FLAT_COLOR
        draw.line((x, y_axis, x, y_contour), fill=HALFWIDTH_OUTLINE, width=9)
        draw.line((x, y_axis, x, y_contour), fill=line_color, width=5)
        radius = 6
        draw.ellipse(
            (x - radius, y_contour - radius, x + radius, y_contour + radius),
            fill=line_color,
            outline=HALFWIDTH_OUTLINE,
            width=2,
        )
        draw.polygon(
            [(x, y_contour - 12), (x - 9, y_contour + 4), (x + 9, y_contour + 4)],
            fill=line_color,
            outline=HALFWIDTH_OUTLINE,
        )
        if selected_rule == "rule1_first_u_shaped_valley":
            label_text = f"{'上' if side == 'upper' else '下'}规则1 U形谷底 {valley_halfwidth * 100:.1f} cm"
        elif selected_rule == "rule2_slow_to_fast_width_increase":
            label_text = f"{'上' if side == 'upper' else '下'}规则2 慢增转快增 {valley_halfwidth * 100:.1f} cm"
        elif selected_rule == "rule2_widening_plateau_supplement":
            label_text = f"{'上' if side == 'upper' else '下'}规则2 平台中心 {valley_halfwidth * 100:.1f} cm"
        else:
            label_text = f"{'上' if side == 'upper' else '下'}规则3 峰后谷底 {valley_halfwidth * 100:.1f} cm"
        text_box = draw.textbbox((0, 0), label_text, font=LABEL_FONT, stroke_width=2)
        text_width = text_box[2] - text_box[0]
        place_on_right = x + 12 + text_width <= PANEL_WIDTH - 12
        label_x = x + 12 if place_on_right else x - 12
        anchor = "ls" if place_on_right else "rs"
        if side == "upper":
            label_y = max(PLOT_TOP + 17, y_contour - 15)
            vertical_anchor = "s"
        else:
            label_y = min(PLOT_BOTTOM - 17, y_contour + 15)
            vertical_anchor = "a"
        draw.text(
            (label_x, label_y),
            label_text,
            font=LABEL_FONT,
            fill=line_color,
            anchor=anchor[0] + vertical_anchor,
            stroke_width=2,
            stroke_fill=BACKGROUND,
        )
    else:
        draw.text(
            (
                PANEL_WIDTH / 2,
                PLOT_TOP + 22 if side == "upper" else PLOT_BOTTOM - 22,
            ),
            f"{'上侧' if side == 'upper' else '下侧'}规则1/2/3均未找到",
            font=LABEL_FONT,
            fill=NO_MATCH_COLOR,
            anchor="ms" if side == "upper" else "ma",
            stroke_width=2,
            stroke_fill=BACKGROUND,
        )

    prefix = f"pca_{side}"
    side_record = {
        f"{prefix}_boundary_selected_rule": selected_rule,
        f"{prefix}_width_valley_m": valley_halfwidth,
        f"{prefix}_width_valley_station_from_rear_m": valley_station,
        f"{prefix}_width_valley_station_ratio_full_axis": (
            valley_station / full_length if valley_station is not None else None
        ),
        f"{prefix}_width_valley_axis_x_m": float(axis_xy[0]) if axis_xy is not None else None,
        f"{prefix}_width_valley_axis_y_m": float(axis_xy[1]) if axis_xy is not None else None,
        f"{prefix}_width_valley_contour_x_m": float(contour_xy[0]) if contour_xy is not None else None,
        f"{prefix}_width_valley_contour_y_m": float(contour_xy[1]) if contour_xy is not None else None,
        f"{prefix}_width_valley_raw_m": float(raw_upper[valley_index]) if valley_index is not None else None,
        f"{prefix}_width_valley_slice_point_count": int(point_counts[valley_index]) if valley_index is not None else None,
        f"{prefix}_width_valley_detection_status": detection_status,
        f"{prefix}_width_valley_prominence_m": valley_prominence,
        f"{prefix}_piecewise_near70_slope": piecewise_near70_slope,
        f"{prefix}_piecewise_left_slope": piecewise_left_slope,
        f"{prefix}_piecewise_fit_improvement": piecewise_fit_improvement,
        f"{prefix}_rule3_peak_ratio_from_rear": rule3_peak_ratio,
        f"{prefix}_rule3_peak_to_valley_drop_m": rule3_peak_to_valley_drop_m,
        f"{prefix}_rule3_left_confirmation_rise_m": rule3_left_confirmation_rise_m,
        f"{prefix}_rule3_support_left_ratio": RULE3_SUPPORT_LEFT_RATIO,
        f"{prefix}_plateau_start_ratio_from_rear": plateau_start_ratio,
        f"{prefix}_plateau_end_ratio_from_rear": plateau_end_ratio,
        f"{prefix}_plateau_near70_slope": plateau_near70_slope,
        f"{prefix}_plateau_middle_slope": plateau_middle_slope,
        f"{prefix}_plateau_left_slope": plateau_left_slope,
        f"{prefix}_plateau_fit_improvement": plateau_fit_improvement,
        f"{prefix}_width_valley_search_left_ratio": SEARCH_LEFT_RATIO,
        f"{prefix}_width_valley_search_right_ratio": SEARCH_RIGHT_RATIO,
        f"{prefix}_halfwidth_edge_quantile": UPPER_EDGE_QUANTILE,
        f"{prefix}_halfwidth_profile_bin_m": PROFILE_BIN_M,
        f"{prefix}_halfwidth_smoothing_m": PROFILE_BIN_M * PROFILE_SMOOTHING_BINS,
        f"{prefix}_width_valley_method_version": "pca_three_rule_with_plateau_supplement_v7.0",
    }
    return {**base_record, **side_record}


def main() -> None:
    rows = read_rows()
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    # This output folder is dedicated to generated previews. Remove the older
    # PNGs so it contains only the current three-rule version.
    for old_path in INDIVIDUAL_DIR.glob("*.png"):
        old_path.unlink()
    for old_path in GROUP_DIR.glob("*.png"):
        old_path.unlink()
    panels: list[tuple[int, Image.Image]] = []
    records: list[dict[str, object]] = []
    for row in rows:
        panel, base_record = make_panel(row, marker_ratios=(0.70,))
        upper_record = add_three_rule_boundary(row, panel, base_record, side="upper")
        record = add_three_rule_boundary(row, panel, upper_record, side="lower")
        cow_id = int(record["cow_id"])
        path = INDIVIDUAL_DIR / f"cow_{cow_id:03d}_pca_horizontal_upper_lower_three_rule_boundary.png"
        panel.save(path, optimize=True)
        panels.append((cow_id, panel))
        records.append(record)

    group_files: list[str] = []
    for group_number, start in enumerate(range(0, len(panels), GROUP_SIZE), start=1):
        group = panels[start : start + GROUP_SIZE]
        rows_count = math.ceil(len(group) / GROUP_COLUMNS)
        canvas = Image.new("RGB", (PANEL_WIDTH * GROUP_COLUMNS, PANEL_HEIGHT * rows_count), BACKGROUND)
        group_draw = ImageDraw.Draw(canvas)
        for index, (_, panel) in enumerate(group):
            x = (index % GROUP_COLUMNS) * PANEL_WIDTH
            y = (index // GROUP_COLUMNS) * PANEL_HEIGHT
            canvas.paste(panel, (x, y))
            if index % GROUP_COLUMNS == 0:
                group_draw.line((PANEL_WIDTH - 1, y + 8, PANEL_WIDTH - 1, y + PANEL_HEIGHT - 8), fill=SEPARATOR, width=2)
            if index < len(group) - GROUP_COLUMNS:
                group_draw.line((x + 10, y + PANEL_HEIGHT - 1, x + PANEL_WIDTH - 10, y + PANEL_HEIGHT - 1), fill=SEPARATOR, width=2)
        first_id, last_id = group[0][0], group[-1][0]
        group_path = GROUP_DIR / f"pca_upper_lower_three_rule_group_{group_number:02d}_cows_{first_id:03d}_{last_id:03d}.png"
        canvas.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(group_path, optimize=True)
        group_files.append(str(group_path.relative_to(ROOT)))
        for index in range(start, start + len(group)):
            records[index]["preview_group_number"] = group_number
            records[index]["preview_group_file"] = str(group_path.relative_to(ROOT))

    with RECORDS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    upper_values = np.asarray(
        [
            float(row["pca_upper_width_valley_m"])
            for row in records
            if row["pca_upper_width_valley_m"] is not None
        ]
    )
    lower_values = np.asarray(
        [
            float(row["pca_lower_width_valley_m"])
            for row in records
            if row["pca_lower_width_valley_m"] is not None
        ]
    )
    upper_rule1_count = sum(row["pca_upper_boundary_selected_rule"] == "rule1_first_u_shaped_valley" for row in records)
    upper_rule1_broad_supplement_count = sum(
        row["pca_upper_width_valley_detection_status"]
        == "broad_u_valley_piecewise_slope_supplement"
        for row in records
    )
    upper_rule2_count = sum(row["pca_upper_boundary_selected_rule"].startswith("rule2_") for row in records)
    upper_rule3_count = sum(row["pca_upper_boundary_selected_rule"] == "rule3_post_peak_valley" for row in records)
    upper_no_match_count = len(records) - upper_rule1_count - upper_rule2_count - upper_rule3_count
    lower_rule1_count = sum(row["pca_lower_boundary_selected_rule"] == "rule1_first_u_shaped_valley" for row in records)
    lower_rule2_count = sum(row["pca_lower_boundary_selected_rule"].startswith("rule2_") for row in records)
    lower_plateau_supplement_count = sum(
        row["pca_lower_boundary_selected_rule"]
        == "rule2_widening_plateau_supplement"
        for row in records
    )
    lower_rule3_count = sum(row["pca_lower_boundary_selected_rule"] == "rule3_post_peak_valley" for row in records)
    lower_no_match_count = len(records) - lower_rule1_count - lower_rule2_count - lower_rule3_count
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "cattle_count": len(records),
                "definition": "apply the same ordered three-rule boundary detector independently to upper and lower PCA half-width profiles; rule 2 includes an increase-flat-increase plateau supplement",
                "marker": "70% of full PCA long-axis length measured from rear toward head",
                "display_orientation": "rear left; head right",
                "search_ratio_range": [SEARCH_LEFT_RATIO, SEARCH_RIGHT_RATIO],
                "rule3_support_ratio_range": [RULE3_SUPPORT_LEFT_RATIO, SEARCH_LEFT_RATIO],
                "upper_rule1_u_shaped_valley_count": upper_rule1_count,
                "upper_rule1_broad_u_supplement_count": upper_rule1_broad_supplement_count,
                "upper_rule2_slope_acceleration_count": upper_rule2_count,
                "upper_rule3_post_peak_valley_count": upper_rule3_count,
                "upper_no_rule_match_count": upper_no_match_count,
                "lower_rule1_u_shaped_valley_count": lower_rule1_count,
                "lower_rule2_slope_acceleration_count": lower_rule2_count,
                "lower_rule2_plateau_supplement_count": lower_plateau_supplement_count,
                "lower_rule3_post_peak_valley_count": lower_rule3_count,
                "lower_no_rule_match_count": lower_no_match_count,
                "mean_selected_upper_halfwidth_m": float(np.mean(upper_values)),
                "mean_selected_lower_halfwidth_m": float(np.mean(lower_values)) if len(lower_values) else None,
                "group_count": len(group_files),
                "group_files": group_files,
                "records": str(RECORDS_CSV.relative_to(ROOT)),
                "method_version": "pca_upper_lower_three_rule_with_plateau_supplement_v7.0",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"cattle: {len(records)}")
    print(f"groups: {len(group_files)}")
    print(f"upper rules 1/2/3/no-match: {upper_rule1_count}/{upper_rule2_count}/{upper_rule3_count}/{upper_no_match_count}")
    print(f"lower rules 1/2/3/no-match: {lower_rule1_count}/{lower_rule2_count}/{lower_rule3_count}/{lower_no_match_count}")
    print(f"mean selected upper/lower half-width: {np.mean(upper_values):.4f}/{np.mean(lower_values):.4f} m")
    for path in group_files:
        print(path)


if __name__ == "__main__":
    main()
