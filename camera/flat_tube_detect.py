import time

import cv2
import numpy as np
import pyrealsense2 as rs


DEPTH_STREAM_SIZE = (848, 480)
COLOR_STREAM_SIZE = (848, 480)
DEPTH_FPS = 30
COLOR_FPS = 30

LOCK_RGB_SETTINGS = True
RGB_EXPOSURE = 1250
RGB_GAIN = 0
RGB_WHITE_BALANCE = 5200.0

FRAME_SKIP_COUNT = 78
REQUIRED_STABLE_FRAMES = 3
MAX_LOST_FRAMES = 5

MIN_PICK_CONFIDENCE = 0.18
MIN_EDGE_SCORE = 0.18

NEAR_THRESHOLD_M = 0.47
COMBINED_CLOSE_SIZE = (7, 7)
FOREGROUND_CLOSE_SIZE = (9, 9)
FOREGROUND_DILATE_SIZE = (3, 3)

MIN_CONTOUR_AREA = 2500
MAX_CONTOUR_AREA = 14000
MIN_ASPECT_RATIO = 1.8
MAX_ASPECT_RATIO = 5.5
MIN_FILL_RATIO = 0.45
MARGIN_PX = 8
DEPTH_WINDOW_SIZE = 5

USE_FIXED_ROI = False
ROI_POLYGON_NORM = np.array(
    [
        [0.12, 0.06],
        [0.84, 0.06],
        [0.84, 0.94],
        [0.12, 0.94],
    ],
    dtype=np.float32,
)

HSV_BLUE_LOWER = np.array([100, 100, 80])
HSV_BLUE_UPPER = np.array([115, 255, 255])
HSV_GREEN_LOWER = np.array([65, 75, 40])
HSV_GREEN_UPPER = np.array([98, 255, 255])
HSV_YELLOW_LOWER = np.array([10, 40, 20])
HSV_YELLOW_UPPER = np.array([40, 255, 255])

MERGE_ANGLE_DIFF_DEG = 20.0
MERGE_DEPTH_DIFF_M = 0.03
MERGE_PERP_OFFSET_PX = 20.0
MERGE_AXIS_GAP_PX = 80.0

WIDTH_PROFILE_BIN_COUNT = 24
END_WIDTH_FRACTION = 0.20
MIN_CAP_WIDTH_RATIO = 1.12
CAP_CONSISTENCY_WEIGHT = 0.35
GRASP_INSET_FRACTION = 0.12
GRASP_BAND_FRACTION = 0.04
MIN_PIXELS_PER_WIDTH_BIN = 10
MIN_PIXELS_IN_GRASP_BAND = 20


def build_contour_mask(contour, mask_shape):
    contour_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    return contour_mask


def build_roi_mask(mask_shape, roi_polygon_norm):
    img_h, img_w = mask_shape
    roi_polygon_px = np.column_stack(
        (
            np.round(roi_polygon_norm[:, 0] * img_w),
            np.round(roi_polygon_norm[:, 1] * img_h),
        )
    ).astype(np.int32)

    roi_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_polygon_px], 255)
    return roi_mask, roi_polygon_px


def get_dominant_color(blue_mask, green_mask, yellow_mask, contour, mask_shape):
    contour_mask = build_contour_mask(contour, mask_shape)

    blue_count = cv2.countNonZero(cv2.bitwise_and(blue_mask, contour_mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, contour_mask))
    yellow_count = cv2.countNonZero(cv2.bitwise_and(yellow_mask, contour_mask))

    counts = {
        "BLUE": blue_count,
        "GREEN": green_count,
        "YELLOW": yellow_count,
    }
    label = max(counts, key=counts.get)

    if label == "BLUE":
        box_color = (255, 0, 0)
    elif label == "GREEN":
        box_color = (0, 255, 0)
    else:
        box_color = (0, 255, 255)

    return label, box_color


def get_window_median_depth_m(depth_image, depth_scale, cx, cy, window_size):
    half = window_size // 2
    img_h, img_w = depth_image.shape

    x1 = max(0, cx - half)
    x2 = min(img_w, cx + half + 1)
    y1 = max(0, cy - half)
    y2 = min(img_h, cy + half + 1)

    depth_window = depth_image[y1:y2, x1:x2]
    valid_depths = depth_window[depth_window > 0]

    if valid_depths.size == 0:
        return 0.0

    return float(np.median(valid_depths) * depth_scale)


def get_long_axis_angle_deg(rect):
    (_, _), (width, height), angle = rect

    if width < height:
        angle += 90.0

    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0

    return angle


def normalize_axis_angle_diff_deg(angle_a, angle_b):
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def average_axis_angle_deg(angle_a, angle_b):
    angle_a_rad = np.deg2rad(angle_a * 2.0)
    angle_b_rad = np.deg2rad(angle_b * 2.0)

    x = np.cos(angle_a_rad) + np.cos(angle_b_rad)
    y = np.sin(angle_a_rad) + np.sin(angle_b_rad)

    average_angle = 0.5 * np.rad2deg(np.arctan2(y, x))
    if average_angle < 0:
        average_angle += 180.0
    return average_angle


def get_axis_vectors(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    axis_u = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
    return axis_u, perp_u


def get_projection_interval(points_xy, axis_u):
    projections = points_xy @ axis_u
    return float(np.min(projections)), float(np.max(projections))


def get_projection_gap(min_a, max_a, min_b, max_b):
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def compute_shape_metrics(contour):
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (_, _), (rect_w, rect_h), _ = rect

    if rect_w <= 1 or rect_h <= 1:
        return None

    long_side = max(rect_w, rect_h)
    short_side = min(rect_w, rect_h)
    aspect_ratio = long_side / short_side
    rect_area = rect_w * rect_h
    fill_ratio = area / rect_area
    angle_deg = get_long_axis_angle_deg(rect)
    axis_u, perp_u = get_axis_vectors(angle_deg)

    points_xy = contour.reshape(-1, 2).astype(np.float32)
    projection_min, projection_max = get_projection_interval(points_xy, axis_u)
    line_start = np.array(rect[0], dtype=np.float32) - axis_u * (long_side * 0.5)
    line_end = np.array(rect[0], dtype=np.float32) + axis_u * (long_side * 0.5)

    return {
        "area": area,
        "rect": rect,
        "rect_w": rect_w,
        "rect_h": rect_h,
        "long_side": long_side,
        "short_side": short_side,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "angle_deg": angle_deg,
        "axis_u": axis_u,
        "perp_u": perp_u,
        "points_xy": points_xy,
        "projection_min": projection_min,
        "projection_max": projection_max,
        "line_start": line_start,
        "line_end": line_end,
    }


def passes_shape_filters(shape_metrics):
    if shape_metrics is None:
        return False
    if shape_metrics["area"] < MIN_CONTOUR_AREA or shape_metrics["area"] > MAX_CONTOUR_AREA:
        return False
    if (
        shape_metrics["aspect_ratio"] < MIN_ASPECT_RATIO
        or shape_metrics["aspect_ratio"] > MAX_ASPECT_RATIO
    ):
        return False
    if shape_metrics["fill_ratio"] < MIN_FILL_RATIO:
        return False
    return True


def build_raw_candidate(
    contour,
    depth_image,
    depth_scale,
    depth_intrin,
    blue_mask,
    green_mask,
    yellow_mask,
    mask_shape,
):
    shape_metrics = compute_shape_metrics(contour)
    if not passes_shape_filters(shape_metrics):
        return None

    rect_cx, rect_cy = shape_metrics["rect"][0]
    cx = int(round(rect_cx))
    cy = int(round(rect_cy))

    img_h, img_w = mask_shape
    if (
        cx <= MARGIN_PX
        or cy <= MARGIN_PX
        or cx >= img_w - MARGIN_PX
        or cy >= img_h - MARGIN_PX
    ):
        return None

    distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        cx,
        cy,
        DEPTH_WINDOW_SIZE,
    )
    if distance_m <= 0:
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance_m)
    x_m, y_m, z_m = point_3d

    label, box_color = get_dominant_color(
        blue_mask,
        green_mask,
        yellow_mask,
        contour,
        mask_shape,
    )

    return {
        "contour": contour,
        "label": label,
        "box_color": box_color,
        "cx": cx,
        "cy": cy,
        "X_m": x_m,
        "Y_m": y_m,
        "Z_m": z_m,
        "distance_m": distance_m,
        **shape_metrics,
    }


def should_merge_candidates(candidate_a, candidate_b):
    if candidate_a["label"] != candidate_b["label"]:
        return False

    angle_diff = normalize_axis_angle_diff_deg(
        candidate_a["angle_deg"], candidate_b["angle_deg"]
    )
    if angle_diff > MERGE_ANGLE_DIFF_DEG:
        return False

    if abs(candidate_a["Z_m"] - candidate_b["Z_m"]) > MERGE_DEPTH_DIFF_M:
        return False

    average_angle = average_axis_angle_deg(
        candidate_a["angle_deg"], candidate_b["angle_deg"]
    )
    axis_u, perp_u = get_axis_vectors(average_angle)

    min_a, max_a = get_projection_interval(candidate_a["points_xy"], axis_u)
    min_b, max_b = get_projection_interval(candidate_b["points_xy"], axis_u)
    axis_gap = get_projection_gap(min_a, max_a, min_b, max_b)
    if axis_gap > MERGE_AXIS_GAP_PX:
        return False

    center_delta = np.array(
        [candidate_b["cx"] - candidate_a["cx"], candidate_b["cy"] - candidate_a["cy"]],
        dtype=np.float32,
    )
    perpendicular_offset = abs(float(center_delta @ perp_u))
    if perpendicular_offset > MERGE_PERP_OFFSET_PX:
        return False

    return True


def group_raw_candidates(raw_candidates):
    groups = []
    visited = set()

    for start_idx in range(len(raw_candidates)):
        if start_idx in visited:
            continue

        queue = [start_idx]
        component = []

        while queue:
            current_idx = queue.pop()
            if current_idx in visited:
                continue

            visited.add(current_idx)
            component.append(raw_candidates[current_idx])

            for next_idx in range(len(raw_candidates)):
                if next_idx in visited or next_idx == current_idx:
                    continue
                if should_merge_candidates(
                    raw_candidates[current_idx], raw_candidates[next_idx]
                ):
                    queue.append(next_idx)

        groups.append(component)

    return groups


def build_merged_contour(candidate_group, mask_shape):
    merged_mask = np.zeros(mask_shape, dtype=np.uint8)

    all_points = np.vstack(
        [candidate["points_xy"] for candidate in candidate_group]
    ).astype(np.int32)
    if len(candidate_group) == 1:
        cv2.drawContours(
            merged_mask, [candidate_group[0]["contour"]], -1, 255, thickness=-1
        )
    else:
        hull = cv2.convexHull(all_points)
        cv2.drawContours(merged_mask, [hull], -1, 255, thickness=-1)

    contours, _ = cv2.findContours(
        merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def analyze_cap_geometry(contour, mask_shape, angle_deg):
    contour_mask = build_contour_mask(contour, mask_shape)
    ys, xs = np.nonzero(contour_mask)
    if len(xs) == 0:
        return None

    coords_xy = np.column_stack((xs, ys)).astype(np.float32)
    center_xy = np.mean(coords_xy, axis=0)
    axis_u, perp_u = get_axis_vectors(angle_deg)

    centered_xy = coords_xy - center_xy
    axis_values = centered_xy @ axis_u
    perp_values = centered_xy @ perp_u

    axis_min = float(np.min(axis_values))
    axis_max = float(np.max(axis_values))
    tube_length = axis_max - axis_min
    if tube_length <= 1.0:
        return None

    axis_edges = np.linspace(axis_min, axis_max, WIDTH_PROFILE_BIN_COUNT + 1)
    width_profile = np.full(WIDTH_PROFILE_BIN_COUNT, np.nan, dtype=np.float32)

    for idx in range(WIDTH_PROFILE_BIN_COUNT):
        if idx == WIDTH_PROFILE_BIN_COUNT - 1:
            in_bin = (axis_values >= axis_edges[idx]) & (
                axis_values <= axis_edges[idx + 1]
            )
        else:
            in_bin = (axis_values >= axis_edges[idx]) & (
                axis_values < axis_edges[idx + 1]
            )

        if np.count_nonzero(in_bin) < MIN_PIXELS_PER_WIDTH_BIN:
            continue

        bin_perp_values = perp_values[in_bin]
        width_profile[idx] = float(
            np.max(bin_perp_values) - np.min(bin_perp_values)
        )

    end_bin_count = max(3, int(round(WIDTH_PROFILE_BIN_COUNT * END_WIDTH_FRACTION)))
    left_widths = width_profile[:end_bin_count]
    right_widths = width_profile[-end_bin_count:]

    left_valid = left_widths[~np.isnan(left_widths)]
    right_valid = right_widths[~np.isnan(right_widths)]

    if len(left_valid) < max(2, end_bin_count // 2):
        return None
    if len(right_valid) < max(2, end_bin_count // 2):
        return None

    left_avg = float(np.mean(left_valid))
    right_avg = float(np.mean(right_valid))
    left_std = float(np.std(left_valid))
    right_std = float(np.std(right_valid))

    smaller_avg = min(left_avg, right_avg)
    larger_avg = max(left_avg, right_avg)
    if smaller_avg <= 0:
        return None

    width_ratio = larger_avg / smaller_avg
    if width_ratio < MIN_CAP_WIDTH_RATIO:
        return None

    left_score = left_avg + CAP_CONSISTENCY_WEIGHT * left_std
    right_score = right_avg + CAP_CONSISTENCY_WEIGHT * right_std

    if left_score < right_score:
        cap_end = "axis_min"
        cap_tip_axis_value = axis_min
        grasp_axis_value = axis_min + GRASP_INSET_FRACTION * tube_length
    else:
        cap_end = "axis_max"
        cap_tip_axis_value = axis_max
        grasp_axis_value = axis_max - GRASP_INSET_FRACTION * tube_length

    band_half_width = max(4.0, tube_length * GRASP_BAND_FRACTION)
    in_grasp_band = np.abs(axis_values - grasp_axis_value) <= band_half_width
    if np.count_nonzero(in_grasp_band) < MIN_PIXELS_IN_GRASP_BAND:
        return None

    band_perp_values = perp_values[in_grasp_band]
    grasp_perp_value = 0.5 * (
        float(np.min(band_perp_values)) + float(np.max(band_perp_values))
    )
    grasp_xy = center_xy + axis_u * grasp_axis_value + perp_u * grasp_perp_value

    in_tip_band = np.abs(axis_values - cap_tip_axis_value) <= band_half_width
    if np.count_nonzero(in_tip_band) < MIN_PIXELS_IN_GRASP_BAND:
        tip_perp_value = grasp_perp_value
    else:
        tip_perp_values = perp_values[in_tip_band]
        tip_perp_value = 0.5 * (
            float(np.min(tip_perp_values)) + float(np.max(tip_perp_values))
        )

    cap_tip_xy = center_xy + axis_u * cap_tip_axis_value + perp_u * tip_perp_value
    cap_confidence = width_ratio - 1.0

    return {
        "contour_mask": contour_mask,
        "cap_end": cap_end,
        "cap_confidence": cap_confidence,
        "cap_tip_xy": cap_tip_xy,
        "grasp_xy": grasp_xy,
        "tube_length_px": tube_length,
    }


def build_final_candidate(
    candidate_group,
    merged_contour,
    depth_image,
    depth_scale,
    depth_intrin,
    blue_mask,
    green_mask,
    yellow_mask,
    mask_shape,
):
    shape_metrics = compute_shape_metrics(merged_contour)
    if not passes_shape_filters(shape_metrics):
        return None

    rect_cx, rect_cy = shape_metrics["rect"][0]
    cx = int(round(rect_cx))
    cy = int(round(rect_cy))

    img_h, img_w = mask_shape
    if (
        cx <= MARGIN_PX
        or cy <= MARGIN_PX
        or cx >= img_w - MARGIN_PX
        or cy >= img_h - MARGIN_PX
    ):
        return None

    distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        cx,
        cy,
        DEPTH_WINDOW_SIZE,
    )
    if distance_m <= 0:
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance_m)
    x_m, y_m, z_m = point_3d

    label, box_color = get_dominant_color(
        blue_mask,
        green_mask,
        yellow_mask,
        merged_contour,
        mask_shape,
    )

    cap_geometry = analyze_cap_geometry(
        merged_contour, mask_shape, shape_metrics["angle_deg"]
    )
    if cap_geometry is None:
        return None

    grasp_px = int(round(cap_geometry["grasp_xy"][0]))
    grasp_py = int(round(cap_geometry["grasp_xy"][1]))
    if (
        grasp_px <= MARGIN_PX
        or grasp_py <= MARGIN_PX
        or grasp_px >= img_w - MARGIN_PX
        or grasp_py >= img_h - MARGIN_PX
    ):
        return None

    grasp_distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        grasp_px,
        grasp_py,
        DEPTH_WINDOW_SIZE,
    )
    if grasp_distance_m <= 0:
        return None

    grasp_point_3d = rs.rs2_deproject_pixel_to_point(
        depth_intrin, [grasp_px, grasp_py], grasp_distance_m
    )
    grasp_x_m, grasp_y_m, grasp_z_m = grasp_point_3d

    box_points = cv2.boxPoints(shape_metrics["rect"])
    box_points = np.intp(box_points)

    cap_tip_px = int(round(cap_geometry["cap_tip_xy"][0]))
    cap_tip_py = int(round(cap_geometry["cap_tip_xy"][1]))

    return {
        "label": label,
        "box_color": box_color,
        "contour": merged_contour,
        "box_points": box_points,
        "cx": cx,
        "cy": cy,
        "X_m": x_m,
        "Y_m": y_m,
        "Z_m": z_m,
        "angle_deg": shape_metrics["angle_deg"],
        "axis_u": shape_metrics["axis_u"],
        "long_side": shape_metrics["long_side"],
        "grasp_px": grasp_px,
        "grasp_py": grasp_py,
        "grasp_X_m": grasp_x_m,
        "grasp_Y_m": grasp_y_m,
        "grasp_Z_m": grasp_z_m,
        "cap_end": cap_geometry["cap_end"],
        "cap_confidence": cap_geometry["cap_confidence"],
        "cap_tip_px": cap_tip_px,
        "cap_tip_py": cap_tip_py,
        "part_count": len(candidate_group),
        "area": shape_metrics["area"],
        "aspect_ratio": shape_metrics["aspect_ratio"],
        "fill_ratio": shape_metrics["fill_ratio"],
    }


def build_detection_track_key(detection, px_bucket_size=20):
    return (
        detection["label"],
        int(detection["grasp_px"] / px_bucket_size),
        int(detection["grasp_py"] / px_bucket_size),
    )


def update_detection_tracks(detections, track_memory, frame_idx, stale_after_frames=12):
    active_keys = set()
    alpha = 0.35

    for detection in detections:
        key = build_detection_track_key(detection)
        track = track_memory.get(key, {})
        last_seen_frame = track.get("last_seen_frame", -999999)

        if frame_idx - last_seen_frame <= 2:
            stable_frames = track.get("stable_frames", 0) + 1
        else:
            stable_frames = 1

        track["stable_frames"] = stable_frames
        track["last_seen_frame"] = frame_idx

        if "ema_grasp_X_m" in track:
            track["ema_grasp_X_m"] = (
                (1.0 - alpha) * track["ema_grasp_X_m"] + alpha * detection["grasp_X_m"]
            )
            track["ema_grasp_Y_m"] = (
                (1.0 - alpha) * track["ema_grasp_Y_m"] + alpha * detection["grasp_Y_m"]
            )
            track["ema_grasp_Z_m"] = (
                (1.0 - alpha) * track["ema_grasp_Z_m"] + alpha * detection["grasp_Z_m"]
            )
        else:
            track["ema_grasp_X_m"] = detection["grasp_X_m"]
            track["ema_grasp_Y_m"] = detection["grasp_Y_m"]
            track["ema_grasp_Z_m"] = detection["grasp_Z_m"]

        track_memory[key] = track
        active_keys.add(key)

        detection["track_key"] = key
        detection["stable_frames"] = stable_frames
        detection["ema_grasp_X_m"] = track["ema_grasp_X_m"]
        detection["ema_grasp_Y_m"] = track["ema_grasp_Y_m"]
        detection["ema_grasp_Z_m"] = track["ema_grasp_Z_m"]

    stale_keys = [
        key
        for key, track in track_memory.items()
        if frame_idx - track.get("last_seen_frame", -999999) > stale_after_frames
    ]
    for key in stale_keys:
        del track_memory[key]


def compute_edge_margin_score(detection, image_shape, safe_margin_px=90.0):
    img_h, img_w = image_shape[:2]
    edge_distance_px = min(
        detection["grasp_px"],
        detection["grasp_py"],
        img_w - 1 - detection["grasp_px"],
        img_h - 1 - detection["grasp_py"],
    )
    return float(np.clip(edge_distance_px / safe_margin_px, 0.0, 1.0))


def compute_isolation_score(detection, detections, safe_distance_px=150.0):
    if len(detections) <= 1:
        return 1.0

    this_xy = np.array(
        [detection["grasp_px"], detection["grasp_py"]], dtype=np.float32
    )
    min_distance_px = float("inf")

    for other in detections:
        if other is detection:
            continue
        other_xy = np.array([other["grasp_px"], other["grasp_py"]], dtype=np.float32)
        distance_px = float(np.linalg.norm(this_xy - other_xy))
        min_distance_px = min(min_distance_px, distance_px)

    if min_distance_px == float("inf"):
        return 1.0

    return float(np.clip(min_distance_px / safe_distance_px, 0.0, 1.0))


def attach_pickability_scores(detections, image_shape):
    if not detections:
        return

    grasp_depths = [detection["grasp_Z_m"] for detection in detections]
    min_depth = min(grasp_depths)
    max_depth = max(grasp_depths)
    depth_span = max(max_depth - min_depth, 1e-6)

    for detection in detections:
        stability_score = float(
            np.clip(detection.get("stable_frames", 0) / 6.0, 0.0, 1.0)
        )
        confidence_score = float(
            np.clip(detection["cap_confidence"] / 0.45, 0.0, 1.0)
        )
        edge_score = compute_edge_margin_score(detection, image_shape)
        isolation_score = compute_isolation_score(detection, detections)
        depth_score = float(
            np.clip((max_depth - detection["grasp_Z_m"]) / depth_span, 0.0, 1.0)
        )

        pick_score = (
            0.38 * stability_score
            + 0.27 * confidence_score
            + 0.18 * isolation_score
            + 0.12 * edge_score
            + 0.05 * depth_score
        )

        detection["stability_score"] = stability_score
        detection["confidence_score"] = confidence_score
        detection["edge_score"] = edge_score
        detection["isolation_score"] = isolation_score
        detection["depth_score"] = depth_score
        detection["pick_score"] = pick_score


def select_best_detection(detections):
    if not detections:
        return None

    pickable_detections = []
    for detection in detections:
        enough_confidence = detection["cap_confidence"] >= MIN_PICK_CONFIDENCE
        enough_edge_margin = detection["edge_score"] >= MIN_EDGE_SCORE
        detection["is_pickable"] = bool(enough_confidence and enough_edge_margin)
        if detection["is_pickable"]:
            pickable_detections.append(detection)

    if not pickable_detections:
        pickable_detections = detections

    return max(
        pickable_detections,
        key=lambda detection: (
            detection["pick_score"],
            detection.get("stable_frames", 0),
            detection["cap_confidence"],
            -detection["grasp_Z_m"],
        ),
    )


def draw_detection(color_image, detection):
    box_color = detection["box_color"]
    box_points = detection["box_points"]

    cv2.drawContours(color_image, [box_points], 0, box_color, 2)
    cv2.circle(color_image, (detection["cx"], detection["cy"]), 4, (0, 0, 255), -1)

    axis_u = detection["axis_u"]
    line_half_length = int(detection["long_side"] * 0.35)
    dx = int(axis_u[0] * line_half_length)
    dy = int(axis_u[1] * line_half_length)
    cv2.line(
        color_image,
        (detection["cx"] - dx, detection["cy"] - dy),
        (detection["cx"] + dx, detection["cy"] + dy),
        box_color,
        2,
    )

    cv2.circle(
        color_image, (detection["cap_tip_px"], detection["cap_tip_py"]), 6, (255, 255, 255), 2
    )
    cv2.circle(
        color_image, (detection["grasp_px"], detection["grasp_py"]), 6, (0, 0, 255), -1
    )
    cv2.putText(
        color_image,
        "G",
        (detection["grasp_px"] + 8, detection["grasp_py"] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    text_x = int(np.min(box_points[:, 0]))
    text_y = int(np.min(box_points[:, 1])) - 12

    cv2.putText(
        color_image,
        detection["label"],
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2,
    )

    cv2.putText(
        color_image,
        f"A={detection['angle_deg']:.1f} deg",
        (detection["cx"] + 10, detection["cy"] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2,
    )

    cv2.putText(
        color_image,
        f"GZ={detection['grasp_Z_m']:.3f} m",
        (detection["cx"] + 10, detection["cy"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        color_image,
        f"GX={detection['grasp_X_m']:.3f}, GY={detection['grasp_Y_m']:.3f}",
        (detection["cx"] + 10, detection["cy"] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        color_image,
        f"parts={detection['part_count']} conf={detection['cap_confidence']:.2f}",
        (detection["cx"] + 10, detection["cy"] + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    if "pick_score" in detection:
        cv2.putText(
            color_image,
            f"score={detection['pick_score']:.2f} st={detection.get('stable_frames', 0)}",
            (detection["cx"] + 10, detection["cy"] + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            2,
        )


def create_camera_session():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth,
        DEPTH_STREAM_SIZE[0],
        DEPTH_STREAM_SIZE[1],
        rs.format.z16,
        DEPTH_FPS,
    )
    config.enable_stream(
        rs.stream.color,
        COLOR_STREAM_SIZE[0],
        COLOR_STREAM_SIZE[1],
        rs.format.bgr8,
        COLOR_FPS,
    )

    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)

    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_sensor = None
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        if "RGB" in sensor_name or "Color" in sensor_name:
            color_sensor = sensor
            break

    if color_sensor is None:
        pipeline.stop()
        raise RuntimeError("Could not find RealSense color sensor.")

    if LOCK_RGB_SETTINGS:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, RGB_EXPOSURE)

        if color_sensor.supports(rs.option.gain):
            color_sensor.set_option(rs.option.gain, RGB_GAIN)

        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, RGB_WHITE_BALANCE)

        print(f"Locked RGB exposure = {color_sensor.get_option(rs.option.exposure)}")
        if color_sensor.supports(rs.option.gain):
            print(f"Locked RGB gain = {color_sensor.get_option(rs.option.gain)}")
        print(
            "Locked RGB white balance = "
            f"{color_sensor.get_option(rs.option.white_balance)}"
        )

    return {
        "pipeline": pipeline,
        "align": align,
        "colorizer": colorizer,
        "depth_scale": depth_scale,
        "warmup_done": False,
    }


def close_camera_session(session, destroy_windows=True):
    if session is None:
        return

    pipeline = session.get("pipeline")
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass

    if destroy_windows:
        cv2.destroyAllWindows()


def process_frame(session):
    frames = session["pipeline"].wait_for_frames()
    aligned_frames = session["align"].process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_color_frame = session["colorizer"].colorize(depth_frame)
    depth_colormap = np.asanyarray(depth_color_frame.get_data())

    raw_color_image = np.asanyarray(color_frame.get_data()).copy()
    display_image = raw_color_image.copy()
    hsv = cv2.cvtColor(raw_color_image, cv2.COLOR_BGR2HSV)

    if USE_FIXED_ROI:
        roi_mask, roi_polygon_px = build_roi_mask(depth_image.shape, ROI_POLYGON_NORM)
    else:
        roi_mask = np.full(depth_image.shape, 255, dtype=np.uint8)
        roi_polygon_px = None

    blue_mask = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    green_mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    yellow_mask = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)

    combined_mask = cv2.bitwise_or(blue_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    combined_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, COMBINED_CLOSE_SIZE
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, combined_close_kernel
    )
    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

    near_threshold_raw = int(NEAR_THRESHOLD_M / session["depth_scale"])
    near_mask = np.where(
        (depth_image > 0) & (depth_image < near_threshold_raw),
        255,
        0,
    ).astype(np.uint8)
    near_mask = cv2.bitwise_and(near_mask, roi_mask)

    foreground_mask = cv2.bitwise_and(combined_mask, near_mask)
    foreground_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, FOREGROUND_CLOSE_SIZE
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_CLOSE, foreground_close_kernel
    )
    foreground_dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, FOREGROUND_DILATE_SIZE
    )
    foreground_mask = cv2.dilate(
        foreground_mask, foreground_dilate_kernel, iterations=1
    )
    foreground_mask = cv2.bitwise_and(foreground_mask, roi_mask)

    contours, _ = cv2.findContours(
        foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_candidates = []
    for contour in contours:
        candidate = build_raw_candidate(
            contour,
            depth_image,
            session["depth_scale"],
            depth_intrin,
            blue_mask,
            green_mask,
            yellow_mask,
            foreground_mask.shape,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    grouped_candidates = group_raw_candidates(raw_candidates)
    detections = []

    for candidate_group in grouped_candidates:
        merged_contour = build_merged_contour(candidate_group, foreground_mask.shape)
        if merged_contour is None:
            continue

        detection = build_final_candidate(
            candidate_group,
            merged_contour,
            depth_image,
            session["depth_scale"],
            depth_intrin,
            blue_mask,
            green_mask,
            yellow_mask,
            foreground_mask.shape,
        )
        if detection is None:
            continue

        detections.append(detection)
        draw_detection(display_image, detection)

    if roi_polygon_px is not None:
        cv2.polylines(display_image, [roi_polygon_px], True, (255, 255, 255), 2)

    return {
        "display_image": display_image,
        "blue_mask": blue_mask,
        "green_mask": green_mask,
        "yellow_mask": yellow_mask,
        "combined_mask": combined_mask,
        "foreground_mask": foreground_mask,
        "depth_colormap": depth_colormap,
        "detections": detections,
    }


def show_debug_views(frame_data):
    cv2.imshow("Color", frame_data["display_image"])
    cv2.imshow("Blue Mask Raw", frame_data["blue_mask"])
    cv2.imshow("Green Mask Raw", frame_data["green_mask"])
    cv2.imshow("Yellow Mask Raw", frame_data["yellow_mask"])
    cv2.imshow("Combined Mask", frame_data["combined_mask"])
    cv2.imshow("Foreground Mask", frame_data["foreground_mask"])


def draw_locked_target_banner(color_image, target):
    cv2.putText(
        color_image,
        "TARGET",
        (target["grasp_px"] + 10, target["grasp_py"] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def print_detection_list(detections):
    print("Detections:")
    for detection in detections:
        print(
            f'{detection["label"]}  parts={detection["part_count"]}  '
            f'cx={detection["cx"]}  cy={detection["cy"]}  '
            f'GX={detection["grasp_X_m"]:.3f}  GY={detection["grasp_Y_m"]:.3f}  '
            f'GZ={detection["grasp_Z_m"]:.3f}  angle={detection["angle_deg"]:.1f}  '
            f'cap={detection["cap_end"]}  conf={detection["cap_confidence"]:.2f}  '
            f'st={detection.get("stable_frames", 0)}  '
            f'edge={detection.get("edge_score", 0.0):.2f}  '
            f'iso={detection.get("isolation_score", 0.0):.2f}  '
            f'score={detection.get("pick_score", 0.0):.2f}'
        )


def print_candidate(candidate, stable_count):
    print("CANDIDATE:")
    print(
        f'{candidate["label"]}  grasp_px={candidate["grasp_px"]}  '
        f'grasp_py={candidate["grasp_py"]}  '
        f'GX={candidate["grasp_X_m"]:.3f}  GY={candidate["grasp_Y_m"]:.3f}  '
        f'GZ={candidate["grasp_Z_m"]:.3f}  angle={candidate["angle_deg"]:.1f}  '
        f'parts={candidate["part_count"]}  stable_count={stable_count}  '
        f'pick_score={candidate.get("pick_score", 0.0):.2f}  '
        f'st={candidate.get("stable_frames", 0)}  '
        f'conf={candidate.get("confidence_score", 0.0):.2f}  '
        f'iso={candidate.get("isolation_score", 0.0):.2f}  '
        f'edge={candidate.get("edge_score", 0.0):.2f}'
    )


def print_target(target):
    print("TARGET:")
    print(
        f'{target["label"]}  grasp_px={target["grasp_px"]}  '
        f'grasp_py={target["grasp_py"]}  '
        f'GX={target["grasp_X_m"]:.3f}  GY={target["grasp_Y_m"]:.3f}  '
        f'GZ={target["grasp_Z_m"]:.3f}  angle={target["angle_deg"]:.1f}  '
        f'parts={target["part_count"]}'
    )
    print("-" * 50)


def snapshot_target(target):
    return {
        "label": str(target["label"]),
        "grasp_X_m": float(target["grasp_X_m"]),
        "grasp_Y_m": float(target["grasp_Y_m"]),
        "grasp_Z_m": float(target["grasp_Z_m"]),
        "angle_deg": float(target["angle_deg"]),
        "grasp_px": int(target["grasp_px"]),
        "grasp_py": int(target["grasp_py"]),
        "pick_score": float(target.get("pick_score", 0.0)),
        "stable_frames": int(target.get("stable_frames", 0)),
    }


def run_detection_loop(
    timeout_s=None,
    show_windows=True,
    stop_on_lock=False,
    session=None,
    verbose=True,
    destroy_windows_on_exit=True,
    frame_skip_count=None,
    required_stable_frames=None,
    max_lost_frames=None,
):
    owns_session = session is None
    if session is None:
        session = create_camera_session()

    frame_count = 0
    last_target_key = None
    stable_count = 0
    locked_target = None
    lost_target_count = 0
    track_memory = {}
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
    effective_frame_skip_count = max(
        0,
        FRAME_SKIP_COUNT if frame_skip_count is None else int(frame_skip_count),
    )
    effective_required_stable_frames = max(
        1,
        REQUIRED_STABLE_FRAMES
        if required_stable_frames is None
        else int(required_stable_frames),
    )
    effective_max_lost_frames = max(
        0,
        MAX_LOST_FRAMES if max_lost_frames is None else int(max_lost_frames),
    )
    warmup_frames_remaining = (
        0
        if session.get("warmup_done", False)
        else effective_frame_skip_count
    )

    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return None

            frame_data = process_frame(session)
            frame_count += 1
            if frame_data is None:
                continue

            if warmup_frames_remaining > 0:
                warmup_frames_remaining -= 1
                if warmup_frames_remaining == 0:
                    session["warmup_done"] = True
                continue

            session["warmup_done"] = True

            detections = frame_data["detections"]

            if detections:
                update_detection_tracks(detections, track_memory, frame_count)
                attach_pickability_scores(
                    detections, frame_data["foreground_mask"].shape
                )

                if verbose:
                    print_detection_list(detections)

                candidate = select_best_detection(detections)
                target_key = candidate["track_key"]

                if target_key == last_target_key:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_target_key = target_key

                if stable_count >= effective_required_stable_frames:
                    locked_target = candidate.copy()
                    lost_target_count = 0

                if verbose:
                    print_candidate(candidate, stable_count)
            else:
                stable_count = 0
                last_target_key = None

                if locked_target is not None:
                    lost_target_count += 1
                    if lost_target_count > effective_max_lost_frames:
                        locked_target = None
                        lost_target_count = 0

            if locked_target is not None:
                draw_locked_target_banner(frame_data["display_image"], locked_target)
                if verbose:
                    print_target(locked_target)

                if stop_on_lock:
                    if show_windows:
                        show_debug_views(frame_data)
                        cv2.waitKey(1)
                    return snapshot_target(locked_target)

            if show_windows:
                show_debug_views(frame_data)
                if cv2.waitKey(1) == 27:
                    return None

    finally:
        if owns_session:
            close_camera_session(session, destroy_windows=destroy_windows_on_exit)
        elif destroy_windows_on_exit:
            cv2.destroyAllWindows()


def wait_for_locked_target(
    timeout_s,
    show_windows=True,
    session=None,
    verbose=True,
    destroy_windows_on_exit=True,
    frame_skip_count=None,
    required_stable_frames=None,
    max_lost_frames=None,
):
    return run_detection_loop(
        timeout_s=timeout_s,
        show_windows=show_windows,
        stop_on_lock=True,
        session=session,
        verbose=verbose,
        destroy_windows_on_exit=destroy_windows_on_exit,
        frame_skip_count=frame_skip_count,
        required_stable_frames=required_stable_frames,
        max_lost_frames=max_lost_frames,
    )


def main():
    run_detection_loop(timeout_s=None, show_windows=True, stop_on_lock=False)


if __name__ == "__main__":
    main()
