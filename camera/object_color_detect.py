import pyrealsense2 as rs
import numpy as np
import cv2

point = None

def on_mouse(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

def get_dominant_color(blue_mask, green_mask, yellow_mask, x, y, w, h):
    blue_count = cv2.countNonZero(blue_mask[y:y+h, x:x+w])
    green_count = cv2.countNonZero(green_mask[y:y+h, x:x+w])
    yellow_count = cv2.countNonZero(yellow_mask[y:y+h, x:x+w])

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

    # Use the median of a small depth window so one noisy pixel does not
    # move the 3D point too much.
    return float(np.median(valid_depths) * depth_scale)

# Create a pipeline object (this manages the camera stream)
pipeline = rs.pipeline()

# Create a config object (this stores which streams/settings we want)
config = rs.config()

# Conservative settings for USB 2.x
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Create a colorizer object (this will help us visualize depth data)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)

# Start the camera
pipeline_profile = pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
frame_count = 0
last_target_key = None
stable_count = 0
required_stable_frames = 3

locked_target = None
lost_target_count = 0
max_lost_frames = 5

near_threshold_m = 0.40
combined_close_size = (5, 27)
foreground_close_size = (5, 31)
foreground_dilate_size = (3, 3)
min_contour_area = 1800
depth_window_size = 5


try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()
        frame_count += 1
        if frame_count < 50:
            continue

        # NEW: perform alignment here
        aligned_frames = align.process(frames)

        # Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # If either frame is missing, skip this loop and wait for the next one
        if not depth_frame or not color_frame:
            continue

        # Convert camera frame objects into NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([95, 90, 60])
        upper_blue = np.array([140, 255, 255])

        lower_green = np.array([35, 60, 50])
        upper_green = np.array([85, 255, 255])

        lower_yellow = np.array([20, 120, 120])
        upper_yellow = np.array([35, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(blue_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

        combined_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, combined_close_size)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, combined_close_kernel)

        near_threshold_raw = int(near_threshold_m / depth_scale)

        near_mask = np.where(
            (depth_image > 0) & (depth_image < near_threshold_raw),
            255,
            0
        ).astype(np.uint8)

        foreground_mask = cv2.bitwise_and(combined_mask, near_mask)

        foreground_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, foreground_close_size)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, foreground_close_kernel)

        foreground_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, foreground_dilate_size)
        foreground_mask = cv2.dilate(foreground_mask, foreground_dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > min_contour_area: 
                x, y, w, h = cv2.boundingRect(cnt)

                img_h, img_w = foreground_mask.shape
                margin = 0
                if x <= margin or y <= margin or x + w >= img_w - margin or y + h >= img_h - margin:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance_m = get_window_median_depth_m(
                        depth_image,
                        depth_scale,
                        cx,
                        cy,
                        depth_window_size,
                    )
                    if distance_m <= 0:
                        continue
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance_m)
                    X_m, Y_m, Z_m = point_3d

                    label, box_color = get_dominant_color(blue_mask, green_mask, yellow_mask, x, y, w, h)
                    detections.append({
                        "label": label,
                        "cx": cx,
                        "cy": cy,
                        "X_m": X_m,
                        "Y_m": Y_m,
                        "Z_m": Z_m,
                    })
                    

                    cv2.rectangle(color_image, (x, y), (x + w, y + h), box_color, 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

                    cv2.putText(
                        color_image,
                        label,
                        (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        box_color,
                        2,
                    )

                    cv2.putText(
                        color_image,
                        f"Z={Z_m:.3f} m",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        color_image,
                        f"X={X_m:.3f}, Y={Y_m:.3f}",
                        (cx + 10, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
        
        # Show both windows
        target = None

        if detections:
            print("Detections:")
            for d in detections:
                print(
                    f'{d["label"]}  cx={d["cx"]}  cy={d["cy"]}  '
                    f'X={d["X_m"]:.3f}  Y={d["Y_m"]:.3f}  Z={d["Z_m"]:.3f}'
                )

            candidate = min(detections, key=lambda d: d["Z_m"])

            target_key = (
                candidate["label"],
                int(candidate["cx"] / 30),
                int(candidate["cy"] / 30),
            )

            if target_key == last_target_key:
                stable_count += 1
            else:
                stable_count = 1
                last_target_key = target_key

            if stable_count >= required_stable_frames:
                locked_target = candidate
                lost_target_count = 0

            print("CANDIDATE:")
            print(
                f'{candidate["label"]}  cx={candidate["cx"]}  cy={candidate["cy"]}  '
                f'X={candidate["X_m"]:.3f}  Y={candidate["Y_m"]:.3f}  Z={candidate["Z_m"]:.3f}  '
                f'stable_count={stable_count}'
            )
        else:
            stable_count = 0
            last_target_key = None

            if locked_target is not None:
                lost_target_count += 1
                if lost_target_count > max_lost_frames:
                    locked_target = None
                    lost_target_count = 0

        if locked_target is not None:
            target = locked_target

            print("TARGET:")
            print(
                f'{target["label"]}  cx={target["cx"]}  cy={target["cy"]}  '
                f'X={target["X_m"]:.3f}  Y={target["Y_m"]:.3f}  Z={target["Z_m"]:.3f}'
            )
            print("-" * 50)

            cv2.putText(
                color_image,
                "TARGET",
                (target["cx"] + 10, target["cy"] + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        
        # Show both windows
        cv2.imshow("Color", color_image)
        cv2.imshow("Combined Mask", combined_mask)
        cv2.imshow("Near Mask", near_mask)
        cv2.imshow("Foreground Mask", foreground_mask)
        cv2.imshow("Depth", depth_colormap)

        # Press ESC to quit
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # Clean shutdown
    pipeline.stop()
    cv2.destroyAllWindows()
