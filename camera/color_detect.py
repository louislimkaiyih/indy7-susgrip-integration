import pyrealsense2 as rs
import numpy as np
import cv2

point = None


def on_mouse(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)


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
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()

        # NEW: perform alignment here
        aligned_frames = align.process(frames)

        # Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # If either frame is missing, skip this loop and wait for the next one
        if not depth_frame or not color_frame:
            continue

        # Convert camera frame objects into NumPy arrays
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([140, 255, 255])

        lower_green = np.array([35, 60, 50])
        upper_green = np.array([85, 255, 255])

        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([35, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        for mask, label, box_color in [
            (blue_mask, "BLUE", (255, 0, 0)),
            (green_mask, "GREEN", (0, 255, 0)),
            (yellow_mask, "YELLOW", (0, 255, 255)),
        ]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)

                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        cv2.rectangle(color_image, (x, y), (x + w, y + h), box_color, 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(
                            color_image,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            box_color,
                            2,
                        )

        combined_mask = cv2.bitwise_or(blue_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

        cv2.imshow("Color", color_image)
        cv2.imshow("Combined Mask", combined_mask)

        # Press ESC to quit
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # Clean shutdown
    pipeline.stop()
    cv2.destroyAllWindows()
