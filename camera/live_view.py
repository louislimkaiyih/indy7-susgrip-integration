import pyrealsense2 as rs
import numpy as np
import cv2


def get_clicked_camera_xyz(depth_frame, pixel):
    x, y = pixel
    depth_m = depth_frame.get_distance(x, y)

    if depth_m <= 0:
        return None

    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth_m)

    return {
        "u": x,
        "v": y,
        "depth_m": depth_m,
        "X_m": X,
        "Y_m": Y,
        "Z_m": Z,
    }


point = None
last_reported_point = None


def on_mouse(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

colorizer = rs.colorizer()
align = rs.align(rs.stream.color)

profile = pipeline.start(config)

device = profile.get_device()
color_sensor = None

for s in device.query_sensors():
    name = s.get_info(rs.camera_info.name)
    if "RGB Camera" in name:
        color_sensor = s
        break

if color_sensor is None:
    raise RuntimeError("Could not find RGB Camera sensor")

# Freeze color settings to match flat_tube_detect.py baseline
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 625)
color_sensor.set_option(rs.option.gain, 0)

color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
color_sensor.set_option(rs.option.white_balance, 5200)
print(f"Locked RGB exposure = {color_sensor.get_option(rs.option.exposure)}")
if color_sensor.supports(rs.option.gain):
    print(f"Locked RGB gain = {color_sensor.get_option(rs.option.gain)}")
print(
    f"Locked RGB white balance = {color_sensor.get_option(rs.option.white_balance)}"
)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        # Draw clicked point if available
        if point is not None:
            cv2.drawMarker(
                display_image,
                point,
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.circle(display_image, point, 10, (0, 255, 0), 2)
            cv2.putText(
                display_image,
                f"({point[0]}, {point[1]})",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Show windows every loop
        cv2.imshow("Color", display_image)
        cv2.setMouseCallback("Color", on_mouse)
        cv2.imshow("Depth", depth_colormap)

        # Print clicked point only once per new click
        if point is not None and point != last_reported_point:
            result = get_clicked_camera_xyz(depth_frame, point)

            if result is None:
                print(f"Clicked pixel ({point[0]}, {point[1]}): invalid depth")
            else:
                print(
                    f"Pixel (u,v)=({result['u']}, {result['v']}) | "
                    f"Depth={result['depth_m']:.4f} m | "
                    f"Camera XYZ=({result['X_m']:.4f}, {result['Y_m']:.4f}, {result['Z_m']:.4f}) m"
                )

            last_reported_point = point

        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
