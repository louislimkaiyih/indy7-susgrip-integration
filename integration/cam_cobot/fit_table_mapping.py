import numpy as np

# ============================================================
# Camera XY (meters)  ->  Robot XY (mm)
# New fixed-scan calibration dataset collected with TCP Z = 169 mm
#
# IMPORTANT:
# - This is still a fixed-scan-pose mapping.
# - Do NOT include SCAN, HOME, or PLACE_* in the fit.
# - Only use table reference points.
# - This model predicts robot X,Y only.
# - Z / U / V / W should be handled separately.
# - In this new setup, moving UP means +Z.
# ============================================================

# Camera XY in meters
cam_xy = np.array(
    [
        [-0.2165, -0.1204],  # Top Left
        [-0.0199, -0.1200],  # Top Mid
        [0.2121, -0.1164],  # Top Right
        [-0.2153, -0.0005],  # Mid Left
        [-0.0116, 0.0047],  # Mid
        [0.2097, 0.0042],  # Mid Right
        [-0.2144, 0.1132],  # Bottom Left
        [-0.0064, 0.1109],  # Bottom Mid
        [0.2076, 0.1102],  # Bottom Right
    ],
    dtype=float,
)

# Robot XY in mm
robot_xy = np.array(
    [
        [-205.49, -216.01],  # Top Left
        [-7.30, -217.06],  # Top Mid
        [223.87, -224.75],  # Top Right
        [-204.52, -338.01],  # Mid Left
        [-0.65, -340.54],  # Mid
        [219.50, -344.84],  # Mid Right
        [-205.78, -450.37],  # Bottom Left
        [4.65, -450.22],  # Bottom Mid
        [217.64, -450.28],  # Bottom Right
    ],
    dtype=float,
)

# Build affine model:
# robot_x = a1*cam_x + a2*cam_y + a3
# robot_y = b1*cam_x + b2*cam_y + b3
A = np.column_stack([cam_xy[:, 0], cam_xy[:, 1], np.ones(len(cam_xy))])

coef_x, _, _, _ = np.linalg.lstsq(A, robot_xy[:, 0], rcond=None)
coef_y, _, _, _ = np.linalg.lstsq(A, robot_xy[:, 1], rcond=None)


def cam_to_robot_xy(cam_x_m, cam_y_m):
    """
    Convert camera-frame XY (meters) to robot-frame XY (mm).
    """
    robot_x_mm = coef_x[0] * cam_x_m + coef_x[1] * cam_y_m + coef_x[2]
    robot_y_mm = coef_y[0] * cam_x_m + coef_y[1] * cam_y_m + coef_y[2]
    return robot_x_mm, robot_y_mm


def predict_robot_pose(
    cam_x_m,
    cam_y_m,
    z_mm=184.45,
    u_deg=0.17,
    v_deg=-179.93,
    w_deg=62.98,
):
    """
    Build a simple robot pose from camera XY.
    Use this for hover / approach testing.

    Notes:
    - z_mm should still be chosen by you depending on:
        * actual grasp depth
        * hover offset
        * safe test height
    - U/V/W defaults are based on the latest scan orientation.
    """
    robot_x_mm, robot_y_mm = cam_to_robot_xy(cam_x_m, cam_y_m)

    return {
        "X_mm": float(robot_x_mm),
        "Y_mm": float(robot_y_mm),
        "Z_mm": float(z_mm),
        "U_deg": float(u_deg),
        "V_deg": float(v_deg),
        "W_deg": float(w_deg),
    }


def print_fit_report():
    predicted = np.array([cam_to_robot_xy(x, y) for x, y in cam_xy])
    errors = np.linalg.norm(predicted - robot_xy, axis=1)

    print("coef_x =", coef_x)
    print("coef_y =", coef_y)
    print()

    for i, err in enumerate(errors, start=1):
        print(
            f"Point {i}: "
            f"predicted=({predicted[i-1, 0]:.2f}, {predicted[i-1, 1]:.2f}) mm, "
            f"actual=({robot_xy[i-1, 0]:.2f}, {robot_xy[i-1, 1]:.2f}) mm, "
            f"error={err:.2f} mm"
        )

    print()
    print(f"Mean error = {errors.mean():.2f} mm")
    print(f"Max error  = {errors.max():.2f} mm")


if __name__ == "__main__":
    print_fit_report()

    # ------------------------------------------------------------
    # Example: replace these with grasp point camera X,Y from vision
    # ------------------------------------------------------------
    test_cam_x = 0.1145
    test_cam_y = 0.0426

    test_robot_x, test_robot_y = cam_to_robot_xy(test_cam_x, test_cam_y)
    print()
    print(f"Example input cam XY = ({test_cam_x:.4f}, {test_cam_y:.4f}) m")
    print(f"Predicted robot XY   = ({test_robot_x:.2f}, {test_robot_y:.2f}) mm")

    test_pose = predict_robot_pose(
        test_cam_x,
        test_cam_y,
        z_mm=184.45,
        u_deg=0.17,
        v_deg=-179.93,
        w_deg=62.98,
    )
    print()
    print("Predicted robot pose:")
    for k, v in test_pose.items():
        print(f"  {k}: {v:.2f}")
