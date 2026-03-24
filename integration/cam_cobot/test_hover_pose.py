try:
    from .fit_table_mapping import cam_to_robot_xy
except ImportError:
    from fit_table_mapping import cam_to_robot_xy


DEFAULT_W_DEG = 62.98


def angle_to_robot_w(object_angle_deg: float) -> float:
    """
    Convert tube angle from vision into robot W angle.
    Fitted from your manual calibration data.
    """
    w = -0.9896761984861228 * object_angle_deg + 150.58477782450237

    while w < 0:
        w += 180.0
    while w >= 180.0:
        w -= 180.0

    return w


def choose_shortest_pick_w(
    raw_pick_w_deg: float, default_w_deg: float = DEFAULT_W_DEG
) -> tuple[float, float]:
    """
    Flat tubes are 180-degree symmetric for grasp, so raw W and raw W +/- 180
    are equivalent. Choose the candidate closest to the default scan wrist
    angle so the robot uses the smallest signed wrist rotation.
    """
    candidates = [
        float(raw_pick_w_deg - 180.0),
        float(raw_pick_w_deg),
        float(raw_pick_w_deg + 180.0),
    ]
    chosen_pick_w_deg = min(
        candidates,
        key=lambda candidate_w_deg: (
            abs(candidate_w_deg - default_w_deg),
            abs(candidate_w_deg),
        ),
    )
    pick_delta_w_deg = float(chosen_pick_w_deg - default_w_deg)
    return float(chosen_pick_w_deg), pick_delta_w_deg


def predict_hover_pose(
    grasp_x_m: float,
    grasp_y_m: float,
    object_angle_deg: float,
    hover_z_mm: float,
    u_deg: float = 0.17,
    v_deg: float = -179.93,
):
    """
    Build a hover pose from:
    - grasp point camera X/Y
    - detected object angle
    - shortest-equivalent wrist W around the latest default scan wrist angle
    """
    robot_x_mm, robot_y_mm = cam_to_robot_xy(grasp_x_m, grasp_y_m)
    raw_pick_w_deg = angle_to_robot_w(object_angle_deg)
    chosen_pick_w_deg, pick_delta_w_deg = choose_shortest_pick_w(raw_pick_w_deg)

    return {
        "X_mm": float(robot_x_mm),
        "Y_mm": float(robot_y_mm),
        "Z_mm": float(hover_z_mm),
        "U_deg": float(u_deg),
        "V_deg": float(v_deg),
        "W_deg": float(chosen_pick_w_deg),
        "W_raw_deg": float(raw_pick_w_deg),
        "W_delta_from_default_deg": float(pick_delta_w_deg),
    }


if __name__ == "__main__":
    # ==========================================
    # REPLACE THESE WITH REAL VALUES FROM VISION
    # ==========================================
    grasp_x_m = -0.011
    grasp_y_m = -0.003
    object_angle_deg = 116.3

    # IMPORTANT:
    # Set this to your own safe hover height above the tube.
    # Do NOT use the grasp Z yet.
    hover_z_mm = 60.0

    pose = predict_hover_pose(
        grasp_x_m=grasp_x_m,
        grasp_y_m=grasp_y_m,
        object_angle_deg=object_angle_deg,
        hover_z_mm=hover_z_mm,
    )
    print(f"Default W_deg: {DEFAULT_W_DEG:.2f}")
    print(f"Raw W_deg: {pose['W_raw_deg']:.2f}")
    print(f"Chosen W_deg (shortest): {pose['W_deg']:.2f}")
    print(f"Delta from default: {pose['W_delta_from_default_deg']:+.2f}")

    print("Predicted hover pose:")
    for k, v in pose.items():
        print(f"{k}: {v:.2f}")
