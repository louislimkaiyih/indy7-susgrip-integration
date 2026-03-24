import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuromeka import IndyDCP3

try:
    from neuromeka import (
        BlendingType,
        EndtoolState,
        JointBaseType,
        OpState,
        StopCategory,
        TaskBaseType,
    )
except Exception:
    BlendingType = None
    EndtoolState = None
    JointBaseType = None
    OpState = None
    StopCategory = None
    TaskBaseType = None

from scripts.camera.flat_tube_detect import (
    close_camera_session,
    create_camera_session,
    wait_for_locked_target,
)
from scripts.integration.cam_cobot.test_hover_pose import (
    choose_shortest_pick_w,
    predict_hover_pose,
)


CURRENT_ROBOT = None


# ============================================================
# constants / poses
# ============================================================

ROBOT_IP = "192.168.1.52" # change if needed
ROBOT_INDEX = 0

ENDTOOL_PORT_OVERRIDE = None
SHOW_VISION_WINDOWS = True

VEL_RATIO = 50
ACC_RATIO = 30
MOTION_TIMEOUT_S = 120.0
STATUS_POLL_S = 0.02

GRIPPER_SETTLE_S = 0.2
INITIAL_GRIPPER_SETTLE_S = 0.5
POST_CLOSE_SETTLE_S = 1.0
POST_PLACE_OPEN_SETTLE_S = 0.5

HOME_J = [146.04, 12.95, 86.46, -0.76, 83.43, -7.78]
SCAN_J = [68.75, 7.65, 109.00, 0.18, 63.26, 5.69]
SCAN_T = [39.45, -416.17, 192.48, 0.17, -179.93, 62.98]

# Use the latest taught scan pose as the wrist reference, but keep the
# simplified fixed motion orientation requested for pick/place execution.
FIXED_U_DEG = 0.0
FIXED_V_DEG = 180.0
DEFAULT_W_DEG = 62.98

# Corrected pick plane for the latest setup. Keep the previous practical
# offsets so hover and lift stay consistent relative to the new grasp height.
PICK_HOVER_Z_MM = 56.59
PICK_GRASP_Z_MM = 16.5
PICK_LIFT_Z_MM = 150.09
PLACE_RELEASE_Z_MM = 140.12
POST_PLACE_LIFT_Z_MM = 156.62
NO_TARGET_TIMEOUT_S = 5.0
VISION_FRAME_SKIP_COUNT = 18
VISION_REQUIRED_STABLE_FRAMES = 3
VISION_MAX_LOST_FRAMES = 3

PLACE_POSE_BY_COLOR = {
    "BLUE": {"X_mm": 14.73, "Y_mm": -679.49, "Z_mm": 140.12},
    "YELLOW": {"X_mm": 102.99, "Y_mm": -679.46, "Z_mm": 140.10},
    "GREEN": {"X_mm": -79.55, "Y_mm": -679.55, "Z_mm": 140.12},
}

BAD_OPSTATE_FALLBACK = {8, 2, 15, 9}
STAGE_CHOICES = [
    "home",
    "scan",
    "gripper",
    "vision",
    "hover",
    "descend",
    "pick",
    "rotate_back",
    "place_blue",
    "place_yellow",
    "place_green",
    "full_cycle",
]
PLACE_STAGE_TO_LABEL = {
    "place_blue": "BLUE",
    "place_yellow": "YELLOW",
    "place_green": "GREEN",
}
VISION_STAGES = {"vision", "hover", "descend", "pick", "rotate_back", "full_cycle"}


class AbortRunError(RuntimeError):
    def __init__(self, step_label: str, message: str, holding_object: bool = False):
        super().__init__(message)
        self.step_label = step_label
        self.holding_object = holding_object


# ============================================================
# robot helpers
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log(f"ERROR: {message}")
    raise SystemExit(exit_code)


def opstate_name(op: int) -> str:
    if OpState is None:
        return str(op)

    for name in dir(OpState):
        if not name.isupper():
            continue
        try:
            if getattr(OpState, name) == op:
                return name
        except Exception:
            continue
    return str(op)


def connect_robot(robot_ip: str, index: int) -> IndyDCP3:
    try:
        log(f"connecting to robot {robot_ip} (index={index})")
        indy = IndyDCP3(robot_ip=robot_ip, index=index)
        control_info = indy.get_control_info()
        robot_data = indy.get_robot_data()
    except Exception as exc:
        raise AbortRunError("connect_robot", f"failed to connect or read robot state: {exc!r}")

    log(f"control info: {control_info}")
    log(
        f"robot op_state={robot_data.get('op_state')} "
        f"({opstate_name(int(robot_data.get('op_state', -1)))}) "
        f"sim_mode={robot_data.get('sim_mode')}"
    )
    return indy


def get_robot_snapshot(indy: IndyDCP3, step_label: str) -> dict:
    try:
        robot_data = indy.get_robot_data()
    except Exception as exc:
        raise AbortRunError(step_label, f"get_robot_data failed: {exc!r}")

    try:
        motion_data = indy.get_motion_data()
    except Exception as exc:
        raise AbortRunError(step_label, f"get_motion_data failed: {exc!r}")

    violation_data = None
    try:
        violation_data = indy.get_violation_data()
    except Exception:
        pass

    servo_data = None
    try:
        servo_data = indy.get_servo_data()
    except Exception:
        pass

    return {
        "robot_data": robot_data,
        "motion_data": motion_data,
        "violation_data": violation_data,
        "servo_data": servo_data,
    }


def extract_bad_op_state(snapshot: dict) -> Optional[str]:
    robot_data = snapshot["robot_data"]
    try:
        op_state = int(robot_data.get("op_state", -1))
    except Exception:
        return "robot op_state is missing or invalid"

    if OpState is not None:
        try:
            bad_states = {
                int(OpState.COLLISION),
                int(OpState.VIOLATE),
                int(OpState.VIOLATE_HARD),
                int(OpState.STOP_AND_OFF),
            }
        except Exception:
            bad_states = BAD_OPSTATE_FALLBACK
    else:
        bad_states = BAD_OPSTATE_FALLBACK

    if op_state in bad_states:
        return f"bad op_state={op_state} ({opstate_name(op_state)})"
    return None


def extract_violation_message(snapshot: dict) -> Optional[str]:
    violation_data = snapshot.get("violation_data")
    if not violation_data:
        return None

    violation_code = str(violation_data.get("violation_code", "0"))
    violation_str = str(violation_data.get("violation_str", "")).strip()

    if violation_code not in {"0", "", "None"}:
        return f"violation_code={violation_code} violation={violation_data}"
    if violation_str not in {"", "None"}:
        return f"violation={violation_data}"
    return None


def extract_servo_problem(snapshot: dict) -> Optional[str]:
    servo_data = snapshot.get("servo_data")
    if not servo_data:
        return None

    servo_actives = servo_data.get("servo_actives")
    if not servo_actives:
        return None

    if any(not bool(active) for active in servo_actives):
        return f"servo_actives={servo_actives}"
    return None


def validate_robot_snapshot(
    snapshot: dict,
    step_label: str,
    allow_in_motion: bool,
    holding_object: bool = False,
) -> None:
    bad_op_state = extract_bad_op_state(snapshot)
    if bad_op_state:
        raise AbortRunError(step_label, bad_op_state, holding_object=holding_object)

    violation_message = extract_violation_message(snapshot)
    if violation_message:
        raise AbortRunError(step_label, violation_message, holding_object=holding_object)

    servo_problem = extract_servo_problem(snapshot)
    if servo_problem:
        raise AbortRunError(step_label, servo_problem, holding_object=holding_object)

    if allow_in_motion:
        return

    motion_data = snapshot["motion_data"]
    is_in_motion = bool(motion_data.get("is_in_motion", False))
    has_motion = bool(motion_data.get("has_motion", False))
    if is_in_motion or has_motion:
        raise AbortRunError(
            step_label,
            f"robot is already in motion. motion_data={motion_data}",
            holding_object=holding_object,
        )


def ensure_robot_ready(
    indy: IndyDCP3,
    step_label: str,
    allow_in_motion: bool = False,
    holding_object: bool = False,
) -> dict:
    snapshot = get_robot_snapshot(indy, step_label)
    validate_robot_snapshot(
        snapshot,
        step_label=step_label,
        allow_in_motion=allow_in_motion,
        holding_object=holding_object,
    )
    return snapshot


def wait_motion_done(
    indy: IndyDCP3,
    label: str,
    timeout_s: float = MOTION_TIMEOUT_S,
    holding_object: bool = False,
) -> None:
    start_time = time.monotonic()
    motion_started = False

    while True:
        snapshot = ensure_robot_ready(
            indy,
            step_label=f"{label}/wait_motion_done",
            allow_in_motion=True,
            holding_object=holding_object,
        )
        motion_data = snapshot["motion_data"]
        is_in_motion = bool(motion_data.get("is_in_motion", False))
        is_target_reached = bool(motion_data.get("is_target_reached", False))
        has_motion = bool(motion_data.get("has_motion", False))

        if is_in_motion or has_motion:
            motion_started = True

        if motion_started and (not is_in_motion) and is_target_reached:
            return

        if (
            (not motion_started)
            and (time.monotonic() - start_time > 0.25)
            and (not is_in_motion)
            and is_target_reached
        ):
            return

        if time.monotonic() - start_time > timeout_s:
            raise AbortRunError(
                label,
                f"motion timed out. motion_data={motion_data}",
                holding_object=holding_object,
            )

        time.sleep(STATUS_POLL_S)


def validate_joint_target(joints_deg, label: str) -> list[float]:
    if len(joints_deg) != 6:
        raise AbortRunError(label, f"expected 6 joint values, got {len(joints_deg)}")

    validated = []
    for index, value in enumerate(joints_deg, start=1):
        try:
            joint_value = float(value)
        except Exception as exc:
            raise AbortRunError(label, f"joint {index} is invalid: {exc!r}")
        if not math.isfinite(joint_value):
            raise AbortRunError(label, f"joint {index} is not finite: {joint_value}")
        validated.append(joint_value)
    return validated


def movej_abs(
    indy: IndyDCP3,
    joints_deg,
    label: str,
    dry_run: bool,
    holding_object: bool = False,
) -> None:
    validated_joints = validate_joint_target(joints_deg, label)
    kwargs = {
        "jtarget": validated_joints,
        "base_type": JointBaseType.ABSOLUTE if JointBaseType is not None else 0,
        "vel_ratio": VEL_RATIO,
        "acc_ratio": ACC_RATIO,
        "blending_type": BlendingType.NONE if BlendingType is not None else 0,
        "blending_radius": 0.0,
    }

    if dry_run:
        log(f"[DRY RUN] movej {label}: {kwargs}")
        return

    ensure_robot_ready(indy, f"{label}/pre_movej", holding_object=holding_object)
    try:
        indy.movej(**kwargs)
    except Exception as exc:
        raise AbortRunError(label, f"movej failed: {exc!r}", holding_object=holding_object)
    wait_motion_done(indy, label, holding_object=holding_object)


def movel_abs(
    indy: IndyDCP3,
    pose_mm_deg,
    label: str,
    dry_run: bool,
    holding_object: bool = False,
) -> None:
    validated_pose = list(pose_mm_deg)
    kwargs = {
        "ttarget": validated_pose,
        "base_type": TaskBaseType.ABSOLUTE if TaskBaseType is not None else 0,
        "vel_ratio": VEL_RATIO,
        "acc_ratio": ACC_RATIO,
        "blending_type": BlendingType.NONE if BlendingType is not None else 0,
        "blending_radius": 0.0,
        "bypass_singular": False,
    }

    if dry_run:
        log(f"[DRY RUN] movel {label}: {kwargs}")
        return

    ensure_robot_ready(indy, f"{label}/pre_movel", holding_object=holding_object)
    try:
        indy.movel(**kwargs)
    except Exception as exc:
        raise AbortRunError(label, f"movel failed: {exc!r}", holding_object=holding_object)
    wait_motion_done(indy, label, holding_object=holding_object)


def stop_motion_best_effort(indy: IndyDCP3) -> None:
    try:
        if StopCategory is not None:
            indy.stop_motion(StopCategory.CAT0)
        else:
            indy.stop_motion(0)
    except Exception:
        pass


def safe_sleep_with_monitoring(
    indy: IndyDCP3,
    step_label: str,
    seconds: float,
    dry_run: bool,
    holding_object: bool = False,
) -> None:
    if dry_run:
        log(f"[DRY RUN] sleep({seconds})")
        return

    start_time = time.monotonic()
    while True:
        ensure_robot_ready(
            indy,
            step_label=f"{step_label}/sleep",
            allow_in_motion=False,
            holding_object=holding_object,
        )
        elapsed = time.monotonic() - start_time
        if elapsed >= seconds:
            return
        time.sleep(min(STATUS_POLL_S, seconds - elapsed))


def validate_pose_dict(pose_dict: dict, label: str) -> dict:
    validated = dict(pose_dict)
    for key in ("X_mm", "Y_mm", "Z_mm", "U_deg", "V_deg", "W_deg"):
        if key not in validated:
            raise AbortRunError(label, f"pose is missing required field {key!r}")
        try:
            value = float(validated[key])
        except Exception as exc:
            raise AbortRunError(label, f"{key} is invalid: {exc!r}")
        if not math.isfinite(value):
            raise AbortRunError(label, f"{key} is not finite: {value}")
        validated[key] = value
    return validated


def pose_dict_to_list(pose_dict: dict) -> list[float]:
    return [
        float(pose_dict["X_mm"]),
        float(pose_dict["Y_mm"]),
        float(pose_dict["Z_mm"]),
        float(pose_dict["U_deg"]),
        float(pose_dict["V_deg"]),
        float(pose_dict["W_deg"]),
    ]


# ============================================================
# gripper helpers
# ============================================================

def detect_endtool_port(indy: IndyDCP3, override_port: Optional[str]) -> str:
    if override_port:
        return override_port

    try:
        data = indy.get_endtool_do()
    except Exception as exc:
        raise AbortRunError("detect_endtool_port", f"get_endtool_do failed: {exc!r}")

    signals = data.get("signals", [])
    if not signals:
        raise AbortRunError("detect_endtool_port", "get_endtool_do returned no endtool signals")

    ports = [str(signal.get("port", "")) for signal in signals]
    if "C" in ports:
        return "C"
    if "A" in ports:
        return "A"
    return ports[0]


def set_gripper_open(
    indy: IndyDCP3,
    port: str,
    label: str,
    dry_run: bool,
    holding_object: bool = False,
) -> None:
    if dry_run:
        log(f"[DRY RUN] endtool DO.0 open {label}: port={port} state=HIGH_PNP")
        return

    if EndtoolState is None:
        raise AbortRunError(label, "EndtoolState enum is not available", holding_object)

    ensure_robot_ready(indy, f"{label}/pre_io", holding_object=holding_object)
    signal = [{"port": port, "states": [EndtoolState.HIGH_PNP]}]
    try:
        indy.set_endtool_do(signal)
    except Exception as exc:
        raise AbortRunError(label, f"set_endtool_do open failed: {exc!r}", holding_object)


def set_gripper_close(
    indy: IndyDCP3,
    port: str,
    label: str,
    dry_run: bool,
    holding_object: bool = False,
) -> None:
    if dry_run:
        log(f"[DRY RUN] endtool DO.0 close {label}: port={port} state=LOW_PNP")
        return

    if EndtoolState is None:
        raise AbortRunError(label, "EndtoolState enum is not available", holding_object)

    ensure_robot_ready(indy, f"{label}/pre_io", holding_object=holding_object)
    signal = [{"port": port, "states": [EndtoolState.LOW_PNP]}]
    try:
        indy.set_endtool_do(signal)
    except Exception as exc:
        raise AbortRunError(label, f"set_endtool_do close failed: {exc!r}", holding_object)


# ============================================================
# vision target acquisition
# ============================================================

def acquire_locked_target(timeout_s: float, show_windows: bool, vision_session=None):
    # Keep the shared RealSense session open, but force a short reacquire
    # settle each time we ask for a new target after robot motion.
    if vision_session is not None:
        vision_session["warmup_done"] = False

    return wait_for_locked_target(
        timeout_s=timeout_s,
        show_windows=show_windows,
        session=vision_session,
        verbose=False,
        destroy_windows_on_exit=False,
        frame_skip_count=VISION_FRAME_SKIP_COUNT,
        required_stable_frames=VISION_REQUIRED_STABLE_FRAMES,
        max_lost_frames=VISION_MAX_LOST_FRAMES,
    )


def validate_target_snapshot(target) -> dict:
    if not isinstance(target, dict):
        raise AbortRunError("vision_target", f"expected dict target snapshot, got {type(target)!r}")

    validated = dict(target)
    label = str(validated.get("label", "")).strip().upper()
    if not label:
        raise AbortRunError("vision_target", "target label is missing")
    validated["label"] = label

    for key in ("grasp_X_m", "grasp_Y_m", "grasp_Z_m", "angle_deg"):
        if key not in validated:
            raise AbortRunError("vision_target", f"target field {key!r} is missing")
        try:
            value = float(validated[key])
        except Exception as exc:
            raise AbortRunError("vision_target", f"target field {key!r} is invalid: {exc!r}")
        if not math.isfinite(value):
            raise AbortRunError("vision_target", f"target field {key!r} is not finite: {value}")
        validated[key] = value

    if "pick_score" in validated:
        try:
            validated["pick_score"] = float(validated["pick_score"])
        except Exception:
            pass
    if "stable_frames" in validated:
        try:
            validated["stable_frames"] = int(validated["stable_frames"])
        except Exception:
            pass

    return validated


# ============================================================
# mapping + angle conversion
# ============================================================

def build_pick_hover_pose(target_snapshot: dict) -> dict:
    try:
        pose = predict_hover_pose(
            grasp_x_m=target_snapshot["grasp_X_m"],
            grasp_y_m=target_snapshot["grasp_Y_m"],
            object_angle_deg=target_snapshot["angle_deg"],
            hover_z_mm=PICK_HOVER_Z_MM,
            u_deg=FIXED_U_DEG,
            v_deg=FIXED_V_DEG,
        )
    except Exception as exc:
        raise AbortRunError("build_pick_hover_pose", f"failed to build hover pose: {exc!r}")
    return validate_pose_dict(pose, "build_pick_hover_pose")


def build_pose_with_z(base_pose: dict, z_mm: float, label: str) -> dict:
    pose = dict(base_pose)
    pose["Z_mm"] = float(z_mm)
    return validate_pose_dict(pose, label)


def build_pose_with_w(base_pose: dict, w_deg: float, label: str) -> dict:
    pose = dict(base_pose)
    pose["W_deg"] = float(w_deg)
    return validate_pose_dict(pose, label)


def build_place_pose(label: str, z_mm: float, w_deg: float) -> dict:
    try:
        place_pose = PLACE_POSE_BY_COLOR[label]
    except KeyError as exc:
        raise AbortRunError("build_place_pose", f"unsupported color label: {label!r}") from exc

    pose = {
        "X_mm": float(place_pose["X_mm"]),
        "Y_mm": float(place_pose["Y_mm"]),
        "Z_mm": float(z_mm),
        "U_deg": float(FIXED_U_DEG),
        "V_deg": float(FIXED_V_DEG),
        "W_deg": float(w_deg),
    }
    return validate_pose_dict(pose, f"build_place_pose_{label}")


# ============================================================
# workflow helpers
# ============================================================

def move_to_home(indy: IndyDCP3, dry_run: bool, holding_object: bool = False) -> None:
    log("moving to home")
    movej_abs(indy, HOME_J, "home_pose", dry_run, holding_object=holding_object)


def move_to_scan(indy: IndyDCP3, dry_run: bool, holding_object: bool = False) -> None:
    log("moving to scan")
    movej_abs(indy, SCAN_J, "scan_pose", dry_run, holding_object=holding_object)


def open_gripper_with_wait(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    label: str,
    settle_s: float,
    holding_object: bool = False,
) -> None:
    log("opening gripper")
    set_gripper_open(
        indy,
        endtool_port,
        label=label,
        dry_run=dry_run,
        holding_object=holding_object,
    )
    safe_sleep_with_monitoring(
        indy,
        step_label=f"{label}_settle",
        seconds=settle_s,
        dry_run=dry_run,
        holding_object=holding_object,
    )


def close_gripper_with_wait(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    label: str,
    settle_s: float,
    holding_object: bool = False,
) -> bool:
    log("closing gripper")
    set_gripper_close(
        indy,
        endtool_port,
        label=label,
        dry_run=dry_run,
        holding_object=holding_object,
    )
    holding_object = True
    safe_sleep_with_monitoring(
        indy,
        step_label=f"{label}_settle",
        seconds=settle_s,
        dry_run=dry_run,
        holding_object=holding_object,
    )
    return holding_object


def prepare_scan_for_detection(indy: IndyDCP3, endtool_port: str, dry_run: bool) -> None:
    # The normal V1 cycle always goes back to scan and opens the gripper before vision.
    move_to_scan(indy, dry_run)
    open_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="scan_open_gripper",
        settle_s=GRIPPER_SETTLE_S,
    )


def log_target_context(context: dict) -> None:
    target = context["target"]
    log(
        f"target {target['label']} | "
        f"GX={target['grasp_X_m']:.3f} GY={target['grasp_Y_m']:.3f} "
        f"GZ={target['grasp_Z_m']:.3f} angle={target['angle_deg']:.1f}"
    )
    log(
        f"mapped target -> robot XY: "
        f"X={context['pick_hover_pose']['X_mm']:.2f} "
        f"Y={context['pick_hover_pose']['Y_mm']:.2f}"
    )
    log(
        f"wrist plan -> raw_pick_w={context['raw_pick_w_deg']:.2f} "
        f"chosen_pick_w={context['chosen_pick_w_deg']:.2f} "
        f"pick_delta_w_deg={context['pick_delta_w_deg']:+.2f}"
    )
    log(
        "lift+return target -> "
        f"Z={context['pick_lift_rotate_back_pose']['Z_mm']:.2f} "
        f"W={context['pick_lift_rotate_back_pose']['W_deg']:.2f}"
    )


def build_pick_context(target_snapshot: dict) -> dict:
    raw_pick_hover_pose = build_pick_hover_pose(target_snapshot)
    raw_pick_w_deg = float(raw_pick_hover_pose.get("W_raw_deg", raw_pick_hover_pose["W_deg"]))
    chosen_pick_w_deg, pick_delta_w_deg = choose_shortest_pick_w(
        raw_pick_w_deg,
        DEFAULT_W_DEG,
    )
    reverse_delta_w_deg = -pick_delta_w_deg

    pick_hover_pose = build_pose_with_w(
        raw_pick_hover_pose,
        chosen_pick_w_deg,
        "pick_hover_pose",
    )
    pick_grasp_pose = build_pose_with_z(
        pick_hover_pose,
        PICK_GRASP_Z_MM,
        "pick_grasp_pose",
    )
    pick_lift_pose = build_pose_with_z(
        pick_hover_pose,
        PICK_LIFT_Z_MM,
        "pick_lift_pose",
    )
    pick_lift_rotate_back_pose = build_pose_with_w(
        pick_lift_pose,
        chosen_pick_w_deg + reverse_delta_w_deg,
        "pick_lift_rotate_back_pose",
    )

    return {
        "target": target_snapshot,
        "raw_pick_hover_pose": raw_pick_hover_pose,
        "raw_pick_w_deg": raw_pick_w_deg,
        "chosen_pick_w_deg": chosen_pick_w_deg,
        "pick_delta_w_deg": pick_delta_w_deg,
        "reverse_delta_w_deg": reverse_delta_w_deg,
        "pick_hover_pose": pick_hover_pose,
        "pick_grasp_pose": pick_grasp_pose,
        "pick_lift_pose": pick_lift_pose,
        "pick_lift_rotate_back_pose": pick_lift_rotate_back_pose,
    }


def acquire_target_context(show_vision_windows: bool, vision_session=None) -> Optional[dict]:
    log("waiting for locked target")
    target = acquire_locked_target(
        timeout_s=NO_TARGET_TIMEOUT_S,
        show_windows=show_vision_windows,
        vision_session=vision_session,
    )
    if target is None:
        log(f"no locked target for {NO_TARGET_TIMEOUT_S:.1f}s, stopping")
        return None

    validated_target = validate_target_snapshot(target)
    context = build_pick_context(validated_target)
    log_target_context(context)
    return context


def execute_pick_hover(indy: IndyDCP3, context: dict, dry_run: bool) -> None:
    log("moving to pick hover")
    movel_abs(
        indy,
        pose_dict_to_list(context["pick_hover_pose"]),
        "pick_hover",
        dry_run,
        holding_object=False,
    )


def execute_descend_only(
    indy: IndyDCP3,
    context: dict,
    dry_run: bool,
    descend_z_mm: float,
    lift_back_to_hover: bool = True,
) -> None:
    descend_pose = build_pose_with_z(
        context["pick_hover_pose"],
        descend_z_mm,
        "stage_descend_pose",
    )

    log(f"descending vertically to Z={descend_pose['Z_mm']:.2f}")
    movel_abs(
        indy,
        pose_dict_to_list(descend_pose),
        "stage_descend",
        dry_run,
        holding_object=False,
    )

    if lift_back_to_hover:
        log("lifting back to hover")
        movel_abs(
            indy,
            pose_dict_to_list(context["pick_hover_pose"]),
            "stage_descend_return_hover",
            dry_run,
            holding_object=False,
        )


def execute_pick_until_lift(
    indy: IndyDCP3,
    endtool_port: str,
    context: dict,
    dry_run: bool,
) -> bool:
    execute_pick_hover(indy, context, dry_run)

    log("moving to grasp")
    movel_abs(
        indy,
        pose_dict_to_list(context["pick_grasp_pose"]),
        "pick_grasp",
        dry_run,
        holding_object=False,
    )

    holding_object = close_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="pick_close_gripper",
        settle_s=POST_CLOSE_SETTLE_S,
        holding_object=False,
    )

    log("lifting")
    movel_abs(
        indy,
        pose_dict_to_list(context["pick_lift_pose"]),
        "pick_lift",
        dry_run,
        holding_object=holding_object,
    )
    return holding_object


def execute_pick_until_lift_and_rotate_back(
    indy: IndyDCP3,
    endtool_port: str,
    context: dict,
    dry_run: bool,
) -> bool:
    execute_pick_hover(indy, context, dry_run)

    log("moving to grasp")
    movel_abs(
        indy,
        pose_dict_to_list(context["pick_grasp_pose"]),
        "pick_grasp",
        dry_run,
        holding_object=False,
    )

    holding_object = close_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="pick_close_gripper",
        settle_s=POST_CLOSE_SETTLE_S,
        holding_object=False,
    )

    log(
        "lifting while rotating back to default W: "
        f"reverse_delta={context['reverse_delta_w_deg']:+.2f} deg "
        f"-> Z={context['pick_lift_rotate_back_pose']['Z_mm']:.2f} "
        f"W={context['pick_lift_rotate_back_pose']['W_deg']:.2f}"
    )
    movel_abs(
        indy,
        pose_dict_to_list(context["pick_lift_rotate_back_pose"]),
        "pick_lift_rotate_back_combined",
        dry_run,
        holding_object=holding_object,
    )
    return holding_object


def execute_place_path_for_bin(
    indy: IndyDCP3,
    endtool_port: str,
    bin_label: str,
    dry_run: bool,
    holding_object: bool,
) -> bool:
    place_transfer_pose = build_place_pose(
        bin_label,
        POST_PLACE_LIFT_Z_MM,
        DEFAULT_W_DEG,
    )
    place_release_pose = build_place_pose(
        bin_label,
        PLACE_RELEASE_Z_MM,
        DEFAULT_W_DEG,
    )
    post_place_lift_pose = build_place_pose(
        bin_label,
        POST_PLACE_LIFT_Z_MM,
        DEFAULT_W_DEG,
    )

    log(f"moving above {bin_label} bin")
    movel_abs(
        indy,
        pose_dict_to_list(place_transfer_pose),
        f"place_transfer_{bin_label}",
        dry_run,
        holding_object=holding_object,
    )

    log("descending to release height")
    movel_abs(
        indy,
        pose_dict_to_list(place_release_pose),
        f"place_release_{bin_label}",
        dry_run,
        holding_object=holding_object,
    )

    open_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label=f"place_open_gripper_{bin_label}",
        settle_s=POST_PLACE_OPEN_SETTLE_S,
        holding_object=holding_object,
    )
    holding_object = False

    log("lifting from bin")
    movel_abs(
        indy,
        pose_dict_to_list(post_place_lift_pose),
        "post_place_lift",
        dry_run,
        holding_object=holding_object,
    )
    return holding_object


def run_startup_sequence(indy: IndyDCP3, endtool_port: str, dry_run: bool) -> None:
    # Full-cycle startup: HOME once, then open the gripper before scanning.
    move_to_home(indy, dry_run)
    open_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="startup_open_gripper",
        settle_s=INITIAL_GRIPPER_SETTLE_S,
    )


# ============================================================
# one pick-and-place cycle
# ============================================================

def run_pick_and_place_cycle(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> bool:
    prepare_scan_for_detection(indy, endtool_port, dry_run)

    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return False

    holding_object = execute_pick_until_lift_and_rotate_back(
        indy,
        endtool_port,
        context,
        dry_run,
    )
    holding_object = execute_place_path_for_bin(
        indy,
        endtool_port,
        context["target"]["label"],
        dry_run,
        holding_object=holding_object,
    )

    log("returning to scan")
    movej_abs(indy, SCAN_J, "scan_pose_return", dry_run, holding_object=holding_object)

    log("cycle complete")
    return True


# ============================================================
# staged test modes
# ============================================================

def run_stage_home(indy: IndyDCP3, dry_run: bool) -> None:
    move_to_home(indy, dry_run)


def run_stage_scan(indy: IndyDCP3, dry_run: bool) -> None:
    move_to_home(indy, dry_run)
    move_to_scan(indy, dry_run)


def run_stage_gripper(indy: IndyDCP3, endtool_port: str, dry_run: bool) -> None:
    move_to_home(indy, dry_run)
    open_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="stage_gripper_open_1",
        settle_s=INITIAL_GRIPPER_SETTLE_S,
    )
    close_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="stage_gripper_close",
        settle_s=POST_CLOSE_SETTLE_S,
    )
    open_gripper_with_wait(
        indy,
        endtool_port,
        dry_run=dry_run,
        label="stage_gripper_open_2",
        settle_s=INITIAL_GRIPPER_SETTLE_S,
    )


def run_stage_vision(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> None:
    prepare_scan_for_detection(indy, endtool_port, dry_run)
    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return

    target = context["target"]
    log("frozen target snapshot:")
    log(f"  label={target['label']}")
    log(f"  grasp_X_m={target['grasp_X_m']:.3f}")
    log(f"  grasp_Y_m={target['grasp_Y_m']:.3f}")
    log(f"  grasp_Z_m={target['grasp_Z_m']:.3f}")
    log(f"  angle_deg={target['angle_deg']:.1f}")
    log(f"  raw_pick_w={context['raw_pick_w_deg']:.2f}")
    log(f"  chosen_pick_w={context['chosen_pick_w_deg']:.2f}")
    log(f"  pick_delta_w_deg={context['pick_delta_w_deg']:+.2f}")


def run_stage_hover(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> None:
    prepare_scan_for_detection(indy, endtool_port, dry_run)
    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return
    execute_pick_hover(indy, context, dry_run)


def run_stage_descend(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    test_z: Optional[float],
    vision_session=None,
) -> None:
    prepare_scan_for_detection(indy, endtool_port, dry_run)
    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return
    execute_pick_hover(indy, context, dry_run)
    descend_z_mm = PICK_GRASP_Z_MM if test_z is None else float(test_z)
    if not math.isfinite(descend_z_mm):
        raise AbortRunError("stage_descend", f"test Z is not finite: {descend_z_mm}")
    log(f"descend test Z={descend_z_mm:.2f}")
    execute_descend_only(
        indy,
        context,
        dry_run,
        descend_z_mm=descend_z_mm,
        lift_back_to_hover=True,
    )


def run_stage_pick(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> None:
    prepare_scan_for_detection(indy, endtool_port, dry_run)
    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return
    execute_pick_until_lift(indy, endtool_port, context, dry_run)


def run_stage_rotate_back(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> None:
    prepare_scan_for_detection(indy, endtool_port, dry_run)
    context = acquire_target_context(show_vision_windows, vision_session=vision_session)
    if context is None:
        return
    execute_pick_until_lift_and_rotate_back(
        indy,
        endtool_port,
        context,
        dry_run,
    )


def run_stage_place_bin(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    stage_name: str,
) -> None:
    move_to_home(indy, dry_run)
    bin_label = PLACE_STAGE_TO_LABEL[stage_name]
    log(f"testing place path only for {bin_label} bin")
    execute_place_path_for_bin(
        indy,
        endtool_port,
        bin_label,
        dry_run,
        holding_object=False,
    )


def run_stage_full_cycle(
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    vision_session=None,
) -> None:
    run_startup_sequence(indy, endtool_port, dry_run)

    cycle_index = 1
    while True:
        log(f"starting cycle {cycle_index}")
        cycle_completed = run_pick_and_place_cycle(
            indy=indy,
            endtool_port=endtool_port,
            dry_run=dry_run,
            show_vision_windows=show_vision_windows,
            vision_session=vision_session,
        )
        if not cycle_completed:
            break
        cycle_index += 1


def run_selected_stage(
    stage: str,
    indy: IndyDCP3,
    endtool_port: str,
    dry_run: bool,
    show_vision_windows: bool,
    test_z: Optional[float],
    vision_session=None,
) -> None:
    if stage == "home":
        run_stage_home(indy, dry_run)
        return
    if stage == "scan":
        run_stage_scan(indy, dry_run)
        return
    if stage == "gripper":
        run_stage_gripper(indy, endtool_port, dry_run)
        return
    if stage == "vision":
        run_stage_vision(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            vision_session=vision_session,
        )
        return
    if stage == "hover":
        run_stage_hover(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            vision_session=vision_session,
        )
        return
    if stage == "descend":
        run_stage_descend(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            test_z,
            vision_session=vision_session,
        )
        return
    if stage == "pick":
        run_stage_pick(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            vision_session=vision_session,
        )
        return
    if stage == "rotate_back":
        run_stage_rotate_back(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            vision_session=vision_session,
        )
        return
    if stage in PLACE_STAGE_TO_LABEL:
        run_stage_place_bin(indy, endtool_port, dry_run, stage)
        return
    if stage == "full_cycle":
        run_stage_full_cycle(
            indy,
            endtool_port,
            dry_run,
            show_vision_windows,
            vision_session=vision_session,
        )
        return
    raise AbortRunError("run_selected_stage", f"unsupported stage: {stage!r}")


def stage_requires_vision(stage: str) -> bool:
    return stage in VISION_STAGES


# ============================================================
# main loop
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP, help="Robot controller IP")
    parser.add_argument("--index", type=int, default=ROBOT_INDEX, help="Robot index")
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Actually move the robot and write endtool IO",
    )
    parser.add_argument(
        "--hide-vision",
        action="store_true",
        help="Run target acquisition without OpenCV windows",
    )
    parser.add_argument(
        "--endtool-port",
        default=ENDTOOL_PORT_OVERRIDE,
        help="Override endtool port letter if auto-detect is not correct",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="full_cycle",
        help="Run one staged test mode or the normal full cycle",
    )
    parser.add_argument(
        "--test-z",
        type=float,
        default=None,
        help="Optional descend test Z override for --stage descend",
    )
    return parser.parse_args()


def main():
    global CURRENT_ROBOT

    args = parse_args()
    dry_run = not args.arm
    show_vision_windows = SHOW_VISION_WINDOWS and (not args.hide_vision)
    vision_session = None

    log("Fixed-Scan Tube Sort Runner V1")
    log(f"mode: {'DRY RUN' if dry_run else 'ARMED'}")
    log(f"Running stage: {args.stage}")
    log(f"home joint pose: {HOME_J}")
    log(f"scan joint pose: {SCAN_J}")
    log(f"scan cartesian pose: {SCAN_T}")
    log("gripper IO rule: endtool DO.0 PNP ON=open, PNP OFF=close")
    if args.test_z is not None:
        log(f"test descend Z override: {args.test_z:.2f}")

    indy = connect_robot(args.ip, args.index)
    CURRENT_ROBOT = indy

    try:
        endtool_port = detect_endtool_port(indy, args.endtool_port)
        log(f"using endtool port: {endtool_port}")

        if stage_requires_vision(args.stage):
            log("opening shared camera session")
            vision_session = create_camera_session()

        run_selected_stage(
            stage=args.stage,
            indy=indy,
            endtool_port=endtool_port,
            dry_run=dry_run,
            show_vision_windows=show_vision_windows,
            test_z=args.test_z,
            vision_session=vision_session,
        )
    finally:
        if vision_session is not None:
            log("closing shared camera session")
            close_camera_session(
                vision_session,
                destroy_windows=show_vision_windows,
            )


if __name__ == "__main__":
    try:
        main()
    except AbortRunError as exc:
        log(f"aborting at {exc.step_label}: {exc}")
        if exc.holding_object:
            log("manual recovery may be needed: the robot may still be holding an object")
        if CURRENT_ROBOT is not None:
            stop_motion_best_effort(CURRENT_ROBOT)
        sys.exit(1)
    except KeyboardInterrupt:
        log("keyboard interrupt, stopping")
        if CURRENT_ROBOT is not None:
            stop_motion_best_effort(CURRENT_ROBOT)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        log(f"unhandled exception: {exc!r}")
        log(traceback.format_exc())
        if CURRENT_ROBOT is not None:
            stop_motion_best_effort(CURRENT_ROBOT)
        sys.exit(1)
