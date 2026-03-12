"""
indy_pick_place_manual.py
Manual-first IndyDCP3 pick-and-place script (Neuromeka Indy7)

IMPORTANT SAFETY NOTES:
- DRY RUN is default. Robot will NOT move unless you pass --arm.
- When --arm is used, the robot can move immediately after a command succeeds.
- This script continuously monitors for collision/violation and issues STOP CAT0
  immediately (emergency stop) if detected.
"""

import argparse
import sys
import time
import traceback

from neuromeka import IndyDCP3


try:
    from neuromeka import (
        StopCategory,      # CAT0, CAT1, CAT2
        OpState,           # IDLE, MOVING, COLLISION, etc.
        BlendingType,      # NONE, OVERRIDE, DUPLICATE
        JointBaseType,     # ABSOLUTE, RELATIVE
        TaskBaseType,      # ABSOLUTE, RELATIVE, TCP
        DigitalState,      # OFF, ON, UNUSED
        EndtoolState       # UNUSED, HIGH_PNP, HIGH_NPN, LOW_NPN, LOW_PNP
    )
except Exception:
    StopCategory = None
    OpState = None
    BlendingType = None
    JointBaseType = None
    TaskBaseType = None
    DigitalState = None
    EndtoolState = None


# Move settings (manual: vel_ratio 0~100, acc_ratio 0~900)
VEL_RATIO = 40
ACC_RATIO = 100

APPROACH_MM = 0.01  # mm (approach distance is small because we use MoveL to move to target position actually)
PICK_RETRACT_MM = 250.0  # mm
PLACE_RETRACT_MM = 313.0  # mm

# Blend radius — manual describes blending/smoothing in millimeters
BLEND_RADIUS_MM = 50.0

# Waypoints
MOVE_TO_OBJ_J = [-5.18, -59.31, -43.73, -0.79, -77.62, 0.03]  # deg

MOVE_TO_PICK_T = [705.85, -254.33, -16.87, -179.22, 0.66, 174.97]  # mm, deg (X Y Z U V W) (XYZ in mm, UVW in deg) )

MOVE_TO_BOX_J = [60.23, -59.37, -42.28, -0.79, -79.00, 0.27]  # deg

PLACE_WP1_T = [465.12, 338.33, 233.08, 0.78, -179.35, 60.11]     # mm, deg
PLACE_WP2_T = [465.12, 338.33, -481.02, 0.78, -179.35, 60.11]    # mm, deg

MOVE_ABOVE_BOX_J = [56.82, -44.30, -106.60, -1.57, -31.29, -1.96]  # deg

# Tool payload (SusGrip)
TOOL_MASS_KG = 1.4
TOOL_COM_MM = [0.0, 0.0, 63.0]


# ============================================================
# Small "boring" helpers (pure Python; robot calls use manual APIs)
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)

def fail(msg: str, exit_code: int = 1) -> None:
    log(f"\n❌ {msg}")
    sys.exit(exit_code)

def now_s() -> float:
    return time.time()

def enum_val(enum_obj, fallback_int: int) -> int:
    """Return enum int value if enum exists; else use fallback_int."""
    try:
        return int(enum_obj)
    except Exception:
        return int(fallback_int)

def opstate_name(op: int) -> str:
    if OpState is None:
        return str(op)
    # Best effort: map known ones (manual lists many states) :contentReference[oaicite:4]{index=4}
    for name in dir(OpState):
        if name.isupper():
            try:
                if getattr(OpState, name) == op:
                    return name
            except Exception:
                pass
    return str(op)


# ============================================================
# Safety: collision/violation monitoring + emergency stop
# ============================================================

def emergency_stop(indy: IndyDCP3, reason: str) -> None:
    """
    Immediate stop (CAT0). Manual: CAT0 cuts power (emergency stop). :contentReference[oaicite:5]{index=5}
    """
    log(f"\n🚨 EMERGENCY STOP: {reason}")
    try:
        if StopCategory is not None:
            indy.stop_motion(StopCategory.CAT0)
        else:
            indy.stop_motion(0)  # CAT0
    except Exception as e:
        log(f"⚠️ stop_motion(CAT0) failed: {repr(e)}")
    # Exit immediately after sending stop
    sys.exit(2)

def check_collision_or_violation(indy: IndyDCP3) -> None:
    """
    Called frequently while moving/sleeping.
    If collision/violation appears, stop immediately.
    """
    # Check op_state
    try:
        rd = indy.get_robot_data()
        op = int(rd.get("op_state", -1))
    except Exception as e:
        emergency_stop(indy, f"Lost robot data (connection issue): {repr(e)}")
        return

    # Manual lists COLLISION=8, VIOLATE=2, VIOLATE_HARD=15, STOP_AND_OFF=9 etc. 
    if OpState is not None:
        # Compare using enum values when available
        if op in [OpState.COLLISION, OpState.VIOLATE, OpState.VIOLATE_HARD, OpState.STOP_AND_OFF]:
            # Pull violation details for logging
            try:
                vd = indy.get_violation_data()
            except Exception:
                vd = {}
            emergency_stop(indy, f"op_state={opstate_name(op)} violation={vd}")
    else:
        # Fallback numeric checks
        if op in [8, 2, 15, 9]:
            try:
                vd = indy.get_violation_data()
            except Exception:
                vd = {}
            emergency_stop(indy, f"op_state={op} violation={vd}")

    # Also check violation_data directly (even if op_state not yet updated)
    try:
        vd = indy.get_violation_data()
        vcode = str(vd.get("violation_code", "0"))
        vstr = str(vd.get("violation_str", ""))
        if vcode not in ["0", "", "None"] and vstr.strip() not in ["", "None"]:
            emergency_stop(indy, f"violation_code={vcode} violation_str={vstr}")
    except Exception:
        # If violation read fails, don't stop purely due to that
        pass


# ============================================================
# Waiting / sleeping while watching collision
# (Uses get_motion_data() from manual) :contentReference[oaicite:7]{index=7}
# ============================================================

def safe_sleep(indy: IndyDCP3, seconds: float, tick_s: float = 0.02) -> None:
    """
    Sleep in small increments so we can stop immediately if collision occurs.
    """
    t0 = now_s()
    while True:
        check_collision_or_violation(indy)
        if now_s() - t0 >= seconds:
            return
        time.sleep(tick_s)

def wait_motion_done(indy: IndyDCP3, timeout_s: float, label: str) -> None:
    """
    Poll get_motion_data until motion is finished.
    Manual fields: is_in_motion, is_target_reached, has_motion, traj_state, etc. :contentReference[oaicite:8]{index=8}
    Also checks collision/violation continuously.
    """
    t0 = now_s()
    started = False

    while True:
        check_collision_or_violation(indy)

        try:
            md = indy.get_motion_data()
        except Exception as e:
            emergency_stop(indy, f"{label}: get_motion_data failed: {repr(e)}")
            return

        is_in_motion = bool(md.get("is_in_motion", False))
        is_target_reached = bool(md.get("is_target_reached", False))
        has_motion = bool(md.get("has_motion", False))
        # remain_distance is in mm (manual) :contentReference[oaicite:9]{index=9}
        remain_dist = md.get("remain_distance", None)

        if is_in_motion or has_motion:
            started = True

        # Consider done when target reached AND not in motion.
        if started and (not is_in_motion) and is_target_reached:
            return

        if now_s() - t0 > timeout_s:
            emergency_stop(indy, f"{label}: Timeout waiting motion done. motion_data={md}")
            return

        time.sleep(0.02)


# ============================================================
# Gripper control using ONLY manual IO APIs
# ============================================================

def detect_endtool_port(indy: IndyDCP3) -> str:
    """
    Manual: get_endtool_do() returns {'signals': [{'port':'A','states':[...]} ... ]} 
    Choose a port automatically.
    """
    data = indy.get_endtool_do()
    signals = data.get("signals", [])
    if not signals:
        return ""
    # If only one port exists (RevC: 'C'), use it; else use 'A' if exists; else first.
    ports = [str(s.get("port", "")) for s in signals]
    if "C" in ports:
        return "C"
    if "A" in ports:
        return "A"
    return ports[0] if ports else ""

def set_gripper_open_endtool(indy: IndyDCP3, port: str, dry_run: bool) -> None:
    """
    OPEN: PNP HIGH.
    Manual: EndtoolState.HIGH_PNP used in examples. :contentReference[oaicite:11]{index=11}
    """
    if EndtoolState is None:
        fail("EndtoolState enum not available in your neuromeka package. Use --gripper-io do instead.")

    signal = [{'port': port, 'states': [EndtoolState.HIGH_PNP]}]
    if dry_run:
        log(f"[DRY RUN] set_endtool_do({signal})  # OPEN")
        return
    indy.set_endtool_do(signal)

def set_gripper_close_endtool(indy: IndyDCP3, port: str, dry_run: bool) -> None:
    """
    CLOSE: output disabled (UNUSED) to get 0V/floating.
    Manual: EndtoolState.UNUSED exists; examples show states list integers. :contentReference[oaicite:12]{index=12}
    """
    if EndtoolState is None:
        fail("EndtoolState enum not available in your neuromeka package. Use --gripper-io do instead.")

    signal = [{'port': port, 'states': [EndtoolState.UNUSED]}]
    if dry_run:
        log(f"[DRY RUN] set_endtool_do({signal})  # CLOSE")
        return
    indy.set_endtool_do(signal)

def set_gripper_open_do(indy: IndyDCP3, address: int, dry_run: bool) -> None:
    """
    OPEN: DO ON. Manual shows DigitalState.ON and set_do(signal_list). :contentReference[oaicite:13]{index=13}
    """
    if DigitalState is None:
        # fallback: ON=1
        signal = [{'address': address, 'state': 1}]
    else:
        signal = [{'address': address, 'state': DigitalState.ON}]
    if dry_run:
        log(f"[DRY RUN] set_do({signal})  # OPEN")
        return
    indy.set_do(signal)

def set_gripper_close_do(indy: IndyDCP3, address: int, dry_run: bool) -> None:
    """
    CLOSE: DO OFF.
    """
    if DigitalState is None:
        signal = [{'address': address, 'state': 0}]
    else:
        signal = [{'address': address, 'state': DigitalState.OFF}]
    if dry_run:
        log(f"[DRY RUN] set_do({signal})  # CLOSE")
        return
    indy.set_do(signal)


# ============================================================
# Tool payload setting (manual get_tool_property / set_tool_property) :contentReference[oaicite:14]{index=14}
# ============================================================

def maybe_apply_tool_payload(indy: IndyDCP3, apply: bool, dry_run: bool) -> None:
    tool = indy.get_tool_property()
    log(f"Tool property (current): {tool}")

    if not apply:
        log("Tool payload not changed. (Use --set-tool to apply your 1.4kg CoM settings.)")
        return

    inertia = tool.get("inertia", None)
    if inertia is None:
        fail("Tool property missing 'inertia' field; cannot safely set tool payload.")

    if dry_run:
        log(f"[DRY RUN] set_tool_property(mass={TOOL_MASS_KG}, center_of_mass={TOOL_COM_MM}, inertia=KEEP_EXISTING)")
        return

    indy.set_tool_property(mass=TOOL_MASS_KG, center_of_mass=TOOL_COM_MM, inertia=inertia)
    log(f"Tool property (updated): {indy.get_tool_property()}")


# ============================================================
# Motion commands using ONLY manual move_home/movej/movel/stop_motion APIs :contentReference[oaicite:15]{index=15}
# ============================================================

def move_home(indy: IndyDCP3, dry_run: bool) -> None:
    log("→ move_home()")
    if dry_run:
        log("[DRY RUN] move_home()")
        return
    indy.move_home()
    wait_motion_done(indy, timeout_s=180.0, label="move_home")

def movej_abs(indy: IndyDCP3, jtarget_deg, label: str, dry_run: bool,
              blend: bool = False, queue_duplicate: bool = False) -> None:
    """
    movej(jtarget, blending_type, base_type, blending_radius, vel_ratio, acc_ratio, ...)
    Manual: base_type JointBaseType.ABSOLUTE/RELATIVE; blending_type NONE/OVERRIDE/DUPLICATE. :contentReference[oaicite:16]{index=16}
    """
    log(f"→ movej [{label}] jtarget={jtarget_deg} vel_ratio={VEL_RATIO} acc_ratio={ACC_RATIO} blend={blend}")

    if JointBaseType is not None:
        base_type = JointBaseType.ABSOLUTE
    else:
        base_type = 0  # ABSOLUTE

    if BlendingType is not None:
        if blend:
            blending_type = BlendingType.DUPLICATE if queue_duplicate else BlendingType.OVERRIDE
        else:
            blending_type = BlendingType.NONE
    else:
        blending_type = 2 if blend else 0

    kwargs = dict(
        jtarget=jtarget_deg,
        base_type=base_type,
        vel_ratio=VEL_RATIO,
        acc_ratio=ACC_RATIO,
    )

    if blend:
        kwargs["blending_type"] = blending_type
        kwargs["blending_radius"] = BLEND_RADIUS_MM
    else:
        # be explicit anyway
        kwargs["blending_type"] = blending_type
        kwargs["blending_radius"] = 0.0

    if dry_run:
        log(f"[DRY RUN] movej(**{kwargs})")
        return

    indy.movej(**kwargs)

def movel_abs(indy: IndyDCP3, ttarget_mm_deg, label: str, dry_run: bool,
              blend: bool = False, queue_duplicate: bool = False,
              bypass_singular: bool = False) -> None:
    """
    movel(ttarget, blending_type, base_type, blending_radius, vel_ratio, acc_ratio, ..., bypass_singular)
    Manual: TaskBaseType.ABSOLUTE/RELATIVE/TCP. 
    """
    log(f"→ movel [{label}] ttarget={ttarget_mm_deg} vel_ratio={VEL_RATIO} acc_ratio={ACC_RATIO} blend={blend}")

    if TaskBaseType is not None:
        base_type = TaskBaseType.ABSOLUTE
    else:
        base_type = 0  # ABSOLUTE

    if BlendingType is not None:
        if blend:
            blending_type = BlendingType.DUPLICATE if queue_duplicate else BlendingType.OVERRIDE
        else:
            blending_type = BlendingType.NONE
    else:
        blending_type = 2 if blend else 0

    kwargs = dict(
        ttarget=ttarget_mm_deg,
        base_type=base_type,
        vel_ratio=VEL_RATIO,
        acc_ratio=ACC_RATIO,
        bypass_singular=bypass_singular,
    )

    if blend:
        kwargs["blending_type"] = blending_type
        kwargs["blending_radius"] = BLEND_RADIUS_MM
    else:
        kwargs["blending_type"] = blending_type
        kwargs["blending_radius"] = 0.0

    if dry_run:
        log(f"[DRY RUN] movel(**{kwargs})")
        return

    indy.movel(**kwargs)


# ============================================================
# "Pick/Place application" implemented as:
# ApproachPose -> TargetPose -> RetractPose
# ============================================================

def make_pose_with_z_offset(target_t, dz_mm: float):
    p = list(target_t)
    p[2] = float(p[2]) + float(dz_mm)
    return p


def pick_place_application(
    indy: IndyDCP3,
    target_t,
    approach_mm: float,
    retract_mm: float,
    label: str,
    dry_run: bool,
) -> None:
    if approach_mm < 0.1:
        log(
            f"⚠️ WARNING: approach_mm={approach_mm}mm is extremely small (almost no approach)."
        )

    approach_t = make_pose_with_z_offset(target_t, approach_mm)
    retract_t = make_pose_with_z_offset(target_t, retract_mm)

    movel_abs(indy, approach_t, f"{label}_Approach", dry_run, blend=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=120.0, label=f"{label}_Approach")

    movel_abs(indy, list(target_t), f"{label}_Target", dry_run, blend=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=120.0, label=f"{label}_Target")

    movel_abs(indy, retract_t, f"{label}_Retract", dry_run, blend=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=120.0, label=f"{label}_Retract")


# ============================================================
# Preflight checks (edge cases)
# ============================================================

def preflight(indy: IndyDCP3, dry_run: bool, enable_servo: bool,
              coll_level, coll_policy) -> None:
    """
    - Verify connectivity
    - Check op_state is safe
    - Optionally enable servo
    - Optionally set collision sensitivity/policy
    """
    # Control info (manual: get_control_info) :contentReference[oaicite:18]{index=18}
    try:
        ci = indy.get_control_info()
        log(f"Control info: {ci}")
    except Exception as e:
        fail(f"get_control_info failed (check IP/ports/version): {repr(e)}")

    rd = indy.get_robot_data()
    op = int(rd.get("op_state", -1))
    sim_mode = bool(rd.get("sim_mode", False))
    log(f"Robot op_state={op} ({opstate_name(op)}) sim_mode={sim_mode}")

    if sim_mode:
        log("⚠️ WARNING: sim_mode=True (controller in simulation mode). Real robot might not move.")

    # If currently in collision/violate, stop and exit
    if OpState is not None and op in [OpState.COLLISION, OpState.VIOLATE, OpState.VIOLATE_HARD]:
        fail(f"Robot is in unsafe state at start: {opstate_name(op)}. Recover in CONTY, then retry.")
    if OpState is None and op in [8, 2, 15]:
        fail(f"Robot is in unsafe state at start: {op}. Recover in CONTY, then retry.")

    # Servo status
    try:
        sd = indy.get_servo_data()
        servos = sd.get("servo_actives", [])
        log(f"Servo actives: {servos}")
        if enable_servo and (not dry_run):
            if any(s is False for s in servos):
                log("Enabling all servos (set_servo_all(True))...")
                indy.set_servo_all(enable=True)
                # Re-check
                sd2 = indy.get_servo_data()
                log(f"Servo actives (after enable): {sd2.get('servo_actives', [])}")
    except Exception as e:
        log(f"⚠️ get_servo_data/set_servo_all not available or failed: {repr(e)}")

    # Collision settings (optional)
    # Manual: get_coll_sens_level / set_coll_sens_level(level), get_coll_policy / set_coll_policy(policy, sleep_time, gravity_time) 
    try:
        cur_level = indy.get_coll_sens_level()
        log(f"Collision sensitivity (current): {cur_level}")
    except Exception as e:
        log(f"⚠️ get_coll_sens_level failed: {repr(e)}")

    if coll_level is not None:
        if dry_run:
            log(f"[DRY RUN] set_coll_sens_level(level={coll_level})")
        else:
            log(f"Setting collision sensitivity level to {coll_level} ...")
            indy.set_coll_sens_level(level=coll_level)
            log(f"Collision sensitivity (after): {indy.get_coll_sens_level()}")

    try:
        cur_pol = indy.get_coll_policy()
        log(f"Collision policy (current): {cur_pol}")
    except Exception as e:
        log(f"⚠️ get_coll_policy failed: {repr(e)}")

    if coll_policy is not None:
        policy, sleep_time, gravity_time = coll_policy
        if dry_run:
            log(f"[DRY RUN] set_coll_policy(policy={policy}, sleep_time={sleep_time}, gravity_time={gravity_time})")
        else:
            log(f"Setting collision policy to policy={policy}, sleep_time={sleep_time}, gravity_time={gravity_time} ...")
            indy.set_coll_policy(policy=policy, sleep_time=sleep_time, gravity_time=gravity_time)
            log(f"Collision policy (after): {indy.get_coll_policy()}")


# ============================================================
# Main pick-and-place sequence (exact order)
# ============================================================

def run_sequence(indy: IndyDCP3, dry_run: bool,
                 gripper_io: str,
                 do_addr: int,
                 endtool_port: str) -> None:
    """
    Order:
    1. Move Home
    2. Set signal ON (OPEN)
    3. Sleep 1s
    4. MoveJ MoveToObj
    5. MoveL MoveToPick
    6. Set signal OFF (CLOSE)
    7. Sleep 2s
    8. Pick application (approach/target/retract)
    9. MoveJ MoveToBox (blend)
    10. MoveL Place (WP1 -> WP2)
    11. Set signal ON (OPEN)
    12. Sleep 1s
    13. Place application (approach/target/retract)
    14. MoveJ MoveAboveBox (blend)
    15. Move Home
    """

    # helper: gripper open/close using chosen IO
    def gripper_open():
        if gripper_io == "endtool":
            set_gripper_open_endtool(indy, endtool_port, dry_run)
        else:
            set_gripper_open_do(indy, do_addr, dry_run)

    def gripper_close():
        if gripper_io == "endtool":
            set_gripper_close_endtool(indy, endtool_port, dry_run)
        else:
            set_gripper_close_do(indy, do_addr, dry_run)

    # 1 Move Home
    log("\n=== 1) Move Home ===")
    move_home(indy, dry_run)
    if not dry_run:
        wait_motion_done(indy, timeout_s=180.0, label="Home")

    # 2 Set signal ON (OPEN)
    log("\n=== 2) Gripper OPEN (signal ON) ===")
    gripper_open()

    # 3 Sleep 1s (safe sleep checks collision)
    log("\n=== 3) Sleep 1 second ===")
    if dry_run:
        log("[DRY RUN] sleep(1)")
    else:
        safe_sleep(indy, 1.0)

    # 4 MoveJ MoveToObj (no blend)
    log("\n=== 4) MoveJ MoveToObj ===")
    movej_abs(indy, MOVE_TO_OBJ_J, "MoveToObj", dry_run, blend=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=120.0, label="MoveToObj")

    # 5 MoveL MoveToPick (no blend)
    log("\n=== 5) MoveL MoveToPick ===")
    movel_abs(indy, MOVE_TO_PICK_T, "MoveToPick", dry_run, blend=False, bypass_singular=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=120.0, label="MoveToPick")

    # 6 Set signal OFF (CLOSE)
    log("\n=== 6) Gripper CLOSE (signal OFF) ===")
    gripper_close()

    # 7 Sleep 2s
    log("\n=== 7) Sleep 2 seconds ===")
    if dry_run:
        log("[DRY RUN] sleep(2)")
    else:
        safe_sleep(indy, 2.5)

    # 8 Pick application (approach/target/retract)
    log("\n=== 8) Pick application (Approach/Target/Retract) ===")
    pick_place_application(
        indy,
        target_t=MOVE_TO_PICK_T,
        approach_mm=APPROACH_MM,
        retract_mm=PICK_RETRACT_MM,
        label="PickApp",
        dry_run=dry_run
    )

    # 9 MoveJ MoveToBox (blend)
    # To make blending meaningful, we queue the next motion immediately.
    log("\n=== 9) MoveJ MoveToBox (BLEND) ===")
    movej_abs(indy, MOVE_TO_BOX_J, "MoveToBox", dry_run, blend=True, queue_duplicate=True)

    # 10 MoveL Place WP1 -> WP2 (no blend)
    # Immediately queue WP1 after blended MoveJ.
    log("\n=== 10) MoveL Place (WP1 -> WP2) ===")
    movel_abs(indy, PLACE_WP1_T, "Place_WP1", dry_run, blend=False, bypass_singular=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=180.0, label="Place_WP1")

    movel_abs(indy, PLACE_WP2_T, "Place_WP2", dry_run, blend=False, bypass_singular=False)
    if not dry_run:
        wait_motion_done(indy, timeout_s=180.0, label="Place_WP2")

    # 11 Set signal ON (OPEN)
    log("\n=== 11) Gripper OPEN (signal ON) ===")
    gripper_open()

    # 12 Sleep 1s
    log("\n=== 12) Sleep 1 second ===")
    if dry_run:
        log("[DRY RUN] sleep(1)")
    else:
        safe_sleep(indy, 1.0)

    # 13 Place application (approach/target/retract)
    log("\n=== 13) Place application (Approach/Target/Retract) ===")
    pick_place_application(
        indy,
        target_t=PLACE_WP2_T,
        approach_mm=APPROACH_MM,
        retract_mm=PLACE_RETRACT_MM,
        label="PlaceApp",
        dry_run=dry_run
    )

    # 14 MoveJ MoveAboveBox (blend)
    log("\n=== 14) MoveJ MoveAboveBox (BLEND) ===")
    movej_abs(indy, MOVE_ABOVE_BOX_J, "MoveAboveBox", dry_run, blend=True, queue_duplicate=True)

    # 15 Move Home (queued quickly after blend)
    log("\n=== 15) Move Home ===")
    # We can queue home immediately; but move_home has no blending_type.
    # So we do: wait a tiny moment so queue clears, then move_home.
    if not dry_run:
        # Give controller a moment to start MoveAboveBox before calling move_home
        safe_sleep(indy, 0.2)
    move_home(indy, dry_run)
    if not dry_run:
        wait_motion_done(indy, timeout_s=180.0, label="FinalHome")

    log("\n✅ SEQUENCE COMPLETE")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="192.168.0.157", help="Robot controller IP")
    p.add_argument("--index", type=int, default=0, help="Robot index (default 0)")

    # Safety arming
    p.add_argument("--arm", action="store_true", help="Actually move the robot / write IO (default DRY RUN)")

    # Optional tool payload application
    p.add_argument("--set-tool", action="store_true", help="Apply tool mass/CoM (keeps inertia)")

    # Optional servo enable
    p.add_argument("--enable-servo", action="store_true", help="Try to enable all servos if not active")

    # Collision settings
    p.add_argument("--coll-level", type=int, default=None,
                   help="Optionally set collision sensitivity level (int). If omitted, keep controller setting.")
    # Policy tuple: policy sleep_time gravity_time
    p.add_argument("--coll-policy", nargs=3, type=float, default=None,
                   help="Optionally set collision policy: policy sleep_time gravity_time (e.g., --coll-policy 1 0.1 0.1)")

    # Gripper IO selection
    p.add_argument("--gripper-io", choices=["auto", "endtool", "do"], default="auto",
                   help="How to control gripper: endtool (PNP), do (controller DO), or auto (try endtool then fallback do).")
    p.add_argument("--do-addr", type=int, default=0, help="Controller DO address for gripper (if using --gripper-io do)")
    p.add_argument("--endtool-port", default=None,
                   help="Endtool port letter (A/B/C). If omitted, script auto-detects from get_endtool_do().")

    return p.parse_args()


def main():
    args = parse_args()

    dry_run = not args.arm

    log("========================================================")
    log("IndyDCP3 Manual-First Pick-and-Place")
    log("========================================================")
    log(f"MODE: {'DRY RUN (no motion, no IO writes)' if dry_run else 'ARMED (ROBOT WILL MOVE)'}")
    log(f"IP={args.ip}  index={args.index}")
    log(f"vel_ratio={VEL_RATIO} acc_ratio={ACC_RATIO}")
    log(f"approach_mm={APPROACH_MM} retract_pick_mm={PICK_RETRACT_MM} retract_place_mm={PLACE_RETRACT_MM}")
    log(f"blend_radius_mm={BLEND_RADIUS_MM}")
    log("SAFETY: Keep area clear. Be ready to hit E-Stop.")

    # Parse collision policy triple
    coll_policy = None
    if args.coll_policy is not None:
        # policy may be float but manual shows policy is int-like
        policy = int(args.coll_policy[0])
        sleep_time = float(args.coll_policy[1])
        gravity_time = float(args.coll_policy[2])
        coll_policy = (policy, sleep_time, gravity_time)

    try:
        indy = IndyDCP3(robot_ip=args.ip, index=args.index)
    except Exception as e:
        fail(f"Failed to create IndyDCP3 client: {repr(e)}")

    # Preflight
    preflight(
        indy,
        dry_run=dry_run,
        enable_servo=args.enable_servo,
        coll_level=args.coll_level,
        coll_policy=coll_policy
    )

    # Tool payload (optional)
    maybe_apply_tool_payload(indy, apply=args.set_tool, dry_run=dry_run)

    # Decide gripper IO mode
    gripper_io = args.gripper_io
    endtool_port = args.endtool_port

    if gripper_io == "auto":
        # Try endtool first (manual: some robots may not support endtool) :contentReference[oaicite:20]{index=20}
        try:
            data = indy.get_endtool_do()
            sigs = data.get("signals", [])
            if sigs:
                gripper_io = "endtool"
            else:
                gripper_io = "do"
        except Exception:
            gripper_io = "do"

    if gripper_io == "endtool":
        # Pick a port
        if endtool_port is None:
            try:
                endtool_port = detect_endtool_port(indy)
            except Exception as e:
                fail(f"get_endtool_do failed; cannot use endtool IO: {repr(e)}")

        if not endtool_port:
            fail("Endtool IO selected but no endtool port detected. Use --gripper-io do instead.")

        log(f"Gripper IO = ENDTOOL port '{endtool_port}' (OPEN=HIGH_PNP, CLOSE=UNUSED)")
        # Print available endtool signals for clarity
        try:
            log(f"Endtool DO snapshot: {indy.get_endtool_do()}")
        except Exception:
            pass

    else:
        log(f"Gripper IO = CONTROLLER DO address {args.do_addr} (OPEN=ON, CLOSE=OFF)")
        # Print available do signals for clarity
        try:
            log(f"Controller DO snapshot: {indy.get_do()}")
        except Exception:
            pass

    # Final safety check before starting motion
    if not dry_run:
        rd = indy.get_robot_data()
        op = int(rd.get("op_state", -1))
        log(f"Before motion: op_state={op} ({opstate_name(op)})")
        # If not IDLE, stop
        if OpState is not None and op not in [OpState.IDLE]:
            fail(f"Robot not IDLE before start (op_state={opstate_name(op)}). Put robot in IDLE then retry.")
        if OpState is None and op != 5:
            fail(f"Robot not IDLE before start (op_state={op}). Put robot in IDLE then retry.")

    # Run sequence
    run_sequence(
        indy,
        dry_run=dry_run,
        gripper_io=gripper_io,
        do_addr=args.do_addr,
        endtool_port=endtool_port
    )

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        # Ctrl+C safety stop
        log("\n🛑 KeyboardInterrupt detected. Stopping robot safely...")
        try:
            # IMPORTANT: this only works if indy is accessible here.
            # If indy is created inside main(), see Fix #2 below.
            if StopCategory is not None:
                indy.stop_motion(StopCategory.CAT0)
            else:
                indy.stop_motion(0)  # CAT0
        except Exception as e:
            log(f"⚠️ stop_motion failed: {repr(e)}")
        sys.exit(130)

    except SystemExit:
        raise

    except Exception as e:
        log("\n❌ Unhandled exception:")
        log(repr(e))
        log(traceback.format_exc())
        try:
            if StopCategory is not None:
                indy.stop_motion(StopCategory.CAT0)
            else:
                indy.stop_motion(0)
        except Exception:
            pass
        sys.exit(1)
