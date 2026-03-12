import time
import logging

from scripts.gripper.susgrip_client import SusGripClient
from neuromeka import IndyDCP3  # same package you used for get_do()


# ===== Indy7 settings =====
ROBOT_IP = "192.168.1.75"

# ===== SusGrip settings =====
GRIPPER_COM = "COM4"
GRIPPER_ID = 1
OPEN_MAX_MM = 120
CLOSE_MIN_MM = 10

# ===== Behavior =====
POLL_HZ = 10  # how often we check DO0
DEBOUNCE_S = 0.15  # DO0 must stay stable for this long to be trusted
SYNC_ON_START = True  # if True, match gripper to current DO0 immediately 
STARTUP_MATCH_TOL_MM = 2  # startup state is considered matched within this tolerance
SYNC_COUNTDOWN_S = 10  # if SYNC_ON_START, how long to wait before syncing (for safety, to let user back away)
SYNC_STATUS_POLL_S = 0.2  # in sync armed window, poll status at most this often
ROBOT_LOST_S = 3.0 # if we can't read DO0 for this long, assume robot is lost and stop for safety
READ_BACKOFF_S = 0.2 # if we fail to read DO0, wait this long before retrying (to avoid spamming errors if robot is unplugged)
CMD_COOLDOWN_S = 0.4 # after sending a command to the gripper, ignore DO0 changes for this long to avoid rapid toggling if something is unstable
GRIPPER_SPEED_MM_S = 100
GRIPPER_FORCE_PCT = 10
DRY_RUN = False  # if True, don't actually send commands to the gripper (for testing the DO0 reading and debouncing logic)
LOG_LEVEL = "INFO"  # "INFO" or "DEBUG"

logger = logging.getLogger("do0_to_susgrip")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(levelname)s] %(message)s",
)
# Keep our logs controlled; silence noisy library debug logs
logging.getLogger("pymodbus").setLevel(logging.WARNING)

def read_do0(robot) -> int:
    """
    Read DO0 robustly.
    Your neuromeka get_do() returns: {'signals': [{'address':0,'state':0}, ...]}
    """
    data = robot.get_do()

    # Case 1: {'signals': [ {address,state}, ... ]}
    if (
        isinstance(data, dict)
        and "signals" in data
        and isinstance(data["signals"], list)
    ):
        for item in data["signals"]:
            if isinstance(item, dict) and item.get("address") == 0:
                return int(item.get("state", 0))

    # Case 2: list of dicts like [{'address':0,'state':1}, ...]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("address") == 0:
                return int(item.get("state", 0))

    # Case 3: other dict shapes (keep a few guesses)
    if isinstance(data, dict):
        for key in ("do", "dos", "data", "list"):
            v = data.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and item.get("address") == 0:
                        return int(item.get("state", 0))

    raise RuntimeError(f"Cannot parse get_do() output: {data!r}")


def main():
    print("=== DO0 -> SusGrip bridge ===")
    print("Safety: keep fingers away; robot stays still; close SusGrip GUI.\n")

    # 1) Connect to robot
    robot = IndyDCP3(ROBOT_IP)
    print(f"Connecting to Indy7 at {ROBOT_IP} ...")
    # Most setups connect lazily; your get_do() call will validate connection.

    # 2) Connect to gripper
    g = SusGripClient(
        port=GRIPPER_COM,
        slave_id=GRIPPER_ID,
        open_max_mm=OPEN_MAX_MM,
        close_min_mm=CLOSE_MIN_MM,
    )

    print(f"Connecting to SusGrip on {GRIPPER_COM} (id={GRIPPER_ID}) ...")
    if not g.connect():
        print("❌ Failed to connect to SusGrip. (Wrong COM? GUI still open?)")
        return

    def safe_emcy_stop():
        try:
            g.emcy_stop()
        except Exception:
            pass

    def print_timeout_guidance(e):
        logger.error("SusGrip TIMEOUT.")
        print("   -> The driver already tried to send EMCY STOP.")
        print("   -> For safety: keep hands away. If anything is still moving, cut 24V power.")
        print(f"   Details: {e}")
        print("   -> Bridge exiting for safety.")

    try:
        # Make gripper safe
        g.emcy_release()
        g.set_speed_force(speed_mm_s=GRIPPER_SPEED_MM_S, force_pct=GRIPPER_FORCE_PCT)

        print("Reading initial DO0 ...")
        t0 = time.time()
        while True:
            try:
                do0 = read_do0(robot)
                break
            except Exception as e:
                if time.time() - t0 > ROBOT_LOST_S:
                    print(f"❌ Cannot read DO0 at startup for > {ROBOT_LOST_S}s.")
                    print(f"   Details: {e}")
                    return
                time.sleep(READ_BACKOFF_S)
        try:
            st = g.get_status()
        except Exception as e:
            logger.error("SusGrip status read failed at startup")
            print("❌ Failed to read SusGrip status at startup.")
            print(f"   Details: {e}")
            print("   -> For safety: if anything is still moving, cut 24V power.")
            print("   -> Bridge exiting for safety.")
            safe_emcy_stop()
            return

        print(
            f"Initial: DO0={do0} | POS={st.pos_mm}mm | ERR={st.error} | EMCY=0x{st.emcy:04X} "
            f"(SYNC_ON_START={SYNC_ON_START})"
        )

        expected_mm = CLOSE_MIN_MM if do0 == 1 else OPEN_MAX_MM
        expected_state = "CLOSE" if do0 == 1 else "OPEN"
        is_matched = abs(st.pos_mm - expected_mm) <= STARTUP_MATCH_TOL_MM

        if is_matched:
            print(f"✅ Startup matched: DO0={do0} already corresponds to {expected_state}.")
            print("   -> No sync needed. Toggle DO0 to command open/close.")

        if not is_matched and not SYNC_ON_START:
            print(
                f"⚠️ MISMATCH: DO0={do0} means {expected_state}, but POS={st.pos_mm}mm "
                f"(expected around {expected_mm}mm ±{STARTUP_MATCH_TOL_MM}mm)."
            )
            print(
                "   -> Toggle DO0 OFF then ON, or enable SYNC_ON_START to make the gripper match the current DO0 automatically (will move at startup)."
                if do0 == 1 else
                "   -> Toggle DO0 ON then OFF, or enable SYNC_ON_START to make the gripper match the current DO0 automatically (will move at startup)."
            )

        if SYNC_ON_START and not is_matched:
            print(f"\n⚠️ Startup mismatch: DO0 wants {expected_state}, but gripper POS={st.pos_mm}mm.")
            print(
                f"Auto-sync is ARMED for {SYNC_COUNTDOWN_S}s.\n"
                f"   -> If you toggle DO0 to match the current gripper state, auto-sync will cancel.\n"
                f"   -> Otherwise, it will move to match DO0 after the timer."
            )

            deadline = time.time() + SYNC_COUNTDOWN_S
            last_tick = None
            last_status_poll_t = 0.0
            st_now = st

            while time.time() < deadline:
                # show a clean 1-second countdown (no spam)
                remain_s = int(deadline - time.time())
                if remain_s != last_tick:
                    print(f"  syncing in {remain_s}s...")
                    last_tick = remain_s

                # keep reading latest DO0 during the countdown
                try:
                    do0_now = read_do0(robot)
                except Exception:
                    time.sleep(READ_BACKOFF_S)
                    continue

                # check whether current DO0 matches current gripper position
                now_sync = time.time()
                if now_sync - last_status_poll_t >= SYNC_STATUS_POLL_S:
                    try:
                        st_now = g.get_status()
                        last_status_poll_t = now_sync
                    except Exception as e:
                        logger.error("SusGrip status read failed during sync arm window")
                        print("❌ Failed to read SusGrip status during startup sync window.")
                        print(f"   Details: {e}")
                        print("   -> For safety: if anything is still moving, cut 24V power.")
                        print("   -> Bridge exiting for safety.")
                        safe_emcy_stop()
                        return
                expected_mm_now = CLOSE_MIN_MM if do0_now == 1 else OPEN_MAX_MM

                if abs(st_now.pos_mm - expected_mm_now) <= STARTUP_MATCH_TOL_MM:
                    state_now = "CLOSE" if do0_now == 1 else "OPEN"
                    print(f"✅ Now matched (DO0={do0_now} -> {state_now}, POS={st_now.pos_mm}mm).")
                    print("   -> Auto-sync cancelled. You can start toggling DO0 now.")
                    break

                time.sleep(0.05)
            else:
                # Timer expired and still mismatch: sync using the latest DO0
                try:
                    do0_now = read_do0(robot)
                except Exception as e:
                    print("❌ Cannot read DO0 to execute startup sync.")
                    print(f"   Details: {e}")
                    print("   -> Bridge exiting for safety.")
                    safe_emcy_stop()
                    return
                
                action = "CLOSE" if do0_now == 1 else "OPEN"
                print(f"⏱ Auto-sync executing now: DO0={do0_now} -> {action}")

                if DRY_RUN:
                    print(f"🟡 DRY_RUN: would {action} (no gripper command sent)")
                else:
                    if do0_now == 1:
                        try:
                            g.close(wait=True)
                        except TimeoutError as e:
                            print_timeout_guidance(e)
                            return
                        except Exception as e:
                            logger.error("SusGrip command failed")
                            print("❌ SusGrip command failed.")
                            print(f"   Details: {e}")
                            print("   -> For safety: if anything is still moving, cut 24V power.")
                            print("   -> Bridge exiting for safety.")
                            safe_emcy_stop()
                            return
                    else:
                        try:
                            g.open(wait=True)
                        except TimeoutError as e:
                            print_timeout_guidance(e)
                            return
                        except Exception as e:
                            logger.error("SusGrip command failed")
                            print("❌ SusGrip command failed.")
                            print(f"   Details: {e}")
                            print("   -> For safety: if anything is still moving, cut 24V power.")
                            print("   -> Bridge exiting for safety.")
                            safe_emcy_stop()
                            return

                try:
                    st2 = g.get_status()
                except Exception as e:
                    logger.error("SusGrip status read failed after startup sync")
                    print("❌ Failed to read SusGrip status after startup sync.")
                    print(f"   Details: {e}")
                    print("   -> For safety: if anything is still moving, cut 24V power.")
                    print("   -> Bridge exiting for safety.")
                    safe_emcy_stop()
                    return
                print(f"After sync: POS={st2.pos_mm}mm ERR={st2.error} EMCY=0x{st2.emcy:04X}")

        # ----------------------------
        # Debounce state (NEW)
        # ----------------------------
        raw_prev = do0  # last raw sample
        raw_change_t = time.time()  # when raw last changed
        debounced = do0  # trusted, stable DO0
        # ----------------------------

        dt = 1.0 / POLL_HZ

        print("\nNow toggle DO0 in CONTY. (0→1=close, 1→0=open)\n")

        last_robot_ok_t = time.time() # to detect if robot stops responding (e.g. unplugged)
        last_cmd_t = 0.0

        while True:
            try:
                raw = read_do0(robot)
                last_robot_ok_t = time.time()
            except Exception as e:
                # If we temporarily can't read DO0, retry a bit.
                if time.time() - last_robot_ok_t > ROBOT_LOST_S:
                    print(f"\n⚠️ Lost robot DO0 for > {ROBOT_LOST_S}s.")
                    print(f"   Details: {e}")
                    print("   -> Sending EMCY STOP (best-effort) and exiting for safety.")
                    try:
                        g.emcy_stop()
                    except Exception:
                        pass
                    break

                time.sleep(READ_BACKOFF_S)
                continue
            now = time.time()

            # If raw changed, reset the stability timer
            if raw != raw_prev:
                raw_prev = raw
                raw_change_t = now

            # Only when raw has been stable long enough AND differs from current debounced state:
            if (now - raw_change_t) >= DEBOUNCE_S and raw_prev != debounced:
                debounced = raw_prev  # <-- THIS is the real edge

                if now - last_cmd_t < CMD_COOLDOWN_S:
                    remain = CMD_COOLDOWN_S - (now - last_cmd_t)
                    print(f"⏳ Ignoring DO0 edge (cooldown {remain:.2f}s left)")
                    time.sleep(dt)
                    continue

                action = "CLOSE" if debounced == 1 else "OPEN"
                logger.info(f"DO0 debounced -> {action}")

                if DRY_RUN:
                    print(f"🟡 DRY_RUN: would {action} (no gripper command sent)")
                else:
                    if debounced == 1:
                        try:
                            g.close(wait=True)
                        except TimeoutError as e:
                            print_timeout_guidance(e)
                            break
                        except Exception as e:
                            logger.error("SusGrip command failed")
                            print("❌ SusGrip command failed.")
                            print(f"   Details: {e}")
                            print("   -> For safety: if anything is still moving, cut 24V power.")
                            print("   -> Bridge exiting for safety.")
                            safe_emcy_stop()
                            break
                    else:
                        try:
                            g.open(wait=True)
                        except TimeoutError as e:
                            print_timeout_guidance(e)
                            break
                        except Exception as e:
                            logger.error("SusGrip command failed")
                            print("❌ SusGrip command failed.")
                            print(f"   Details: {e}")
                            print("   -> For safety: if anything is still moving, cut 24V power.")
                            print("   -> Bridge exiting for safety.")
                            safe_emcy_stop()
                            break

                    last_cmd_t = time.time()

                try:
                    status = g.get_status()
                except Exception as e:
                    logger.error("SusGrip status read failed after command")
                    print("❌ Failed to read SusGrip status after command.")
                    print(f"   Details: {e}")
                    print("   -> For safety: if anything is still moving, cut 24V power.")
                    print("   -> Bridge exiting for safety.")
                    safe_emcy_stop()
                    break
                print(f"Status: POS={status.pos_mm}mm MOT={status.motion} ERR={status.error} OBJ=0x{status.obj:04X}")

                # ---- Safety checks ----
                if status.error != 0:
                    print(f"⚠️ SusGrip ERROR detected: {status.error}. Sending EMCY STOP and exiting.")
                    try:
                        g.emcy_stop()
                    except Exception:
                        pass
                    break

                if status.emcy != 0:
                    print(f"⚠️ SusGrip EMCY is active (0x{status.emcy:04X}). Exiting bridge for safety.")
                    break

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C -> EMCY STOP and exit")
        try:
            g.emcy_stop()
        except Exception:
            pass
    finally:
        g.close()
        print("Disconnected (COM released).")


if __name__ == "__main__":
    main()
