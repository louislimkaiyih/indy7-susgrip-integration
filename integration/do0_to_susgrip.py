import time

from scripts.gripper.susgrip_client import SusGripClient
from neuromeka import IndyDCP3  # same package you used for get_do()


# ===== Indy7 settings =====
ROBOT_IP = "192.168.0.157"

# ===== SusGrip settings =====
GRIPPER_COM = "COM4"
GRIPPER_ID = 1
OPEN_MAX_MM = 120
CLOSE_MIN_MM = 10

# ===== Behavior =====
POLL_HZ = 10  # how often we check DO0
DEBOUNCE_S = 0.15  # ignore very fast flips
SYNC_ON_START = False  # if True, match gripper to current DO0 immediately


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

    # Fallback: try indexing (rare)
    try:
        return int(data[0])
    except Exception:
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

    busy = False
    last_change_t = 0.0

    try:
        # Make gripper safe
        g.emcy_release()
        g.set_speed_force(speed_mm_s=20, force_pct=10)

        do0 = read_do0(robot)
        print(f"Initial DO0 = {do0} (SYNC_ON_START={SYNC_ON_START})")

        if SYNC_ON_START:
            print("Syncing gripper to DO0 now...")
            if do0 == 1:
                g.open(wait=True)
            else:
                g.close(wait=True)

        prev = do0
        dt = 1.0 / POLL_HZ

        print("\nNow toggle DO0 in CONTY. (0→1=open, 1→0=close)\n")

        while True:
            cur = read_do0(robot)

            if cur != prev:
                now = time.time()

                # debounce
                if now - last_change_t < DEBOUNCE_S:
                    prev = cur
                    continue

                last_change_t = now

                # Edge detected
                if cur == 1:
                    print("DO0 rising (0→1) -> gripper.open()")
                    busy = True
                    g.open(wait=True)
                    busy = False
                else:
                    print("DO0 falling (1→0) -> gripper.close()")
                    busy = True
                    g.close(wait=True)
                    busy = False

                status = g.get_status()
                print(
                    f"Status: POS={status.pos_mm}mm MOT={status.motion} ERR={status.error} OBJ=0x{status.obj:04X}"
                )

                prev = cur

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
