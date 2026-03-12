import time
from pymodbus.client import ModbusSerialClient

# =======================
# YOUR CONNECTION SETTINGS
# =======================
PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

# =======================
# SUSGRIP REGISTERS (Manual)
# Motion Command (Holding RW): 00xx
# =======================
REG_SET_EMCY = 0x0000
REG_SET_POSITION = 0x0001
REG_SET_SPEED = 0x0002
REG_SET_ACCEL = 0x0003
REG_SET_FORCE = 0x0004
REG_SET_DEFORM = 0x0005
REG_SET_COLLISION = 0x0006

# Monitoring (Input RO): 00xx
REG_INFO = 0x0000
REG_POS = 0x0001
REG_SPD = 0x0002
REG_RMS = 0x0003
REG_ERR = 0x0004
REG_MOTION = 0x0005
REG_OBJ = 0x0006
REG_EMCY = 0x0007
REG_VBUS = 0x0008
REG_IBUS = 0x0009


# =======================
# GENTLE "COSMETIC BOX" PRESET
# (You can tune these later)
# =======================
OPEN_MM = 70
CLOSE_MM = 10  # IMPORTANT: manual says 0mm is invalid and can trigger error/EMCY

CLOSE_SPEED_MM_S = 50
CLOSE_ACCEL_MM_S2 = 150

# Holding force in % (0..100) when Uncaged is OFF
HOLD_FORCE_START_PCT = 5
HOLD_FORCE_FINAL_PCT = 5
FORCE_RAMP_STEP_PCT = 0
FORCE_RAMP_STEP_S = 0.25

# Allowable deform (mm): fragile = ~1–3mm
ALLOW_DEFORM_MM = 1

# Collision threshold (mArms): lower = more sensitive, less impact
# If your old gripper "false detects" / gets stuck, increase this.
COLLISION_THRESHOLD_MARMS = (
    900  # try 1200 first; if sticky, try 2000; stuck-rescue 3000-4000
)

POLL_S = 0.10
TIMEOUT_S = 12.0


def twos_comp_16(x: int) -> int:
    return x - 65536 if x >= 32768 else x


def read_in(client, addr, count=1):
    rr = client.read_input_registers(address=addr, count=count, slave=SLAVE_ID)
    if rr.isError():
        raise RuntimeError(rr)
    return rr.registers


def write_reg(client, addr, value):
    wr = client.write_register(address=addr, value=value, slave=SLAVE_ID)
    if wr.isError():
        raise RuntimeError(wr)


def motion_bit(client) -> int:
    # MOTION bit0: 1 moving, 0 stopped
    return read_in(client, REG_MOTION)[0] & 1


def obj_code(client) -> int:
    # OBJ codes (manual):
    # 0x00 no object, 0x01 detected while closing, 0xF1 full force while closing
    return read_in(client, REG_OBJ)[0] & 0x00FF


def snapshot(client) -> dict:
    pos = read_in(client, REG_POS)[0]
    spd = twos_comp_16(read_in(client, REG_SPD)[0])
    rms = twos_comp_16(read_in(client, REG_RMS)[0])
    err = read_in(client, REG_ERR)[0] & 0x00FF
    mot = motion_bit(client)
    obj = obj_code(client)
    emc = read_in(client, REG_EMCY)[0]
    vbus = read_in(client, REG_VBUS)[0]
    ibus = read_in(client, REG_IBUS)[0]
    return dict(
        pos=pos,
        spd=spd,
        rms=rms,
        err=err,
        mot=mot,
        obj=obj,
        emc=emc,
        vbus=vbus,
        ibus=ibus,
    )


def print_snap(tag: str, s: dict):
    print(
        f"{tag:>6}  POS={s['pos']:3d}  SPD={s['spd']:4d}  RMS={s['rms']:5d}  "
        f"MOT={s['mot']}  OBJ=0x{s['obj']:02X}  ERR={s['err']}  EMCY=0x{s['emc']:04X}  "
        f"V={s['vbus']}mV  I={s['ibus']}mA"
    )


def emcy_stop(client):
    # EMCY engage: write 0xFFFF (stops motion + drops holding force)
    write_reg(client, REG_SET_EMCY, 0xFFFF)


def emcy_release(client):
    # EMCY release: write 0x0000 (also clears non-persistent errors)
    write_reg(client, REG_SET_EMCY, 0x0000)


def set_gentle_params(client):
    # These are the key “gentle” knobs:
    write_reg(client, REG_SET_SPEED, CLOSE_SPEED_MM_S)
    write_reg(client, REG_SET_ACCEL, CLOSE_ACCEL_MM_S2)
    write_reg(client, REG_SET_FORCE, HOLD_FORCE_START_PCT)
    write_reg(client, REG_SET_DEFORM, ALLOW_DEFORM_MM)
    write_reg(client, REG_SET_COLLISION, COLLISION_THRESHOLD_MARMS)


def move_to(client, target_mm: int, timeout_s: float = TIMEOUT_S) -> bool:
    """
    Command SET_POSITION (executes immediately).
    Wait for motion to start then stop (more reliable than only waiting for stop).
    """
    s0 = snapshot(client)
    write_reg(client, REG_SET_POSITION, int(target_mm))

    # wait start
    t0 = time.time()
    started = False
    while time.time() - t0 < 1.0:
        s = snapshot(client)
        if s["mot"] == 1 or s["pos"] != s0["pos"]:
            started = True
            break
        time.sleep(0.05)

    # wait stop
    t1 = time.time()
    while time.time() - t1 < timeout_s:
        s = snapshot(client)
        if s["mot"] == 0:
            return True
        time.sleep(0.05)

    return False


def gentle_grip(client):
    """
    Close until object is detected, then gently ramp holding force a bit.
    Fix: wait for motion to START before treating MOTION==0 as "done".
    """
    print("\nSetting gentle parameters...")
    set_gentle_params(client)

    print(f"\nCommand CLOSE -> {CLOSE_MM}mm (watch OBJ)...")

    # Remember starting position so we can detect "movement started"
    s0 = snapshot(client)
    write_reg(client, REG_SET_POSITION, CLOSE_MM)

    # --- NEW: wait for motion to start (or position to change) ---
    started = False
    t_start = time.time()
    while time.time() - t_start < 1.0:
        s = snapshot(client)
        # If MOTION becomes 1 OR position changes, we consider movement started
        if s["mot"] == 1 or s["pos"] != s0["pos"]:
            started = True
            break
        time.sleep(0.05)
    # -------------------------------------------------------------

    t0 = time.time()
    detected = False
    reached_close = False

    while time.time() - t0 < TIMEOUT_S:
        s = snapshot(client)
        print_snap("CLOSE", s)

        # error or emcy -> stop
        if s["err"] != 0 or s["emc"] == 0xFFFF:
            print("❌ ERROR or EMCY detected -> stopping for safety.")
            emcy_stop(client)
            return

        # object detection logic
        if s["obj"] in (0x01, 0xF1):
            detected = True
            print(f"✅ Object detected (OBJ=0x{s['obj']:02X}) at ~POS={s['pos']}mm")
            break

        # If we've reached close position and stopped, we're done (empty close)
        if s["pos"] <= CLOSE_MM + 1 and s["mot"] == 0:
            reached_close = True
            break

        # IMPORTANT CHANGE:
        # Only allow "MOT==0 means done" AFTER motion has started.
        # This prevents the early-exit bug you saw.
        if started and s["mot"] == 0:
            break

        time.sleep(POLL_S)

    if not detected:
        if reached_close:
            print("✅ No object detected (empty). Reached close target safely.")
        elif not started:
            print(
                "⚠️ Close motion never started. (GUI holding port? command ignored? already at target?)"
            )
        else:
            print(
                "⚠️ No object detected. If you DID place an object, collision threshold may be too high."
            )
        return

    # Gentle tighten WITHOUT extra travel: ramp holding force slightly
    print("\nRamping holding force gently (to secure grip without crushing)...")
    force = HOLD_FORCE_START_PCT
    while force < HOLD_FORCE_FINAL_PCT:
        force = min(force + FORCE_RAMP_STEP_PCT, HOLD_FORCE_FINAL_PCT)
        write_reg(client, REG_SET_FORCE, force)
        s = snapshot(client)
        print_snap(f"FORCE{force:02d}", s)

        if s["err"] != 0:
            print("⚠️ Error occurred during force ramp -> stopping for safety.")
            emcy_stop(client)
            return

        time.sleep(FORCE_RAMP_STEP_S)

    print("\n✅ Grip test finished. (Holding at gentle force.)")


def main():
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=8,
        timeout=1.0,
    )

    print("=== SusGrip Delicate Grip Test (Normal/Modbus mode) ===")
    print("Safety: keep fingers away; close SusGrip GUI.\n")

    if not client.connect():
        print("❌ Cannot open COM port. (GUI open? wrong COM?)")
        return

    try:
        # release emergency & clear errors
        emcy_release(client)

        info = read_in(client, REG_INFO)[0]
        print(f"Connected. INFO={info}  (type mailbox)\n")

        # 1) OPEN fully
        print(f"Opening to {OPEN_MM}mm...")
        ok = move_to(client, OPEN_MM, timeout_s=15.0)
        print_snap("OPENED", snapshot(client))
        if not ok:
            print("⚠️ Open timeout -> engaging EMCY STOP.")
            emcy_stop(client)
            return

        input(
            "\nPlace the cosmetic box between fingers (hands clear), then press Enter..."
        )

        # 2) GENTLE GRIP
        gentle_grip(client)

        input("\nPress Enter to OPEN (release object)...")
        ok = move_to(client, OPEN_MM, timeout_s=15.0)
        print_snap("OPENED", snapshot(client))
        if not ok:
            print("⚠️ Open timeout -> engaging EMCY STOP.")
            emcy_stop(client)

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C -> engaging EMCY STOP")
        try:
            emcy_stop(client)
        except Exception:
            pass
    finally:
        client.close()
        print("\nDisconnected (COM released).")


if __name__ == "__main__":
    main()
