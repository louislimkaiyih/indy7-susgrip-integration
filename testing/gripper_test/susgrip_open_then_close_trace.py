import time
from pymodbus.client import ModbusSerialClient

PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

# Holding (RW) motion commands:contentReference[oaicite:6]{index=6}
REG_SET_EMCY     = 0x0000
REG_SET_POSITION = 0x0001
REG_SET_SPEED    = 0x0002
REG_SET_FORCE    = 0x0004

# Input (RO) monitoring registers:contentReference[oaicite:7]{index=7}
REG_POSITION = 0x0001
REG_SPEED    = 0x0002
REG_ERROR    = 0x0004
REG_MOTION   = 0x0005
REG_OBJ      = 0x0006
REG_EMCY     = 0x0007
REG_VBUS     = 0x0008
REG_IBUS     = 0x0009

SAFE_SPEED_MM_S = 20
SAFE_FORCE_PCT  = 10

OPEN_TARGET_MM  = 120
CLOSE_TARGET_MM = 10

def twos_comp_16(x):
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

def motion_bit(client):
    # MOTION register holds motion status; bit0 is the status bit:contentReference[oaicite:8]{index=8}
    return read_in(client, REG_MOTION)[0] & 1

def wait_move_done(client, start_timeout_s=1.0, stop_timeout_s=12.0):
    """
    Waits for motion to START (MOTION bit becomes 1),
    then waits for motion to STOP (MOTION bit becomes 0).
    This avoids the 'MOTION was 0 for a split second' bug.
    """
    # Wait for motion to start
    t0 = time.time()
    started = False
    while time.time() - t0 < start_timeout_s:
        if motion_bit(client) == 1:
            started = True
            break
        time.sleep(0.02)

    # If it never started, it might already be at target (or ignored).
    # We'll still wait briefly for STOP, but don't block forever.
    t1 = time.time()
    while time.time() - t1 < stop_timeout_s:
        if motion_bit(client) == 0:
            return True
        time.sleep(0.05)

    return False

def print_line(t0, tag, client):
    pos = read_in(client, REG_POSITION)[0]
    spd = twos_comp_16(read_in(client, REG_SPEED)[0])
    err = read_in(client, REG_ERROR)[0]
    mot = motion_bit(client)
    obj = read_in(client, REG_OBJ)[0]
    emc = read_in(client, REG_EMCY)[0]
    vbus_mv = read_in(client, REG_VBUS)[0]
    ibus_ma = read_in(client, REG_IBUS)[0]
    dt = time.time() - t0
    print(f"{dt:6.2f}s {tag:>5}  POS={pos:3d}  SPD={spd:4d}  MOT={mot}  OBJ=0x{obj:04X}  ERR={err}  EMCY=0x{emc:04X}  V={vbus_mv}mV  I={ibus_ma}mA")

def main():
    client = ModbusSerialClient(
        port=PORT, baudrate=BAUDRATE, parity=PARITY, stopbits=STOPBITS,
        bytesize=8, timeout=1.0
    )

    print(f"Connecting to {PORT} ...")
    if not client.connect():
        print("❌ Cannot open COM port (GUI open? wrong COM?).")
        return

    try:
        # Clear EMCY + set safe speed/force
        # EMCY engaged stops all motion and drops holding force:contentReference[oaicite:9]{index=9}
        write_reg(client, REG_SET_EMCY, 0x0000)
        write_reg(client, REG_SET_SPEED, SAFE_SPEED_MM_S)
        write_reg(client, REG_SET_FORCE, SAFE_FORCE_PCT)

        t0 = time.time()
        print_line(t0, "START", client)

        # 1) OPEN first
        print(f"\nCommand OPEN -> {OPEN_TARGET_MM}mm (SET POSITION executes immediately)")
        write_reg(client, REG_SET_POSITION, OPEN_TARGET_MM)
        ok = wait_move_done(client, start_timeout_s=1.0, stop_timeout_s=12.0)
        print_line(t0, "OPEN", client)
        if not ok:
            print("⚠️ Open did not reach 'stopped' within timeout. Engaging EMCY.")
            write_reg(client, REG_SET_EMCY, 0xFFFF)
            return

        time.sleep(0.5)

        # 2) CLOSE with trace
        print(f"\nCommand CLOSE -> {CLOSE_TARGET_MM}mm, tracing for 10s ...")
        write_reg(client, REG_SET_POSITION, CLOSE_TARGET_MM)

        TOL = 1          # allow 1mm tolerance
        MAX_CLOSE_S = 12 # safety timeout so it won't loop forever

        t_close_start = time.time()
        while True:
            print_line(t0, "CLOSE", client)

            pos = read_in(client, REG_POSITION)[0]
            mot = motion_bit(client)

            # Stop collecting once it's closed AND stopped
            if pos <= CLOSE_TARGET_MM + TOL and mot == 0:
                print("✅ Reached close target and stopped. Ending trace.")
                break

            # Safety timeout
            if time.time() - t_close_start > MAX_CLOSE_S:
                print("⚠️ Close trace timeout -> engaging EMCY STOP")
                write_reg(client, REG_SET_EMCY, 0xFFFF)
                break

            time.sleep(0.10)

        # If still moving at end, stop it
        if motion_bit(client) == 1:
            print("⚠️ Still moving -> engaging EMCY STOP")
            write_reg(client, REG_SET_EMCY, 0xFFFF)

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C -> engaging EMCY STOP")
        try:
            write_reg(client, REG_SET_EMCY, 0xFFFF)
        except Exception:
            pass
    finally:
        client.close()

if __name__ == "__main__":
    main()