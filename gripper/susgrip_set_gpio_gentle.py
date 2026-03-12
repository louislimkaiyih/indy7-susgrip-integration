import time
from pymodbus.client import ModbusSerialClient

# =======================
# CONNECTION SETTINGS
# =======================
PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1
TIMEOUT_S = 0.5

# =======================
# SUSGRIP CONFIG REGISTERS (Holding RW): 0x010x
# (These define IO Only / GPIO behavior)
# =======================
REG_CONFIG_SAVE = 0x0100  # write 0xAAF5 to save to non-volatile memory
REG_CONFIG_MODE = (
    0x0101  # 0=Normal(Modbus), 1=IO Only(GPIO)
)

REG_POS_H = (
    0x0102  # input HIGH target position (OPEN)
)
REG_SPEED_H = 0x0103  # input HIGH speed (mm/s)
REG_ACCEL_H = 0x0104  # input HIGH accel (mm/s/s)

REG_POS_L = (
    0x0105  # input LOW target position (CLOSE)
)
REG_SPEED_L = 0x0106  # input LOW speed (mm/s)
REG_ACCEL_L = 0x0107  # input LOW accel (mm/s/s)

REG_FORCE = 0x0108  # IO holding force (% when uncaged off)
REG_DEFORM = 0x0109  # allowable deformation (mm)
REG_COLLISION = 0x010B  # IO collision/object detect threshold (mArms)

# Extended holding behavior (optional but useful)
REG_UNCAGED = (
    0x0111  # 0x0000 disable (recommended)
)
REG_HOLD_MS = 0x0112  # hold time (ms)
REG_DECAY_MARMS = 0x0113  # decay level (mArms)
REG_CURRISE_LEVEL = 0x0114  # force rise/decay speed 0..3 (0 gentlest)

# =======================
# PRESET: "Delicate cosmetic box" (starting point)
# =======================
OPEN_MM = 60
CLOSE_MM = 10

OPEN_SPEED = 120  # mm/s (open can be faster)
OPEN_ACCEL = 800  # mm/s/s

CLOSE_SPEED = 50  # mm/s 
CLOSE_ACCEL = 150  # mm/s/s (moderate, not snappy)

IO_FORCE_PCT = 5  # start low; raise if it slips
IO_DEFORM_MM = 0  # delicate packaging
IO_COLLISION = 900  # lower = gentler, higher = less sensitive

# Extended behavior:
FORCE_RISE_SPEED = (
    0  # 0=slowest/gentlest, 3=fastest
)
HOLD_TIME_MS = 0  # 0 = no decay (keeps target force until next motion command)
DECAY_LEVEL_MARMS = 0  # irrelevant if HOLD_TIME_MS=0


def _call_with_slave_or_unit(fn, **kwargs):
    """
    PyModbus versions differ: some use slave=, older examples use unit=.
    We try slave first, then unit.
    """
    try:
        return fn(**kwargs, slave=SLAVE_ID)
    except TypeError:
        return fn(**kwargs, unit=SLAVE_ID)


def write_reg(client, addr, val):
    rr = _call_with_slave_or_unit(client.write_register, address=addr, value=val)
    if rr.isError():
        raise RuntimeError(f"Write failed addr=0x{addr:04X}, val={val}, resp={rr}")


def read_reg(client, addr, count=1):
    rr = _call_with_slave_or_unit(
        client.read_holding_registers, address=addr, count=count
    )
    if rr.isError():
        raise RuntimeError(f"Read failed addr=0x{addr:04X}, resp={rr}")
    return rr.registers


def main():
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=8,
        timeout=TIMEOUT_S,
    )

    if not client.connect():
        raise RuntimeError("Could not connect. Check COM port and wiring.")

    try:
        # Safety: keep uncaged OFF
        write_reg(client, REG_UNCAGED, 0x0000)

        # IO Only preset values
        write_reg(client, REG_POS_H, OPEN_MM)
        write_reg(client, REG_SPEED_H, OPEN_SPEED)
        write_reg(client, REG_ACCEL_H, OPEN_ACCEL)

        write_reg(client, REG_POS_L, CLOSE_MM)
        write_reg(client, REG_SPEED_L, CLOSE_SPEED)
        write_reg(client, REG_ACCEL_L, CLOSE_ACCEL)

        write_reg(client, REG_FORCE, IO_FORCE_PCT)
        write_reg(client, REG_DEFORM, IO_DEFORM_MM)
        write_reg(client, REG_COLLISION, IO_COLLISION)

        # Extended holding behavior (gentler ramp)
        write_reg(client, REG_CURRISE_LEVEL, FORCE_RISE_SPEED)
        write_reg(client, REG_HOLD_MS, HOLD_TIME_MS)
        write_reg(client, REG_DECAY_MARMS, DECAY_LEVEL_MARMS)

        # Switch to IO Only (GPIO mode)
        write_reg(
            client, REG_CONFIG_MODE, 1
        )  # 1 = IO Only

        # Save so it persists after power off
        # During save, the gripper becomes unresponsive for up to ~10s; do NOT power off.
        write_reg(client, REG_CONFIG_SAVE, 0xAAF5)
        print("Saving settings (do NOT power off). Waiting 12s...")
        time.sleep(12)

        # Read back a few key regs
        mode = read_reg(client, REG_CONFIG_MODE)[0]
        pos_h = read_reg(client, REG_POS_H)[0]
        pos_l = read_reg(client, REG_POS_L)[0]
        force = read_reg(client, REG_FORCE)[0]
        currise = read_reg(client, REG_CURRISE_LEVEL)[0]

        print("Done.")
        print(f"CONFIG_MODE={mode} (1 means IO Only/GPIO)")
        print(
            f"OPEN pos={pos_h}mm, CLOSE pos={pos_l}mm, FORCE={force}%, CURRISE={currise}"
        )

        print("\nNow test from CONTY:")
        print("- DO ON (24V to blue input)  -> OPEN")
        print("- DO OFF (0V/float)          -> CLOSE")

    finally:
        client.close()


if __name__ == "__main__":
    main()
