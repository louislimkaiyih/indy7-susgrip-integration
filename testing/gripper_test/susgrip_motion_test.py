import time
from pymodbus.client import ModbusSerialClient

PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

# Registers (from manual table)
REG_SET_EMCY = 0x0000  # Holding
REG_SET_POSITION = 0x0001  # Holding (executes immediately)
REG_SET_SPEED = 0x0002  # Holding
REG_SET_FORCE = 0x0004  # Holding

REG_POSITION = 0x0001  # Input
REG_SPEED = 0x0002  # Input
REG_ERROR = 0x0004  # Input
REG_MOTION = 0x0005  # Input
REG_EMCY = 0x0007  # Input

SAFE_SPEED_MM_S = 20
SAFE_FORCE_PCT = 10


def read_in(client, addr, count=1):
    rr = client.read_input_registers(address=addr, count=count, slave=SLAVE_ID)
    if rr.isError():
        raise RuntimeError(rr)
    return rr.registers


def read_hold(client, addr, count=1):
    rr = client.read_holding_registers(address=addr, count=count, slave=SLAVE_ID)
    if rr.isError():
        raise RuntimeError(rr)
    return rr.registers


def write_reg(client, addr, value):
    wr = client.write_register(address=addr, value=value, slave=SLAVE_ID)
    if wr.isError():
        raise RuntimeError(wr)


def twos_comp_16(x):
    return x - 65536 if x >= 32768 else x


def main():
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=8,
        timeout=1.0,
    )

    print(f"Connecting to {PORT} ...")
    if not client.connect():
        print("❌ Cannot open COM port (GUI open? wrong COM?).")
        return

    try:
        # Initial snapshot
        pos0 = read_in(client, REG_POSITION)[0]
        spd0 = twos_comp_16(read_in(client, REG_SPEED)[0])
        err0 = read_in(client, REG_ERROR)[0]
        mot0 = read_in(client, REG_MOTION)[0] & 1
        emc0 = read_in(client, REG_EMCY)[0]
        print(
            f"START: POS={pos0}mm SPEED={spd0} ERROR={err0} MOTION={mot0} EMCY={hex(emc0)}"
        )

        # Safe settings
        write_reg(client, REG_SET_EMCY, 0x0000)
        write_reg(client, REG_SET_SPEED, SAFE_SPEED_MM_S)
        write_reg(client, REG_SET_FORCE, SAFE_FORCE_PCT)

        # Command a move
        OPEN_MAX = 120
        CLOSE_MIN = 10
        EDGE = 2      # within 2mm counts as “at the edge”
        STEP = 20

        # if we're near/open max, go the other direction
        if pos0 >= OPEN_MAX - EDGE:
            target = CLOSE_MIN
        else:
            target = min(pos0 + STEP, OPEN_MAX)

        print(f"Commanding SET_POSITION -> {target}mm")
        before_cmd = read_hold(client, REG_SET_POSITION)[0]
        write_reg(client, REG_SET_POSITION, target)
        after_cmd = read_hold(client, REG_SET_POSITION)[0]
        print(f"SET_POSITION holding reg: before={before_cmd} after={after_cmd}")

        # Wait for motion to START (up to 1.0s)
        started = False
        t0 = time.time()
        while time.time() - t0 < 1.0:
            mot = read_in(client, REG_MOTION)[0] & 1
            pos = read_in(client, REG_POSITION)[0]
            if mot == 1 or pos != pos0:
                started = True
                break
            time.sleep(0.05)

        print("Motion started?", started)

        # Watch for ~2 seconds (so we can SEE changes)
        for i in range(20):
            pos = read_in(client, REG_POSITION)[0]
            spd = twos_comp_16(read_in(client, REG_SPEED)[0])
            err = read_in(client, REG_ERROR)[0]
            mot = read_in(client, REG_MOTION)[0] & 1
            print(f"{i:02d}: POS={pos}mm SPEED={spd} ERROR={err} MOTION={mot}")
            time.sleep(0.1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
