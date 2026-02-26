from pymodbus.client import ModbusSerialClient

PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

REG_SET_SPEED = 0x0002  # Holding Register: SET SPEED (RW)
REG_ERROR = 0x0004  # Input Register: ERROR (RO)
REG_MOTION = 0x0005  # Input Register: MOTION (RO)

SAFE_SPEED_MM_S = 20


def main():
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=8,
        timeout=1.0,
    )

    print(f"Connecting to {PORT} (id={SLAVE_ID}, {BAUDRATE}-{PARITY}-{STOPBITS}) ...")
    if not client.connect():
        print("❌ Failed to open serial port (GUI open? wrong COM?).")
        return

    try:
        # Read current speed (holding reg)
        r0 = client.read_holding_registers(
            address=REG_SET_SPEED, count=1, slave=SLAVE_ID
        )
        if r0.isError():
            print(f"❌ Read holding register error: {r0}")
            return
        print(f"Current SET SPEED = {r0.registers[0]} mm/s")

        # Read status (input regs)
        st0 = client.read_input_registers(address=REG_ERROR, count=2, slave=SLAVE_ID)
        if st0.isError():
            print(f"❌ Read status error: {st0}")
            return
        print(f"Status before: ERROR={st0.registers[0]}  MOTION={st0.registers[1]}")

        # Write new speed (should NOT move)
        w = client.write_register(
            address=REG_SET_SPEED, value=SAFE_SPEED_MM_S, slave=SLAVE_ID
        )
        if w.isError():
            print(f"❌ Write register error: {w}")
            return
        print(f"Wrote SET SPEED = {SAFE_SPEED_MM_S} mm/s")

        # Read back
        r1 = client.read_holding_registers(
            address=REG_SET_SPEED, count=1, slave=SLAVE_ID
        )
        print(f"Read-back SET SPEED = {r1.registers[0]} mm/s")

        st1 = client.read_input_registers(address=REG_ERROR, count=2, slave=SLAVE_ID)
        print(f"Status after:  ERROR={st1.registers[0]}  MOTION={st1.registers[1]}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
