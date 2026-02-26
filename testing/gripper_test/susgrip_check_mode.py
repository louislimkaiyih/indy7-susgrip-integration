from pymodbus.client import ModbusSerialClient

PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

REG_CONFIG_MODE = 0x0101  # CONFIG MODE (Holding Register)


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
        print("❌ Cannot open COM port (GUI still open? wrong COM?).")
        return

    try:
        rr = client.read_holding_registers(
            address=REG_CONFIG_MODE, count=1, slave=SLAVE_ID
        )
        if rr.isError():
            print(f"❌ Read error: {rr}")
            return

        mode = rr.registers[0]
        if mode == 0:
            print("✅ CONFIG MODE = 0  (Normal mode: Modbus controls motion)")
        elif mode == 1:
            print(
                "⚠️ CONFIG MODE = 1  (IO Only mode: blue input pin overrides Modbus motion)"
            )
        else:
            print(f"⚠️ CONFIG MODE = {mode}  (unexpected/reserved value)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
