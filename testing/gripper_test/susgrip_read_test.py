from pymodbus.client import ModbusSerialClient

# ======= EDIT THESE to match your SusGrip GUI =======
PORT = "COM4"  # e.g., "COM5"
SLAVE_ID = 1  # Modbus server/slave id
BAUDRATE = 115200  # e.g., 115200
PARITY = "N"  # "N" (None), "E" (Even), "O" (Odd)
STOPBITS = 1  # 1 or 2
# ================================================


def main():
    client = ModbusSerialClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=8,  # SusGrip uses 8 data bits
        timeout=1.0,
    )

    print(f"Connecting to {PORT} (id={SLAVE_ID}, {BAUDRATE}-{PARITY}-{STOPBITS}) ...")
    if not client.connect():
        print("❌ Failed to open serial port. (Wrong COM? Port busy? Driver issue?)")
        return

    try:
        # SusGrip Monitoring Registers are Input Registers (FC04)
        # INFO = address 0x0000, POSITION = address 0x0001 in the manual map.:contentReference[oaicite:4]{index=4}
        rr = client.read_input_registers(address=0, count=2, slave=SLAVE_ID)

        if rr.isError():
            print(f"❌ Modbus error response: {rr}")
            return

        info_raw = rr.registers[0]
        pos_mm = rr.registers[1]

        print("✅ Response received!")
        print(f"INFO raw      = {info_raw}   (gripper type mailbox)")
        print(f"POSITION (mm)  = {pos_mm}     (instant finger distance in mm)")

    finally:
        client.close()


if __name__ == "__main__":
    main()
