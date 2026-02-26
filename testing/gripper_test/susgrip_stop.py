from pymodbus.client import ModbusSerialClient

PORT = "COM4"
SLAVE_ID = 1
BAUDRATE = 115200
PARITY = "N"
STOPBITS = 1

REG_SET_EMCY = 0x0000   # Holding RW
REG_MOTION   = 0x0005   # Input RO (bit0)
REG_EMCY     = 0x0007   # Input RO (mirrors SET_EMCY)
REG_ERROR    = 0x0004   # Input RO

def read_in(client, addr, count=1):
    rr = client.read_input_registers(address=addr, count=count, slave=SLAVE_ID)
    if rr.isError():
        raise RuntimeError(rr)
    return rr.registers

def write_reg(client, addr, value):
    wr = client.write_register(address=addr, value=value, slave=SLAVE_ID)
    if wr.isError():
        raise RuntimeError(wr)

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
        print("ENGAGING EMCY (STOP NOW) ...")
        write_reg(client, REG_SET_EMCY, 0xFFFF)

        err = read_in(client, REG_ERROR)[0]
        motion = read_in(client, REG_MOTION)[0] & 1
        emcy = read_in(client, REG_EMCY)[0]

        print(f"ERROR={err}")
        print(f"MOTION(bit0)={motion}  (0=stopped, 1=moving)")
        print(f"EMCY status = {hex(emcy)}  (expect 0xffff when engaged)")
    finally:
        client.close()

if __name__ == "__main__":
    main()