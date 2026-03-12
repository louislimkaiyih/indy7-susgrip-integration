import time
from dataclasses import dataclass
from pymodbus.client import ModbusSerialClient


@dataclass
class SusGripStatus:
    pos_mm: int
    speed_mm_s: int
    motion: int          # 0=stopped, 1=moving
    error: int
    obj: int
    emcy: int
    vbus_mv: int
    ibus_ma: int


class SusGripClient:
    # ===== Register addresses (confirmed working in your tests) =====
    # Holding (RW)
    REG_SET_EMCY     = 0x0000
    REG_SET_POSITION = 0x0001
    REG_SET_SPEED    = 0x0002
    REG_SET_FORCE    = 0x0004

    # Input (RO)
    REG_POSITION = 0x0001
    REG_SPEED    = 0x0002
    REG_ERROR    = 0x0004
    REG_MOTION   = 0x0005
    REG_OBJ      = 0x0006
    REG_EMCY     = 0x0007
    REG_VBUS     = 0x0008
    REG_IBUS     = 0x0009

    def __init__(
        self,
        port: str = "COM4",
        slave_id: int = 1,
        baudrate: int = 115200,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        timeout: float = 1.0,
        # Your practical limits (you used these already)
        open_max_mm: int = 120,
        close_min_mm: int = 10,
    ):
        self.port = port
        self.slave_id = slave_id
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize,
            timeout=timeout,
        )
        self.open_max_mm = open_max_mm
        self.close_min_mm = close_min_mm

    # ---------- Low-level helpers ----------
    def connect(self) -> bool:
        """Open the COM port connection."""
        return self.client.connect()

    def close(self) -> None:
        """Close the COM port connection."""
        self.client.close()

    def _read_input(self, addr: int, count: int = 1) -> list[int]:
        rr = self.client.read_input_registers(address=addr, count=count, slave=self.slave_id)
        if rr.isError():
            raise RuntimeError(f"Read input regs error: {rr}")
        return rr.registers

    def _write_holding(self, addr: int, value: int) -> None:
        wr = self.client.write_register(address=addr, value=value, slave=self.slave_id)
        if wr.isError():
            raise RuntimeError(f"Write holding reg error: {wr}")

    @staticmethod
    def _to_signed_16(x: int) -> int:
        """Convert 0..65535 into signed -32768..32767 (useful for SPEED)."""
        return x - 65536 if x >= 32768 else x

    def _motion_bit(self) -> int:
        return self._read_input(self.REG_MOTION, 1)[0] & 1

    def wait_move_done(self, start_timeout_s: float = 1.0, stop_timeout_s: float = 12.0) -> bool:
        """
        Wait for motion to START (MOTION becomes 1),
        then wait for motion to STOP (MOTION becomes 0).
        This avoids the 'MOTION was 0 for a split second' bug.
        """
        # wait for start
        t0 = time.time()
        started = False
        while time.time() - t0 < start_timeout_s:
            if self._motion_bit() == 1:
                started = True
                break
            time.sleep(0.02)

        # even if it didn't start (already at target), still wait for stop briefly
        t1 = time.time()
        while time.time() - t1 < stop_timeout_s:
            if self._motion_bit() == 0:
                return True
            time.sleep(0.05)

        # If we get here, it never became "stopped" within time.
        return False

    # ---------- User-friendly API ----------
    def get_status(self) -> SusGripStatus:
        pos = self._read_input(self.REG_POSITION, 1)[0]
        spd = self._to_signed_16(self._read_input(self.REG_SPEED, 1)[0])
        err = self._read_input(self.REG_ERROR, 1)[0]
        mot = self._motion_bit()
        obj = self._read_input(self.REG_OBJ, 1)[0]
        emc = self._read_input(self.REG_EMCY, 1)[0]
        vbus = self._read_input(self.REG_VBUS, 1)[0]
        ibus = self._read_input(self.REG_IBUS, 1)[0]
        return SusGripStatus(pos, spd, mot, err, obj, emc, vbus, ibus)

    def set_speed_force(self, speed_mm_s: int = 20, force_pct: int = 10) -> None:
        # simple safety clamps
        speed_mm_s = max(1, min(int(speed_mm_s), 200))
        force_pct = max(1, min(int(force_pct), 100))
        self._write_holding(self.REG_SET_SPEED, speed_mm_s)
        self._write_holding(self.REG_SET_FORCE, force_pct)

    def set_position_mm(self, mm: int, wait: bool = True) -> bool:
        """Move to an absolute opening (mm). Returns True if motion finished."""
        mm = int(mm)
        # clamp to your practical safe range
        mm = max(self.close_min_mm, min(mm, self.open_max_mm))

        self._write_holding(self.REG_SET_POSITION, mm)

        if not wait:
            return True

        ok = self.wait_move_done()
        if not ok:
            # Watchdog: if motion didn't finish in time, try to stop immediately.
            try:
                self.emcy_stop()
            except Exception:
                # If comms are broken, we might not be able to stop via Modbus.
                # In real life safety: cut 24V power.
                pass

            raise TimeoutError(f"SusGrip move timeout: target={mm}mm (sent EMCY STOP).")

        return True

    def open(self, wait: bool = True) -> bool:
        return self.set_position_mm(self.open_max_mm, wait=wait)

    def close(self, wait: bool = True) -> bool:
        return self.set_position_mm(self.close_min_mm, wait=wait)

    def emcy_stop(self) -> None:
        """Emergency stop: stops motion and drops holding force."""
        self._write_holding(self.REG_SET_EMCY, 0xFFFF)

    def emcy_release(self) -> None:
        """Release emergency stop (allow motion again)."""
        self._write_holding(self.REG_SET_EMCY, 0x0000)