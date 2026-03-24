# Indy7 ↔ SusGrip 2F Integration (Windows / Python)

This project controls a SusGrip 2F gripper via Modbus RTU (RS485) from a Windows laptop using Python, and integrates it with a Neuromeka Indy7 cobot digital output (DO0).

Goal:
CONTY toggles DO0 → Python detects edge → Python commands SusGrip open/close.

---

## Safety

- Keep fingers away from the gripper during tests.
- Stopping Python does NOT stop gripper motion.
- To stop motion: write 0xFFFF to SET_EMCY (register 0x0000).
- Close SusGrip GUI before running Python.
- If something behaves unexpectedly: cut 24V power.

---

## System Overview

Indy7 Controller IP:
192.168.0.157

SusGrip Connection:
USB-to-RS485 adapter → Windows COM port (e.g. COM4)

Modbus Settings:
- Slave ID: 1
- Baudrate: 115200
- Parity: N
- Stop bits: 1

---

## Folder Structure

Project root: C:\indy_test

```
scripts/
  gripper/
  cobot/
  integration/
  testing/
    gripper_test/
    cobot_test/
```

---

## Python Setup (Windows)

Activate virtual environment:

```
cd C:\indy_test
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```
python -m pip install neuromeka==3.4.2.0 grpcio==1.59.0 protobuf==4.25.4
python -m pip install pymodbus pyserial
```

---

## How To Run

Gripper open/close test:

```
cd C:\indy_test
python C:\indy_test\scripts\gripper\test_client_open_close.py
```

Integration test (DO0 → SusGrip):

```
cd C:\indy_test
python -m scripts.integration.do0_to_susgrip
```

Expected behavior:
DO0 0→1 → gripper.open()
DO0 1→0 → gripper.close()

---

## Troubleshooting

PermissionError / Access denied:
SusGrip GUI is still open. Close it.

No module named scripts:
Run from project root:
cd C:\indy_test
python -m scripts.integration.do0_to_susgrip

Gripper keeps moving:
Use EMCY STOP or cut 24V power.
