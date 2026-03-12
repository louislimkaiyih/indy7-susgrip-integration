import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} RealSense device(s).")

if len(devices) == 0:
    print("No RealSense device found by Python.")
    exit()

for i, dev in enumerate(devices):
    print(f"\nDevice {i}")
    print("Name:", dev.get_info(rs.camera_info.name))
    print("Serial Number:", dev.get_info(rs.camera_info.serial_number))
    print("Firmware Version:", dev.get_info(rs.camera_info.firmware_version))
