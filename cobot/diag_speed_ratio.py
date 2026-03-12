from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"  # change if needed
indy = IndyDCP3(robot_ip=ROBOT_IP, index=0)

print("=== speed-related methods ===")
print([m for m in dir(indy) if "speed" in m.lower() or "ratio" in m.lower()])

print("\n=== BEFORE ===")
print("motion:", indy.get_motion_data())
print("program:", indy.get_program_data())

# Try to set speed ratio to 20 (often means 20%)
if hasattr(indy, "set_speed_ratio"):
    try:
        indy.set_speed_ratio(100)
        print("\nset_speed_ratio(20) called: OK")
    except Exception as e:
        print("\nset_speed_ratio(20) error:", repr(e))
else:
    print("\nNo set_speed_ratio method on this controller/API.")

print("\n=== AFTER ===")
print("motion:", indy.get_motion_data())
print("program:", indy.get_program_data())
