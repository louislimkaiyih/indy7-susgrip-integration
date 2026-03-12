from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"  # <- change if needed

indy = IndyDCP3(robot_ip=ROBOT_IP, index=0)

def safe(name, fn):
    print(f"\n===== {name} =====")
    try:
        out = fn()
        print(out)
    except Exception as e:
        print("ERROR:", repr(e))

safe("get_control_info", indy.get_control_info)
safe("get_robot_data", indy.get_robot_data)
safe("get_motion_data", indy.get_motion_data)
safe("get_program_data", indy.get_program_data)
safe("get_safety_control_data", indy.get_safety_control_data)