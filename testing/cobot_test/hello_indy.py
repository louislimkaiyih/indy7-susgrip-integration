from neuromeka import IndyDCP3

ROBOT_IP = "192.168.0.157"
indy = IndyDCP3(ROBOT_IP)

print("control_state:", indy.get_control_data())
print("robot_data:", indy.get_robot_data())
print("DI:", indy.get_di())
print("DO:", indy.get_do())