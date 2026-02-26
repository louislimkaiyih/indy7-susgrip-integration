from neuromeka import IndyDCP3

ROBOT_IP = "192.168.0.157"
indy = IndyDCP3(ROBOT_IP)

data = indy.get_robot_data()

print("op_state:", data.get("op_state"))
print("q (deg):", data.get("q"))
print("qdot:", data.get("qdot"))

print("DI:", indy.get_di())
print("DO:", indy.get_do())
