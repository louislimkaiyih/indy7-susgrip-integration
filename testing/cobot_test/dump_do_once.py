from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"  # change if needed
indy = IndyDCP3(ROBOT_IP)

d = indy.get_do()
print(d)