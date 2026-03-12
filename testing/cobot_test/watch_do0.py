import time
from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"
indy = IndyDCP3(ROBOT_IP)

print("Watching DO... Toggle DO0 in CONTY to test.")
print("Press Ctrl+C to stop.\n")

while True:
    do = indy.get_do()
    print("DO:", do)
    time.sleep(0.5)