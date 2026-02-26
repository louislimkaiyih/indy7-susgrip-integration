import time
from neuromeka import IndyDCP3

ROBOT_IP = "192.168.0.157"
DO_ADDR = 0  # DO0

indy = IndyDCP3(ROBOT_IP)


def read_do_state(address: int):
    data = indy.get_do()  # returns {'signals': [{'address': x, 'state': y}, ...]}
    for sig in data.get("signals", []):
        if sig.get("address") == address:
            return sig.get("state")
    return None  # not found (shouldn't happen normally)


last = read_do_state(DO_ADDR)
print(f"Watching DO{DO_ADDR}. Current state = {last}")
print("Toggle DO0 in CONTY. Press Ctrl+C to stop.\n")

while True:
    cur = read_do_state(DO_ADDR)
    if cur != last:
        print(f"DO{DO_ADDR} changed: {last} -> {cur}")
        last = cur
    time.sleep(0.1)
