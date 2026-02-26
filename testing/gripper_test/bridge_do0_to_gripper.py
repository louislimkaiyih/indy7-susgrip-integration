import time
from neuromeka import IndyDCP3

ROBOT_IP = "192.168.0.157"
DO_ADDR = 0  # DO0

indy = IndyDCP3(ROBOT_IP)

def read_do_state(address: int):
    data = indy.get_do()  # {'signals': [{'address': x, 'state': y}, ...]}
    for sig in data.get("signals", []):
        if sig.get("address") == address:
            return sig.get("state")
    return None

# Placeholder actions (we’ll replace these with real SusGrip commands later)
def gripper_open():
    print(">>> GRIPPER OPEN (placeholder)")

def gripper_close():
    print(">>> GRIPPER CLOSE (placeholder)")

last = read_do_state(DO_ADDR)
print(f"Bridge running. Watching DO{DO_ADDR}. Current state = {last}")
print("Toggle DO0 in CONTY. Press Ctrl+C to stop.\n")

while True:
    cur = read_do_state(DO_ADDR)

    # Only trigger when the value CHANGES (edge trigger)
    if last == 0 and cur == 1:
        gripper_open()
    elif last == 1 and cur == 0:
        gripper_close()

    last = cur
    time.sleep(0.05)