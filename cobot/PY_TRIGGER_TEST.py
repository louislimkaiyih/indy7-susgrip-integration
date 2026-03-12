import time
from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"
DO_ADDR = 0

indy = IndyDCP3(ROBOT_IP)


def do_state(addr):
    d = indy.get_do()
    for s in d.get("signals", []):
        if s.get("address") == addr:
            return s.get("state")
    return None


pd = indy.get_program_data()
prog_path = pd.get("program_name")
print("op_state:", indy.get_control_data().get("op_state"))
print("program_name (from controller):", prog_path)
print("DO0 before:", do_state(DO_ADDR))

# safety: ensure stopped first
try:
    indy.stop_program()
except Exception as e:
    print("stop_program error (ok to ignore if already stopped):", repr(e))

print("Calling play_program(prog_name=prog_path) ...")
try:
    indy.play_program(prog_name=prog_path)
    print("play_program returned")
except Exception as e:
    print("play_program error:", repr(e))

t0 = time.time()
last = None
print("Watching DO0...")
while time.time() - t0 < 5.0:
    cur = do_state(DO_ADDR)
    if cur != last:
        print(f"  t={time.time()-t0:.2f}s  DO0={cur}")
        last = cur
    time.sleep(0.05)

print("DO0 after:", do_state(DO_ADDR))
print("Done.")
