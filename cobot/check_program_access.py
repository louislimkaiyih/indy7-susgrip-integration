from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"  # change if needed
KEYWORD = (
    "move_weights_direct"  # <-- change to a short part of your program name (e.g., "PP" or "main")
)

indy = IndyDCP3(ROBOT_IP)

print("=== control_data ===")
cd = indy.get_control_data()
print("op_state:", cd.get("op_state"))
print(
    "running_hours/mins/secs:",
    cd.get("running_hours"),
    cd.get("running_mins"),
    cd.get("running_secs"),
)

print("\n=== on_start_program_config ===")
try:
    cfg = indy.get_on_start_program_config()
    print(cfg)
except Exception as e:
    print("get_on_start_program_config failed:", repr(e))

print(f"\n=== search_program('{KEYWORD}') ===")
try:
    progs = indy.search_program(KEYWORD)
    print(progs)
except Exception as e:
    print("search_program failed:", repr(e))
