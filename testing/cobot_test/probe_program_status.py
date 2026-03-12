from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"
indy = IndyDCP3(ROBOT_IP)

print("op_state:", indy.get_control_data().get("op_state"))

# 1) Can we read program data?
try:
    pd = indy.get_program_data()
    print("\nget_program_data(): OK")
    print(pd)
except Exception as e:
    print("\nget_program_data(): FAILED ->", repr(e))

# 2) Does stop/pause also get cancelled? (should be safe)
for fn_name in ["stop_program", "pause_program", "resume_program"]:
    fn = getattr(indy, fn_name, None)
    if fn is None:
        continue
    try:
        fn()
        print(f"{fn_name}(): OK")
    except Exception as e:
        print(f"{fn_name}(): FAILED ->", repr(e))