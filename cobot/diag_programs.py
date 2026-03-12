from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"  # change if needed
indy = IndyDCP3(robot_ip=ROBOT_IP, index=0)


def try_call(name):
    fn = getattr(indy, name, None)
    if fn is None:
        return None, "NO_METHOD"
    try:
        return fn(), "OK"
    except Exception as e:
        return repr(e), "ERR"


print("=== Controller ===")
print(indy.get_control_info())

print("\n=== Available indy methods containing 'program' ===")
prog_methods = [m for m in dir(indy) if "program" in m.lower()]
print(prog_methods)

print("\n=== Try likely 'program list' methods ===")
candidates = [
    "get_program_list",
    "get_program_list_data",
    "get_program_list_info",
    "get_program_list_json",
    "get_program_db",
    "get_programs",
]
for name in candidates:
    out, status = try_call(name)
    print(f"{name}: {status}")
    if status != "NO_METHOD":
        print(out)
