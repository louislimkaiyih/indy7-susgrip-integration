import grpc
from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"
indy = IndyDCP3(ROBOT_IP)

def try_call(name, fn):
    try:
        result = fn()
        print(f"{name}: OK")
        # print small preview (avoid huge prints)
        print(str(result)[:200])
    except grpc.RpcError as e:
        print(f"{name}: FAIL -> code={e.code()} details={e.details()!r}")

try_call("get_control_state", indy.get_control_data)
try_call("get_robot_data", indy.get_robot_data)
try_call("get_di", indy.get_di)
try_call("get_do", indy.get_do)