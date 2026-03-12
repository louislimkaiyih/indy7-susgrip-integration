from neuromeka import IndyDCP3

ROBOT_IP = "192.168.1.75"
indy = IndyDCP3(ROBOT_IP)

pd = indy.get_program_data()
print("program_name:", pd.get("program_name"))
print("program_state:", pd.get("program_state"))
print("speed_ratio:", pd.get("speed_ratio"))
print("program_alarm:", pd.get("program_alarm"))