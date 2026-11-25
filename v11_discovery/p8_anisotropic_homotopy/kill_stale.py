import os
import subprocess
import time

ps_out = subprocess.check_output(["ps", "-ef"], text=True)
for ln in ps_out.splitlines():
    if "run_p8_experiments" in ln:
        parts = ln.split()
        pid = int(parts[1])
        try:
            os.kill(pid, 9)
            print("Killed PID:", pid, flush=True)
        except Exception as e:
            print("Failed kill:", pid, e, flush=True)
time.sleep(1)
