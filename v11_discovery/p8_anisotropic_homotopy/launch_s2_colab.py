import os
import subprocess
import time
import zipfile

import torch

os.makedirs("/content/sha_lm", exist_ok=True)
with zipfile.ZipFile("/content/p8_bundle.zip", "r") as zf:
    zf.extractall("/content/sha_lm")

os.chdir("/content/sha_lm")
print(
    "PyTorch:",
    torch.__version__,
    "CUDA:",
    torch.cuda.is_available(),
    "GPU:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    flush=True,
)
res_dir = "/content/sha_lm/v11_discovery/p8_anisotropic_homotopy/results"
os.makedirs(res_dir, exist_ok=True)
log_path = f"{res_dir}/run_log_s2.txt"

runner_code = """
import shutil, subprocess, sys
r1 = subprocess.run([sys.executable, "-u", "-m", "v11_discovery.p8_anisotropic_homotopy.run_p8_experiments", "--rounds", "64", "--skip-2d", "--skip-cutset"], check=True)
shutil.move("v11_discovery/p8_anisotropic_homotopy/results/homotopy_sweep.json", "v11_discovery/p8_anisotropic_homotopy/results/homotopy_sweep_r64.json")
r2 = subprocess.run([sys.executable, "-u", "-m", "v11_discovery.p8_anisotropic_homotopy.run_p8_experiments", "--skip-sweep"], check=True)
print("[P8 S2] ALL DONE", flush=True)
"""

with open("/content/sha_lm/run_s2_seq.py", "w", encoding="utf-8") as f:
    f.write(runner_code)

with open(log_path, "wb") as log_f:
    proc = subprocess.Popen(
        ["python3", "-u", "/content/sha_lm/run_s2_seq.py"],
        cwd="/content/sha_lm",
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print("Launched S2 PID:", proc.pid, flush=True)
time.sleep(3)
print("Poll:", proc.poll(), flush=True)
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        print("--- S2 LOG HEAD ---", flush=True)
        print(f.read(2000), flush=True)
