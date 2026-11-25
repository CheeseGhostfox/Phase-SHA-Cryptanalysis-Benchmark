import os
import subprocess

res_dir = "/content/sha_lm/v11_discovery/p8_anisotropic_homotopy/results"
ps_out = subprocess.check_output(["ps", "-ef"], text=True)
procs = [ln for ln in ps_out.splitlines() if ("run_p8_experiments" in ln or "run_s2_seq" in ln)]
print("PROCS:", procs if procs else "NONE", flush=True)
if os.path.exists(res_dir):
    files = sorted(os.listdir(res_dir))
    print("FILES:", [(fn, os.path.getsize(os.path.join(res_dir, fn))) for fn in files], flush=True)
log_path = os.path.join(res_dir, "run_log_s2.txt")
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"LOG LINES: {len(lines)}", flush=True)
    print("".join(lines[-30:]), flush=True)
