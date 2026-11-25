"""P1 Colab L4/T4 Deployment, Watchdog, Sync, and Clean Shutdown Helper."""

from __future__ import annotations

import argparse
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
P1_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = P1_ROOT / "p1_bundle.zip"
LOCAL_RESULTS = P1_ROOT / "results"


def safe_colab_exec(
    session: str,
    script: str,
    timeout: int = 90,
    retries: int = 3,
    label: str = "exec",
) -> str:
    last_err = ""
    for attempt in range(1, retries + 1):
        t0 = time.time()
        proc = subprocess.Popen(
            [str(COLAB_EXE), "exec", "-s", session, "--timeout", str(timeout)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(input=script, timeout=timeout + 10)
            if proc.returncode == 0:
                if stdout:
                    print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
                return stdout
            last_err = f"rc={proc.returncode} stderr={stderr.strip()[-800:]} stdout={stdout.strip()[-800:]}"
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            last_err = f"timeout after {timeout}s (elapsed {time.time() - t0:.1f}s)"
        print(f"[safe_colab_exec RETRY {attempt}/{retries}] {label}: {last_err}", flush=True)
        if attempt < retries:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"safe_colab_exec({label}) failed: {last_err}")


def build_bundle() -> Path:
    files: List[tuple[Path, str]] = []
    for p in sorted(P1_ROOT.glob("*.py")):
        if p.name == "colab_deploy_p1.py":
            continue
        files.append((p, f"v11_discovery/p1_infonce/{p.name}"))
    # Ensure v11_discovery/__init__.py exists in archive for module imports
    spec_p = P1_ROOT / "SPEC.md"
    if spec_p.exists():
        files.append((spec_p, "v11_discovery/p1_infonce/SPEC.md"))

    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v11_discovery/__init__.py", "")
        for src, arc in files:
            zf.write(src, arcname=arc)
    print(f"[Pack] {BUNDLE_PATH} ({len(files)} files, {BUNDLE_PATH.stat().st_size:,} bytes)", flush=True)
    return BUNDLE_PATH


def create_session(session: str, gpu: str = "L4") -> None:
    lst = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    if session in (lst.stdout or ""):
        print(f"[Colab] Session {session!r} already exists.", flush=True)
        return

    gpus_to_try = [gpu] if gpu != "L4" else ["L4", "T4"]
    last_err = ""
    for g in gpus_to_try:
        print(f"[Colab] Creating session {session!r} with --gpu {g} ...", flush=True)
        try:
            res = subprocess.run(
                [str(COLAB_EXE), "new", "-s", session, "--gpu", g],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
            )
            print(res.stdout, flush=True)
            if res.stderr:
                print(res.stderr, flush=True)
            if res.returncode == 0:
                return
            last_err = f"gpu={g} rc={res.returncode} stderr={res.stderr}"
        except subprocess.TimeoutExpired:
            last_err = f"gpu={g} timed out after 90s"
            chk = subprocess.run(
                [str(COLAB_EXE), "sessions"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            if session in (chk.stdout or ""):
                print(f"[Colab] Session {session!r} appeared after timeout; using it.", flush=True)
                return
        print(f"[Colab] Failed with --gpu {g}: {last_err}", flush=True)
    raise RuntimeError(f"Failed to create Colab session {session!r}: {last_err}")


def upload_and_verify(session: str) -> None:
    bundle = build_bundle()
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/p1_bundle.zip"],
        timeout=120, check=True,
    )
    verify_script = """
import os, subprocess, sys, zipfile
from pathlib import Path

os.chdir("/content")
Path("/content/p1_runs").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/p1_bundle.zip", "r") as zf:
    zf.extractall("/content/")

print("--- GPU ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"))
import torch
print("TORCH:", torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
sys.path.insert(0, "/content")
subprocess.run(["python3", "-u", "-m", "v11_discovery.p1_infonce.test_p1_cpu"], check=True)
"""
    safe_colab_exec(session, verify_script, timeout=120, retries=2, label="upload_and_verify")


def launch_sweep(session: str, extra_args: str = "") -> None:
    launch_script = f"""
import subprocess, time
from pathlib import Path

out_dir = Path("/content/p1_runs/results")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = Path("/content/p1_runs/p1_sweep.log")

cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v11_discovery.p1_infonce.run_p1_sweep "
    "--out_dir /content/p1_runs/results {extra_args} "
    "> /content/p1_runs/p1_sweep.log 2>&1 < /dev/null &"
)
subprocess.Popen(
    ["bash", "-lc", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
time.sleep(3.5)
print("--- ALIVE PIDS ---")
print(subprocess.getoutput("ps -ef | grep run_p1_sweep | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/p1_runs/p1_sweep.log || true"))
"""
    safe_colab_exec(session, launch_script, timeout=60, retries=3, label="launch_sweep")


def check_status(session: str) -> None:
    status_script = """
import json, subprocess
from pathlib import Path

print("--- GPU ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"))
print("--- PIDS ---")
print(subprocess.getoutput("ps -ef | grep run_p1_sweep | grep -v grep || true"))
sum_p = Path("/content/p1_runs/results/p1_sweep_summary.json")
if sum_p.exists():
    d = json.loads(sum_p.read_text(encoding="utf-8"))
    print("COMPLETED ROUNDS:", list(d.get("rounds", {}).keys()), "wall=", d.get("meta", {}).get("total_wall_sec"))
else:
    print("SUMMARY: <not yet written>")
print("--- LOG TAIL ---")
print(subprocess.getoutput("tail -n 25 /content/p1_runs/p1_sweep.log 2>/dev/null || true"))
"""
    safe_colab_exec(session, status_script, timeout=45, retries=3, label="status")


def sync_results(session: str) -> None:
    pack_script = """
import zipfile
from pathlib import Path

rdir = Path("/content/p1_runs/results")
out_zip = Path("/tmp/p1_sync.zip")
copied = []
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    if rdir.exists():
        for p in sorted(rdir.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".pt", ".log"):
                arc = p.relative_to(rdir).as_posix()
                zf.write(p, arcname=arc)
                copied.append(arc)
    log_p = Path("/content/p1_runs/p1_sweep.log")
    if log_p.exists():
        zf.write(log_p, arcname="p1_sweep.log")
        copied.append("p1_sweep.log")
print("SYNCED FILES:", copied, "zip_bytes=", out_zip.stat().st_size)
"""
    safe_colab_exec(session, pack_script, timeout=60, retries=3, label="pack_sync")
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    local_zip = LOCAL_RESULTS / "p1_sync.zip"
    subprocess.run(
        [str(COLAB_EXE), "download", "-s", session, "/tmp/p1_sync.zip", str(local_zip)],
        timeout=180, check=True,
    )
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(LOCAL_RESULTS)
    local_zip.unlink(missing_ok=True)
    print(f"[Sync] Extracted to {LOCAL_RESULTS}", flush=True)


def stop_session(session: str) -> None:
    subprocess.run([str(COLAB_EXE), "stop", "-s", session], timeout=60, check=True)
    res = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    print(res.stdout, flush=True)
    if session in (res.stdout or ""):
        raise RuntimeError(f"Session {session!r} still listed after stop!")
    print(f"[Colab] Confirmed session {session!r} is stopped.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="P1 Colab Deploy & Watchdog")
    p.add_argument("--session", type=str, default="v11-p1")
    p.add_argument("--gpu", type=str, default="L4")
    p.add_argument("--extra_args", type=str, default="")
    p.add_argument("--create", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--stop", action="store_true")
    args = p.parse_args()

    if args.create:
        create_session(args.session, gpu=args.gpu)
    if args.upload:
        upload_and_verify(args.session)
    if args.launch:
        launch_sweep(args.session, args.extra_args)
    if args.status:
        check_status(args.session)
    if args.sync:
        sync_results(args.session)
    if args.stop:
        stop_session(args.session)


if __name__ == "__main__":
    main()
