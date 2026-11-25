"""
P5 Colab 自动化部署、后台启动、状态监控、结果同步与安全停机工具。
会话名固定 v11-p5（优先 --gpu L4，不可用退 --gpu T4）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

P5_DIR = Path(__file__).resolve().parent
REPO_ROOT = P5_DIR.parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
SESSION_NAME = "v11-p5"
BUNDLE_PATH = P5_DIR / "p5_bundle.zip"


def safe_exec(session: str, script: str, timeout: int = 120, retries: int = 3) -> str:
    last_err = ""
    for attempt in range(1, retries + 1):
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
            stdout, stderr = proc.communicate(input=script, timeout=timeout + 15)
            if proc.returncode == 0:
                if stdout:
                    print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
                return stdout
            last_err = f"rc={proc.returncode} stderr={stderr.strip()[-600:]} stdout={stdout.strip()[-600:]}"
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            last_err = f"timeout after {timeout}s"
        print(f"[safe_exec retry {attempt}/{retries}] {last_err}", flush=True)
        if attempt < retries:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"safe_exec failed: {last_err}")


def create_session(session: str = SESSION_NAME) -> str:
    lst = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if session in (lst.stdout or ""):
        print(f"[Colab] Session {session!r} already exists.", flush=True)
        return "EXISTING"

    for gpu in ("L4", "T4"):
        print(f"[Colab] Creating session {session!r} with --gpu {gpu} ...", flush=True)
        res = subprocess.run(
            [str(COLAB_EXE), "new", "-s", session, "--gpu", gpu],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        print(res.stdout, flush=True)
        if res.returncode == 0:
            return gpu
        print(f"[Colab] --gpu {gpu} unavailable: {res.stderr.strip()}", flush=True)
    raise RuntimeError("Failed to create Colab session with L4 or T4.")


def build_and_upload(session: str = SESSION_NAME) -> None:
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("btc_weak_midstate.py", "test_btc_weak_midstate.py"):
            zf.write(P5_DIR / name, arcname=name)
    print(f"[Upload] {BUNDLE_PATH} -> {session}:/content/p5_bundle.zip", flush=True)
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(BUNDLE_PATH), "/content/p5_bundle.zip"],
        timeout=120,
        check=True,
    )
    setup_py = """
import os, subprocess, zipfile
from pathlib import Path
Path('/content/p5_work/results').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile('/content/p5_bundle.zip', 'r') as zf:
    zf.extractall('/content/p5_work/')
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'))
import torch
print('TORCH:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"""
    safe_exec(session, setup_py, timeout=120)


def launch_remote(session: str = SESSION_NAME, extra_args: str = "") -> None:
    launch_py = f"""
import subprocess, time
cmd = (
    'cd /content/p5_work && '
    'setsid nohup python3 -u btc_weak_midstate.py --out_dir /content/p5_work/results {extra_args} '
    '> /content/p5_work/p5_run.log 2>&1 < /dev/null &'
)
subprocess.Popen(
    ['bash', '-lc', cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
time.sleep(3.0)
print('--- ALIVE PIDS ---')
print(subprocess.getoutput('ps -ef | grep btc_weak_midstate | grep -v grep || true'))
print('--- LOG HEAD ---')
print(subprocess.getoutput('cat /content/p5_work/p5_run.log || true'))
"""
    safe_exec(session, launch_py, timeout=60)


def check_status(session: str = SESSION_NAME) -> None:
    status_py = """
import subprocess
from pathlib import Path
print('--- ALIVE PIDS ---')
print(subprocess.getoutput('ps -ef | grep btc_weak_midstate | grep -v grep || true'))
print('--- GPU ---')
print(subprocess.getoutput('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true'))
print('--- FILES ---')
for p in sorted(Path('/content/p5_work/results').glob('*')):
    print(p.name, p.stat().st_size)
print('--- LOG TAIL ---')
print(subprocess.getoutput('tail -n 30 /content/p5_work/p5_run.log || true'))
"""
    safe_exec(session, status_py, timeout=60)


def download_results(session: str = SESSION_NAME) -> None:
    res_dir = P5_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("p5_sweep_summary.json", "p5_run_meta.json"):
        remote = f"/content/p5_work/results/{fname}"
        local = res_dir / fname
        print(f"[Download] {remote} -> {local}", flush=True)
        subprocess.run(
            [str(COLAB_EXE), "download", "-s", session, remote, str(local)],
            timeout=120,
            check=True,
        )
    # 下载日志
    subprocess.run(
        [str(COLAB_EXE), "download", "-s", session, "/content/p5_work/p5_run.log", str(res_dir / "p5_run.log")],
        timeout=120,
        check=False,
    )


def stop_session(session: str = SESSION_NAME) -> None:
    print(f"[Colab] Stopping session {session!r} ...", flush=True)
    subprocess.run([str(COLAB_EXE), "stop", "-s", session], timeout=60, check=False)
    res = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    print(res.stdout, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("launch", "status", "download", "stop"))
    ap.add_argument("--extra_args", type=str, default="")
    args = ap.parse_args()
    if args.action == "launch":
        create_session(SESSION_NAME)
        build_and_upload(SESSION_NAME)
        launch_remote(SESSION_NAME, args.extra_args)
    elif args.action == "status":
        check_status(SESSION_NAME)
    elif args.action == "download":
        download_results(SESSION_NAME)
    elif args.action == "stop":
        stop_session(SESSION_NAME)


if __name__ == "__main__":
    main()
