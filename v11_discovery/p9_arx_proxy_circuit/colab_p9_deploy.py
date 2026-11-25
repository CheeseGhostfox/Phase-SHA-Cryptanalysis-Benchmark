"""Colab automation helper for P9 (session v11-p9, T4 GPU with --high-mem CPU fallback)."""

from __future__ import annotations

import argparse
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
P9_ROOT = REPO_ROOT / "v11_discovery" / "p9_arx_proxy_circuit"
RESULTS_DIR = P9_ROOT / "results"
BUNDLE_PATH = P9_ROOT / "p9_bundle.zip"
SESSION_NAME = "v11-p9"
REMOTE_OUT_DIR = "/content/p9_results"


def safe_colab_exec(
    session: str,
    script: str,
    timeout: int = 120,
    retries: int = 3,
    label: str = "exec",
) -> str:
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
        print(f"[safe_colab_exec RETRY {attempt}/{retries}] {label}: {last_err}", flush=True)
        if attempt < retries:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"safe_colab_exec({label}) failed: {last_err}")


def build_bundle() -> Path:
    files: List[tuple[Path, str]] = []
    src_dir = REPO_ROOT / "src"
    for name in ("__init__.py", "sha256_ref.py"):
        p = src_dir / name
        if p.exists():
            files.append((p, f"src/{name}"))

    v11_init = REPO_ROOT / "v11_discovery" / "__init__.py"
    if v11_init.exists():
        files.append((v11_init, "v11_discovery/__init__.py"))

    for p in sorted(P9_ROOT.glob("*.py")):
        if p.name == "colab_p9_deploy.py":
            continue
        files.append((p, f"v11_discovery/p9_arx_proxy_circuit/{p.name}"))

    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arcname=arc)
    print(f"[Pack] {BUNDLE_PATH} ({len(files)} files, {BUNDLE_PATH.stat().st_size:,} bytes)", flush=True)
    return BUNDLE_PATH


def create_session(session: str = SESSION_NAME) -> None:
    lst = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if session in (lst.stdout or ""):
        print(f"[Colab] Reusing existing session {session!r}", flush=True)
        return

    print(f"[Colab] Creating session {session!r} with --gpu T4 ...", flush=True)
    res = subprocess.run(
        [str(COLAB_EXE), "new", "-s", session, "--gpu", "T4"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if res.returncode == 0:
        print(res.stdout, flush=True)
        return

    print(f"[Colab] T4 creation failed ({res.stderr.strip()}); falling back to --high-mem CPU ...", flush=True)
    res2 = subprocess.run(
        [str(COLAB_EXE), "new", "-s", session, "--high-mem"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    print(res2.stdout, flush=True)
    if res2.returncode != 0:
        raise RuntimeError(f"Failed to create Colab session: rc={res2.returncode} stderr={res2.stderr}")


def upload_and_setup(session: str = SESSION_NAME) -> None:
    bundle = build_bundle()
    print(f"[Upload] {bundle} -> {session}:/content/p9_bundle.zip", flush=True)
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/p9_bundle.zip"],
        timeout=120,
        check=True,
    )
    setup_script = """
import os, subprocess, sys, zipfile
from pathlib import Path

os.chdir("/content")
Path("/content/p9_results").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/p9_bundle.zip", "r") as zf:
    zf.extractall("/content/")
Path("/content/v11_discovery/__init__.py").touch()

try:
    import z3
    print("z3 already installed:", z3.get_version_string())
except ImportError:
    print("Installing z3-solver ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "z3-solver"], check=True)
    import z3
    print("z3 installed:", z3.get_version_string())

import torch
print("TORCH:", torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("GPU:", subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo NO_GPU"))

# Run unit tests remotely
env = dict(os.environ)
env["PYTHONPATH"] = "/content"
subprocess.run([sys.executable, "-u", "-m", "v11_discovery.p9_arx_proxy_circuit.test_p9_smoke"], env=env, check=True)
"""
    safe_colab_exec(session, setup_script, timeout=180, retries=2, label="upload_and_setup")


def launch_full(session: str = SESSION_NAME) -> None:
    launch_script = f"""
import subprocess, time
from pathlib import Path

# Kill any previous run_p9_experiment process before fresh launch
subprocess.run(["bash", "-lc", "pkill -f run_p9_experiment || true"], check=False)
time.sleep(1.0)

Path({REMOTE_OUT_DIR!r}).mkdir(parents=True, exist_ok=True)
for old_f in ("search_summary.json", "top_circuits.json", "z3_verification.json", "run_log.txt"):
    p = Path({REMOTE_OUT_DIR!r}) / old_f
    if p.exists():
        p.unlink()

cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v11_discovery.p9_arx_proxy_circuit.run_p9_experiment "
    f"--mode full --out_dir {REMOTE_OUT_DIR!r} "
    f"> {REMOTE_OUT_DIR}/run_log.txt 2>&1 < /dev/null &"
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
print(subprocess.getoutput("ps -ef | grep run_p9_experiment | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 {REMOTE_OUT_DIR}/run_log.txt || true"))
"""
    safe_colab_exec(session, launch_script, timeout=90, retries=2, label="launch_full")


def check_status(session: str = SESSION_NAME) -> None:
    status_script = f"""
import json, subprocess
from pathlib import Path

print("--- ALIVE PIDS ---")
print(subprocess.getoutput("ps -ef | grep run_p9_experiment | grep -v grep || true"))
print("--- LOG TAIL ---")
print(subprocess.getoutput("tail -n 25 {REMOTE_OUT_DIR}/run_log.txt || true"))
sp = Path("{REMOTE_OUT_DIR}/search_summary.json")
if sp.exists():
    d = json.loads(sp.read_text())
    print("CHECKPOINT_META:", d.get("meta"))
"""
    safe_colab_exec(session, status_script, timeout=60, retries=2, label="check_status")


def fetch_results(session: str = SESSION_NAME) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("search_summary.json", "top_circuits.json", "z3_verification.json", "run_log.txt"):
        remote_p = f"{REMOTE_OUT_DIR}/{fname}"
        local_p = RESULTS_DIR / fname
        print(f"[Download] {session}:{remote_p} -> {local_p}", flush=True)
        subprocess.run(
            [str(COLAB_EXE), "download", "-s", session, remote_p, str(local_p)],
            timeout=120,
            check=True,
        )
        print(f"  Saved {local_p.name} ({local_p.stat().st_size:,} bytes)", flush=True)


def stop_session(session: str = SESSION_NAME) -> None:
    print(f"[Colab] Stopping session {session!r} ...", flush=True)
    res = subprocess.run(
        [str(COLAB_EXE), "stop", "-s", session],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, flush=True)
    lst = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    print("[Colab Sessions After Stop]", lst.stdout.strip(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["launch", "status", "fetch", "stop"],
        required=True,
    )
    parser.add_argument("--session", default=SESSION_NAME)
    args = parser.parse_args()

    if args.action == "launch":
        create_session(args.session)
        upload_and_setup(args.session)
        launch_full(args.session)
    elif args.action == "status":
        check_status(args.session)
    elif args.action == "fetch":
        fetch_results(args.session)
    elif args.action == "stop":
        stop_session(args.session)


if __name__ == "__main__":
    main()
