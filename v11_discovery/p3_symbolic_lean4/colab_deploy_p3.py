"""Colab deployment, monitoring, sync, and shutdown helper for P3 (v11-p3 CPU high-mem)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List

P3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = P3_ROOT.parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
BUNDLE_PATH = P3_ROOT / "p3_bundle.zip"
SESSION_NAME = "v11-p3"
REMOTE_DIR = "/content/p3_work"


def safe_colab_exec(session: str, script: str, timeout: int = 90, retries: int = 3, label: str = "exec") -> str:
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
            proc.communicate(timeout=5)
            last_err = f"timeout after {timeout}s"
        print(f"[RETRY {attempt}/{retries}] {label}: {last_err}", flush=True)
        if attempt < retries:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"safe_colab_exec({label}) failed: {last_err}")


def create_session(session: str) -> None:
    lst = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    if session in (lst.stdout or ""):
        print(f"[Colab] Session {session!r} already exists.", flush=True)
        return
    print(f"[Colab] Creating high-mem CPU session {session!r}...", flush=True)
    res = subprocess.run(
        [str(COLAB_EXE), "new", "-s", session, "--high-mem"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(f"[Colab] --high-mem failed ({res.stderr.strip()}), falling back to standard CPU...", flush=True)
        res2 = subprocess.run(
            [str(COLAB_EXE), "new", "-s", session],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        print(res2.stdout, flush=True)
        if res2.returncode != 0:
            raise RuntimeError(f"Failed to create Colab session: {res2.stderr}")


def build_bundle() -> Path:
    files: List[Path] = [
        P3_ROOT / "ast_spec.py",
        P3_ROOT / "sha_worlds.py",
        P3_ROOT / "z3_auditor.py",
        P3_ROOT / "neural_mcts_engine.py",
        P3_ROOT / "run_local_smoke.py",
        P3_ROOT / "run_colab_search.py",
    ]
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    print(f"[Pack] {BUNDLE_PATH} ({len(files)} files, {BUNDLE_PATH.stat().st_size:,} bytes)", flush=True)
    return BUNDLE_PATH


def upload_and_setup(session: str) -> None:
    bundle = build_bundle()
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/p3_bundle.zip"],
        timeout=120,
        check=True,
    )
    setup_py = f"""
import os, subprocess, sys, zipfile
from pathlib import Path

Path({REMOTE_DIR!r}).mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/p3_bundle.zip", "r") as zf:
    zf.extractall({REMOTE_DIR!r})

try:
    import z3
    print("z3 already installed:", z3.get_version_string())
except ImportError:
    print("Installing z3-solver...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "z3-solver"], check=True)
    import z3
    print("z3 installed:", z3.get_version_string())

import numpy, torch
print("CPU count:", os.cpu_count())
print("RAM info:", subprocess.getoutput("free -h | head -n 2"))
print("numpy:", numpy.__version__, "torch:", torch.__version__)
"""
    safe_colab_exec(session, setup_py, timeout=180, retries=2, label="setup")


def launch_search(session: str, budget_sec: float, z3_timeout_ms: int) -> None:
    launch_py = f"""
import subprocess, time
from pathlib import Path

out_dir = Path({REMOTE_DIR!r}) / "results"
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "colab_p3.log"
cmd = (
    f"cd {REMOTE_DIR} && "
    f"setsid nohup python3 -u run_colab_search.py "
    f"--out_dir {{str(out_dir)}} --budget_sec {budget_sec} --z3_timeout_ms {z3_timeout_ms} "
    f"> {{str(log_path)}} 2>&1 < /dev/null &"
)
subprocess.Popen(
    ["bash", "-lc", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
time.sleep(3.0)
print("--- PIDS ---")
print(subprocess.getoutput("ps -ef | grep run_colab_search | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput(f"head -n 25 {{str(log_path)}} || true"))
"""
    safe_colab_exec(session, launch_py, timeout=60, retries=3, label="launch")


def check_status(session: str) -> None:
    py = f"""
import json, subprocess
from pathlib import Path

print("--- PIDS ---")
print(subprocess.getoutput("ps -ef | grep run_colab_search | grep -v grep || true"))
res_p = Path({REMOTE_DIR!r}) / "results" / "discovered_identities.json"
if res_p.exists():
    d = json.loads(res_p.read_text(encoding="utf-8"))
    print("--- SUMMARY ---", json.dumps(d.get("run_summary", {{}}), ensure_ascii=False))
    print("--- TOP IDENTITIES ---")
    for it in d.get("identities", [])[:15]:
        za = it.get("z3_audit", {{}})
        print(f"  {{it['id']}} | {{it['track']}} | span={{za.get('v6_active_span')}} | rounds={{za.get('v6_active_rounds')}} | nodes={{it.get('ast_nodes')}} | {{it['title']}}")
else:
    print("discovered_identities.json not created yet")
print("--- LOG TAIL ---")
print(subprocess.getoutput(f"tail -n 20 {REMOTE_DIR}/results/colab_p3.log 2>/dev/null || true"))
"""
    safe_colab_exec(session, py, timeout=45, retries=3, label="status")


def sync_results(session: str, local_results: Path) -> None:
    pack_py = f"""
import zipfile
from pathlib import Path
rdir = Path({REMOTE_DIR!r}) / "results"
out_zip = Path("/tmp/p3_sync.zip")
copied = []
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(rdir.rglob("*")):
        if p.is_file():
            arc = p.relative_to(rdir).as_posix()
            zf.write(p, arcname=arc)
            copied.append(arc)
print("SYNCED:", copied, "bytes=", out_zip.stat().st_size)
"""
    safe_colab_exec(session, pack_py, timeout=60, retries=3, label="pack_sync")
    local_results.mkdir(parents=True, exist_ok=True)
    local_zip = local_results / "p3_sync.zip"
    subprocess.run(
        [str(COLAB_EXE), "download", "-s", session, "/tmp/p3_sync.zip", str(local_zip)],
        timeout=120,
        check=True,
    )
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(local_results)
    local_zip.unlink(missing_ok=True)
    print(f"[Sync] Extracted to {local_results}", flush=True)


def stop_session(session: str) -> None:
    subprocess.run([str(COLAB_EXE), "stop", "-s", session], timeout=60, check=False)
    lst = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    print("[Colab Sessions After Stop]", lst.stdout.strip(), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", default=SESSION_NAME)
    p.add_argument("--create", action="store_true")
    p.add_argument("--setup", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--budget_sec", type=float, default=1800.0)
    p.add_argument("--z3_timeout_ms", type=int, default=15000)
    p.add_argument("--status", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--stop", action="store_true")
    args = p.parse_args()

    if args.create:
        create_session(args.session)
    if args.setup:
        upload_and_setup(args.session)
    if args.launch:
        launch_search(args.session, args.budget_sec, args.z3_timeout_ms)
    if args.status:
        check_status(args.session)
    if args.sync:
        sync_results(args.session, P3_ROOT / "results")
    if args.stop:
        stop_session(args.session)


if __name__ == "__main__":
    main()
