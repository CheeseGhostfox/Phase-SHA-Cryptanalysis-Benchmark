"""Independent Colab deployment & watchdog helper for P10 (session: v11-p10)."""

from __future__ import annotations

import argparse
import subprocess
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
P10_DIR = REPO_ROOT / "v11_discovery" / "p10_ugly_algebra"
BUNDLE_PATH = P10_DIR / "p10_bundle.zip"
SESSION_NAME = "v11-p10"
REMOTE_OUT = "/content/p10_runs"


def safe_exec(session: str, script: str, timeout: int = 90, retries: int = 3) -> str:
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
            out, err = proc.communicate(input=script, timeout=timeout + 10)
            if proc.returncode == 0:
                print(out, end="" if out.endswith("\n") else "\n", flush=True)
                return out
            last_err = f"rc={proc.returncode} err={err.strip()[-500:]} out={out.strip()[-500:]}"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)
            last_err = f"timeout after {timeout}s"
        print(f"[RETRY {attempt}/{retries}] {last_err}", flush=True)
        time.sleep(2 * attempt)
    raise RuntimeError(last_err)


def build_bundle() -> Path:
    files = []
    src_ref = REPO_ROOT / "src" / "sha256_ref.py"
    src_init = REPO_ROOT / "src" / "__init__.py"
    v11_init = REPO_ROOT / "v11_discovery" / "__init__.py"
    if src_init.exists():
        files.append((src_init, "src/__init__.py"))
    files.append((src_ref, "src/sha256_ref.py"))
    if v11_init.exists():
        files.append((v11_init, "v11_discovery/__init__.py"))
    for p in sorted(P10_DIR.glob("*.py")):
        files.append((p, f"v11_discovery/p10_ugly_algebra/{p.name}"))
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arcname=arc)
    print(f"[Pack] {BUNDLE_PATH} ({len(files)} files, {BUNDLE_PATH.stat().st_size:,} bytes)", flush=True)
    return BUNDLE_PATH


def ensure_session(session: str, prefer_gpu: str = "L4") -> None:
    lst = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    if session in (lst.stdout or ""):
        print(f"[Colab] Reusing existing session {session}", flush=True)
        return
    gpus = [prefer_gpu, "T4"] if prefer_gpu != "T4" else ["T4"]
    for g in gpus:
        print(f"[Colab] Creating {session} (--gpu {g}) ...", flush=True)
        res = subprocess.run(
            [str(COLAB_EXE), "new", "-s", session, "--gpu", g],
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(res.stdout, flush=True)
        if res.returncode == 0:
            return
        print(f"[Colab] {g} failed: {res.stderr.strip()[-300:]}", flush=True)
    raise RuntimeError(f"Failed to create Colab session {session}")


def launch(session: str, prefer_gpu: str, extra_args: str) -> None:
    ensure_session(session, prefer_gpu=prefer_gpu)
    bundle = build_bundle()
    subprocess.run([str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/p10_bundle.zip"], timeout=120, check=True)
    setup_and_run = f"""
import os, subprocess, zipfile
from pathlib import Path

os.chdir('/content')
with zipfile.ZipFile('/content/p10_bundle.zip', 'r') as zf:
    zf.extractall('/content/')
Path({REMOTE_OUT!r}).mkdir(parents=True, exist_ok=True)
print('GPU:', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'))
cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v11_discovery.p10_ugly_algebra.run_p10_experiment "
    f"--out_dir {REMOTE_OUT!r} {extra_args} "
    "> /content/p10_runs/p10.log 2>&1 < /dev/null &"
)
subprocess.Popen(['bash', '-lc', cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
import time; time.sleep(3.0)
print('PIDS:', subprocess.getoutput("ps -ef | grep run_p10_experiment | grep -v grep || true"))
print('LOG_HEAD:', subprocess.getoutput("head -n 20 /content/p10_runs/p10.log || true"))
"""
    safe_exec(session, setup_and_run, timeout=90)


def status(session: str) -> None:
    py = f"""
import json, subprocess
from pathlib import Path
print('GPU:', subprocess.getoutput('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader'))
print('PIDS:', subprocess.getoutput("ps -ef | grep run_p10_experiment | grep -v grep || true"))
rp = Path({REMOTE_OUT!r}) / 'p10_results.json'
if rp.exists():
    d = json.loads(rp.read_text(encoding='utf-8'))
    print('STAGE:', d.get('status'), 'A:', len(d.get('part_a_scaling', [])), 'B:', len(d.get('part_b_32bit', [])), 'C:', len(d.get('part_c_sha256_round', [])))
else:
    print('STAGE: <no json yet>')
print('LOG_TAIL:')
print(subprocess.getoutput('tail -n 25 /content/p10_runs/p10.log || true'))
"""
    safe_exec(session, py, timeout=60)


def sync(session: str, local_dir: Path) -> None:
    pack = f"""
import zipfile
from pathlib import Path
r = Path({REMOTE_OUT!r})
zp = Path('/tmp/p10_sync.zip')
files = []
with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(r.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.log'):
            zf.write(p, arcname=p.relative_to(r).as_posix())
            files.append(p.name)
print('SYNC_FILES:', files, 'bytes:', zp.stat().st_size)
"""
    safe_exec(session, pack, timeout=60)
    local_dir.mkdir(parents=True, exist_ok=True)
    lzip = local_dir / "p10_sync.zip"
    subprocess.run([str(COLAB_EXE), "download", "-s", session, "/tmp/p10_sync.zip", str(lzip)], timeout=120, check=True)
    with zipfile.ZipFile(lzip, "r") as zf:
        zf.extractall(local_dir)
    lzip.unlink(missing_ok=True)
    print(f"[Sync] Extracted to {local_dir}", flush=True)


def stop(session: str) -> None:
    subprocess.run([str(COLAB_EXE), "stop", "-s", session], check=False, timeout=45)
    lst = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    print(lst.stdout, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", default=SESSION_NAME)
    p.add_argument("--gpu", default="L4")
    p.add_argument("--extra_args", default="")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--stop", action="store_true")
    args = p.parse_args()
    if args.launch:
        launch(args.session, args.gpu, args.extra_args)
    if args.status:
        status(args.session)
    if args.sync:
        sync(args.session, P10_DIR / "results")
    if args.stop:
        stop(args.session)


if __name__ == "__main__":
    main()
