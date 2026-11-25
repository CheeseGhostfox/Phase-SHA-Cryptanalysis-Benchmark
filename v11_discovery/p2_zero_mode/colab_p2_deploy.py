"""P2 Colab L4/T4 自动化部署、监控、全量回传与停机工具（会话名 v11-p2）。"""

from __future__ import annotations

import argparse
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
P2_ROOT = Path(__file__).resolve().parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
BUNDLE_PATH = P2_ROOT / "p2_bundle.zip"
LOCAL_RESULTS = P2_ROOT / "results"
DEFAULT_SESSION = "v11-p2"


def safe_colab_exec(
    session: str,
    script: str,
    timeout: int = 90,
    retries: int = 3,
    label: str = "exec",
) -> str:
    """在 Colab 会话内通过 stdin 执行 Python 代码，带超时强杀与指数退避重试。"""
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
    """只读打包 v10_ouroboros64 核心/网络源码与 v11_discovery/p2_zero_mode 实验源码。"""
    files_to_pack: List[Tuple[Path, str]] = []
    v10_root = REPO_ROOT / "v10_ouroboros64"

    for init_p in [v10_root / "__init__.py"]:
        if init_p.exists():
            files_to_pack.append((init_p, init_p.relative_to(REPO_ROOT).as_posix()))

    for sub in ("ouroboros_core", "ouroboros_net"):
        for p in sorted((v10_root / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files_to_pack.append((p, p.relative_to(REPO_ROOT).as_posix()))

    for p in sorted(P2_ROOT.rglob("*.py")):
        if "__pycache__" in p.parts or p.name == BUNDLE_PATH.name:
            continue
        files_to_pack.append((p, p.relative_to(REPO_ROOT).as_posix()))

    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_pack:
            zf.write(src, arcname=arc)

    print(
        f"[Pack] {BUNDLE_PATH} ({len(files_to_pack)} files, {BUNDLE_PATH.stat().st_size:,} bytes)",
        flush=True,
    )
    return BUNDLE_PATH


def ensure_session(session: str) -> None:
    """创建 Colab 会话 v11-p2（优先 L4，不可用时自动退回 T4）。"""
    lst = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if session in (lst.stdout or ""):
        print(f"[Colab] 会话 {session!r} 已存在，直接复用。", flush=True)
        return

    for gpu in ("L4", "T4"):
        print(f"[Colab] 尝试创建 {gpu} 会话 {session!r} ...", flush=True)
        try:
            res = subprocess.run(
                [str(COLAB_EXE), "new", "-s", session, "--gpu", gpu],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90 if gpu == "L4" else 150,
            )
            print(res.stdout, flush=True)
            if res.returncode == 0:
                return
            print(f"[Colab] {gpu} 开卡失败 (rc={res.returncode}): {res.stderr.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[Colab] {gpu} 开卡超时，检查是否已创建或回退下一档 GPU...", flush=True)
            chk = subprocess.run(
                [str(COLAB_EXE), "sessions"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if session in (chk.stdout or ""):
                print(f"[Colab] 会话 {session!r} 已在服务端就绪，直接复用。", flush=True)
                return
    raise RuntimeError(f"无法创建 {session!r} (L4/T4 均不可用)")


def cmd_launch(session: str) -> None:
    ensure_session(session)
    bundle = build_bundle()
    print(f"[Upload] {bundle} -> {session}:/content/p2_bundle.zip ...", flush=True)
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/p2_bundle.zip"],
        timeout=120,
        check=True,
    )

    setup_and_launch = """
import os, subprocess, sys, time, zipfile
from pathlib import Path

os.chdir("/content")
Path("/content/p2_results").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/p2_bundle.zip", "r") as zf:
    zf.extractall("/content/")

print("--- GPU & TORCH ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"))
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 杀掉任何残留旧 p2 进程
subprocess.run("pkill -f run_p2_sweep || true", shell=True)
time.sleep(0.5)

cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v11_discovery.p2_zero_mode.run_p2_sweep "
    "--mode full --train_steps 160 --batch_train 20 --k_train 20 "
    "--out_dir /content/p2_results "
    "> /content/p2_results/colab_run.log 2>&1 < /dev/null &"
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
print(subprocess.getoutput("ps -ef | grep run_p2_sweep | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/p2_results/colab_run.log || true"))
"""
    safe_colab_exec(session, setup_and_launch, timeout=120, retries=2, label="launch_p2")


def cmd_status(session: str) -> None:
    status_py = """
import subprocess
from pathlib import Path

pids = subprocess.getoutput("ps -ef | grep run_p2_sweep | grep -v grep || true").strip()
print("ALIVE_PIDS:", pids if pids else "<NONE>")
print("GPU:", subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true").strip())
for fname in ("sweep_matrix.json", "wavepacket_structures.json", "sweep_matrix_partial.json", "colab_run.log"):
    p = Path("/content/p2_results") / fname
    print(f"FILE {fname}: exists={p.exists()} size={p.stat().st_size if p.exists() else 0}")
print("--- LOG TAIL ---")
print(subprocess.getoutput("tail -n 35 /content/p2_results/colab_run.log || true"))
"""
    safe_colab_exec(session, status_py, timeout=60, retries=3, label="status_p2")


def cmd_download(session: str) -> None:
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    files = ("sweep_matrix.json", "wavepacket_structures.json", "colab_run.log")
    for fname in files:
        remote = f"/content/p2_results/{fname}"
        local = LOCAL_RESULTS / fname
        print(f"[Download] {session}:{remote} -> {local}", flush=True)
        subprocess.run(
            [str(COLAB_EXE), "download", "-s", session, remote, str(local)],
            timeout=180,
            check=True,
        )
        print(f"  saved {local} ({local.stat().st_size:,} bytes)", flush=True)


def cmd_stop(session: str) -> None:
    print(f"[Stop] 停止会话 {session!r} ...", flush=True)
    res = subprocess.run(
        [str(COLAB_EXE), "stop", "-s", session],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(res.stderr, flush=True)
    sess = subprocess.run(
        [str(COLAB_EXE), "sessions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    print("[Sessions]", sess.stdout.strip(), flush=True)
    with open(LOCAL_RESULTS / "colab_run.log", "a", encoding="utf-8") as f:
        f.write("\n=== LOCAL COLAB STOP VERIFICATION ===\n")
        f.write(sess.stdout.strip() + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="P2 Colab Deployer")
    p.add_argument("action", choices=["launch", "status", "download", "stop"])
    p.add_argument("--session", default=DEFAULT_SESSION)
    args = p.parse_args()
    if args.action == "launch":
        cmd_launch(args.session)
    elif args.action == "status":
        cmd_status(args.session)
    elif args.action == "download":
        cmd_download(args.session)
    elif args.action == "stop":
        cmd_stop(args.session)


if __name__ == "__main__":
    main()
