"""
P12 Colab 自动化部署、监控、结果拉取与强制停机脚本（专属会话名 v12-p12）。

安全规程：
1. 严格禁止在本地调用 RTX 4060 GPU（本地只做轻量校验与调度）；
2. 严禁同时挂载未并发读取的 stdin/stdout/stderr=PIPE，所有 colab.exe 调用均通过临时文件重定向；
3. Colab 远程运行使用 python3 -u 无缓冲输出；
4. 拉回全部 results/*.json 与 p12_colab.log 后立刻执行 `colab.exe stop -s v12-p12` 并核实 `colab.exe sessions` 为空。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

P12_DIR = Path(__file__).resolve().parent
REPO_ROOT = P12_DIR.parents[1]
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
SESSION_NAME = "v12-p12"
BUNDLE_PATH = P12_DIR / "p12_bundle.zip"
REMOTE_WORK = "/content/p12_work"

UPLOAD_FILES = (
    "btc_double_sha_engine.py",
    "classic_asicboost_calib.py",
    "gnbc_gpu_search.py",
    "sat_amortized_bounds.py",
    "test_p12_engine.py",
)


def safe_exec(session: str, script: str, timeout: int = 180) -> str:
    """使用 --file 传入临时脚本并用临时文件承载 stdout/stderr，绝不触发 Windows 管道死锁。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fin, \
         tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False, encoding="utf-8") as fout:
        fin.write(script)
        fin_path = fin.name
        fout_path = fout.name
    try:
        with open(fout_path, "w", encoding="utf-8") as f_out:
            res = subprocess.run(
                [str(COLAB_EXE), "exec", "-s", session, "--file", fin_path, "--timeout", str(timeout)],
                stdin=subprocess.DEVNULL,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                timeout=timeout + 20,
            )
        out_text = Path(fout_path).read_text(encoding="utf-8", errors="replace")
        if out_text:
            print(out_text, end="" if out_text.endswith("\n") else "\n", flush=True)
        if res.returncode != 0:
            raise RuntimeError(f"colab exec rc={res.returncode}: {out_text[-600:]}")
        return out_text
    finally:
        for p in (fin_path, fout_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def create_session(session: str = SESSION_NAME) -> str:
    res = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    if session in (res.stdout or ""):
        print(f"[Colab] Session {session!r} already active.", flush=True)
        return "EXISTING"
    for gpu, tout in (("L4", 45), ("T4", 120)):
        print(f"[Colab] Creating {session!r} with --gpu {gpu} (timeout={tout}s) ...", flush=True)
        with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8") as flog:
            flog_path = flog.name
        try:
            with open(flog_path, "w", encoding="utf-8") as fw:
                r = subprocess.run(
                    [str(COLAB_EXE), "new", "-s", session, "--gpu", gpu],
                    stdin=subprocess.DEVNULL,
                    stdout=fw,
                    stderr=subprocess.STDOUT,
                    timeout=tout,
                )
            out_txt = Path(flog_path).read_text(encoding="utf-8", errors="replace")
            print(out_txt, flush=True)
            if r.returncode == 0:
                return gpu
            print(f"[Colab] --gpu {gpu} rc={r.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[Colab] --gpu {gpu} timed out after {tout}s, trying next ...", flush=True)
            subprocess.run([str(COLAB_EXE), "stop", "-s", session], capture_output=True, timeout=20)
        finally:
            try:
                os.unlink(flog_path)
            except OSError:
                pass
    raise RuntimeError("No GPU Colab runtime available (tried L4, T4).")


def upload_and_prepare(session: str = SESSION_NAME) -> None:
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in UPLOAD_FILES:
            zf.write(P12_DIR / fname, arcname=fname)
        zf.write(REPO_ROOT / "src" / "sha256_ref.py", arcname="sha256_ref.py")
    print(f"[Upload] {BUNDLE_PATH} -> {session}:/content/p12_bundle.zip", flush=True)
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(BUNDLE_PATH), "/content/p12_bundle.zip"],
        timeout=120,
        check=True,
    )
    prep_script = f"""
import subprocess, sys, zipfile
from pathlib import Path
Path("{REMOTE_WORK}/results").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/p12_bundle.zip", "r") as zf:
    zf.extractall("{REMOTE_WORK}/")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"))
try:
    import z3, pysat
    print("z3:", z3.get_version_string(), "pysat:", pysat.__version__)
except Exception as e:
    print("Installing z3-solver and python-sat ...", e, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "z3-solver", "python-sat"])
    import z3, pysat
    print("Installed z3:", z3.get_version_string(), "pysat:", pysat.__version__)
"""
    safe_exec(session, prep_script, timeout=240)


def launch_remote(session: str = SESSION_NAME) -> None:
    runner_py = f"""
import json, subprocess, sys, time
from pathlib import Path
Path("{REMOTE_WORK}/results").mkdir(parents=True, exist_ok=True)
runner = '''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "{REMOTE_WORK}")
import torch
from btc_double_sha_engine import enforce_fp32_strict
from classic_asicboost_calib import main as run_a1
from gnbc_gpu_search import run_full_a2
from sat_amortized_bounds import run_full_a3

t_all = time.time()
fp32 = enforce_fp32_strict()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[P12-COLAB] Start on {{gpu_name}} | allow_tf32={{fp32}}", flush=True)

import sha256_ref
from btc_double_sha_engine import make_main_variant, sha256_compress_trace
from classic_asicboost_calib import verify_bit_exactness
w_chk = torch.randint(0, 1 << 31, (128, 16), dtype=torch.int64)
r_ref = sha256_ref.sha256_trace(w_chk, rounds=64)
r_p12 = sha256_compress_trace(sha256_ref.get_iv_tensor("cpu"), w_chk, rounds=64, variant=make_main_variant(), record_gates=False)
assert int((r_ref.digest ^ r_p12.digest).sum().item()) == 0
print("[P12-COLAB] sha256_ref xor_diff == 0 verified.", flush=True)
print("[P12-COLAB] BTC bit_exactness:", verify_bit_exactness(device="cuda" if torch.cuda.is_available() else "cpu", n_random=256), flush=True)

t1 = time.time()
print("[Stage 1/3] Running A1 calibration ...", flush=True)
run_a1()
t_a1 = time.time() - t1

t2 = time.time()
print("[Stage 2/3] Running A2 GPU GNBC search (N_holdout=262144, MAIN + 3 CTRL) ...", flush=True)
run_full_a2(n_holdout=262144, is_smoke=False, out_path=Path("{REMOTE_WORK}/results/a2_gnbc_search.json"))
t_a2 = time.time() - t2

t3 = time.time()
print("[Stage 3/3] Running A3 Z3 + PySAT CaDiCaL195 exact bounds (MAIN + 3 CTRL) ...", flush=True)
run_full_a3(is_smoke=False, out_path=Path("{REMOTE_WORK}/results/a3_sat_bounds.json"))
t_a3 = time.time() - t3

meta = {{
    "colab_session": "{session}",
    "gpu_name": gpu_name,
    "fp32_lock": fp32,
    "stage_seconds": {{"a1": round(t_a1, 2), "a2": round(t_a2, 2), "a3": round(t_a3, 2)}},
    "total_seconds": round(time.time() - t_all, 2),
    "colab_stopped": False,
}}
Path("{REMOTE_WORK}/results/p12_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("[P12-COLAB] ALL STAGES FINISHED.", flush=True)
'''
Path("{REMOTE_WORK}/remote_all.py").write_text(runner, encoding="utf-8")
cmd = "cd {REMOTE_WORK} && setsid nohup python3 -u remote_all.py > {REMOTE_WORK}/results/p12_colab.log 2>&1 < /dev/null &"
subprocess.Popen(["bash", "-lc", cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
time.sleep(3.0)
print(subprocess.getoutput("ps -ef | grep remote_all | grep -v grep || true"))
print(subprocess.getoutput("cat {REMOTE_WORK}/results/p12_colab.log || true"))
"""
    safe_exec(session, runner_py, timeout=90)


def check_status(session: str = SESSION_NAME) -> str:
    st_py = f"""
import subprocess
from pathlib import Path
print("--- ALIVE ---")
print(subprocess.getoutput("ps -ef | grep remote_all | grep -v grep || true"))
print("--- GPU ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true"))
print("--- FILES ---")
for p in sorted(Path("{REMOTE_WORK}/results").glob("*")):
    print(p.name, p.stat().st_size)
print("--- LOG TAIL ---")
print(subprocess.getoutput("tail -n 30 {REMOTE_WORK}/results/p12_colab.log || true"))
"""
    return safe_exec(session, st_py, timeout=60)


def download_and_stop(session: str = SESSION_NAME) -> None:
    res_dir = P12_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "a1_calibration.json",
        "a2_gnbc_search.json",
        "a3_sat_bounds.json",
        "p12_run_meta.json",
        "p12_colab.log",
    ):
        remote = f"{REMOTE_WORK}/results/{fname}"
        local = res_dir / fname
        print(f"[Download] {remote} -> {local}", flush=True)
        subprocess.run(
            [str(COLAB_EXE), "download", "-s", session, remote, str(local)],
            timeout=120,
            check=True,
        )

    print(f"[Colab] Stopping session {session!r} ...", flush=True)
    subprocess.run([str(COLAB_EXE), "stop", "-s", session], timeout=60, check=False)
    sess = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
    print(sess.stdout, flush=True)
    stopped = session not in (sess.stdout or "")

    meta_path = res_dir / "p12_run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["colab_stopped"] = stopped
        meta["sessions_check_stdout"] = (sess.stdout or "").strip()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Colab] Verified colab_stopped={stopped}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("launch", "status", "download_and_stop", "stop"))
    args = ap.parse_args()
    if args.action == "launch":
        create_session(SESSION_NAME)
        upload_and_prepare(SESSION_NAME)
        launch_remote(SESSION_NAME)
    elif args.action == "status":
        check_status(SESSION_NAME)
    elif args.action == "download_and_stop":
        download_and_stop(SESSION_NAME)
    elif args.action == "stop":
        subprocess.run([str(COLAB_EXE), "stop", "-s", SESSION_NAME], timeout=60, check=False)
        res = subprocess.run([str(COLAB_EXE), "sessions"], capture_output=True, text=True, timeout=30)
        print(res.stdout, flush=True)


if __name__ == "__main__":
    main()
