"""
Phase-SHA V10「衔尾蛇-64」Colab A100-40GB 自动化打包、部署、持续监督与权重同步工具。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
V10_ROOT = REPO_ROOT / "v10_ouroboros64"
BUNDLE_PATH = V10_ROOT / "v10_bundle.zip"


def safe_colab_exec(
    session: str,
    script: str,
    timeout: int = 90,
    retries: int = 3,
    label: str = "exec",
) -> str:
    """在 Colab 会话内通过 stdin 执行 Python 脚本，附带硬超时强杀与指数退避重试。"""
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
            last_err = (
                f"rc={proc.returncode} stderr={stderr.strip()[-900:]} "
                f"stdout={stdout.strip()[-900:]}"
            )
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
    raise RuntimeError(f"safe_colab_exec({label}) 失败（重试 {retries} 次）: {last_err}")


def build_bundle() -> Path:
    """打包 V10 全部源码、C++17 源码与 V10 单测（排除编译产物、缓存与权重目录）。"""
    files_to_pack: List[tuple[Path, str]] = []
    for p in sorted(V10_ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        if (
            "/runs/" in rel
            or "/__pycache__/" in rel
            or rel.endswith((".pyc", ".dll", ".so", ".pyd", ".pt", ".zip"))
        ):
            continue
        if p.suffix in (".py", ".cpp", ".hpp", ".h", ".md"):
            files_to_pack.append((p, rel))

    tests_dir = REPO_ROOT / "tests"
    for tf in sorted(tests_dir.glob("test_v10_*.py")):
        files_to_pack.append((tf, f"tests/{tf.name}"))
    pytest_ini = REPO_ROOT / "pytest.ini"
    if pytest_ini.exists():
        files_to_pack.append((pytest_ini, "pytest.ini"))

    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_pack:
            zf.write(src, arcname=arc)
    print(f"[Pack] 生成部署包 {BUNDLE_PATH} ({len(files_to_pack)} 文件, {BUNDLE_PATH.stat().st_size:,} 字节)", flush=True)
    return BUNDLE_PATH


def create_session(session: str, gpu: str = "A100") -> None:
    """若目标 Colab 会话不存在则创建之（A100）。"""
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
    print(f"[Colab] 正在创建 {gpu} 实例 {session!r} ...", flush=True)
    res = subprocess.run(
        [str(COLAB_EXE), "new", "-s", session, "--gpu", gpu],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    print(res.stdout, flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"创建 Colab 实例失败: rc={res.returncode} stderr={res.stderr}")


def upload_and_verify(session: str, run_pytest: bool = True) -> None:
    """上传最新 V10 部署包，远端安装依赖、编译 _ouroboros_capi.so 并跑单测验真。"""
    bundle = build_bundle()
    print(f"[Upload] 上传 {bundle} -> {session}:/content/v10_bundle.zip ...", flush=True)
    subprocess.run(
        [str(COLAB_EXE), "upload", "-s", session, str(bundle), "/content/v10_bundle.zip"],
        timeout=120,
        check=True,
    )

    test_cmd = (
        'subprocess.run(["python3", "-u", "-m", "pytest", '
        '"tests/test_v10_milestone1.py", "tests/test_v10_milestone2_smoke.py", "-v"], check=True)'
        if run_pytest
        else 'print("SKIP_PYTEST")'
    )
    setup_script = f"""
import os, subprocess, sys, zipfile
from pathlib import Path

os.chdir("/content")
Path("/content/v10_runs").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/v10_bundle.zip", "r") as zf:
    zf.extractall("/content/")

print("--- 检查并安装 python-sat 与 pycryptosat ---")
try:
    import pysat  # noqa: F401
    print("pysat already installed:", pysat.__file__)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-sat"], check=True)
try:
    import pycryptosat  # noqa: F401
    print("pycryptosat already installed:", getattr(pycryptosat, "__version__", pycryptosat.__file__))
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycryptosat"], check=True)
    import pycryptosat  # noqa: F401
    print("pycryptosat installed:", getattr(pycryptosat, "__version__", pycryptosat.__file__))

print("--- GPU 与编译器信息 ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"))
print(subprocess.getoutput("g++ --version | head -n 1"))

sys.path.insert(0, "/content")
from v10_ouroboros64.ouroboros_core.build_cpp import build_capi
so_path = build_capi(force=True)
print("BUILT SO:", so_path, so_path.stat().st_size, "bytes")

import torch
print("TORCH:", torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
{test_cmd}
"""
    safe_colab_exec(session, setup_script, timeout=180, retries=2, label="upload_and_verify")


def launch_training(session: str, remote_dir: str, extra_args: str) -> None:
    """后台启动 Milestone 2 长训并确认存活。"""
    launch_script = f"""
import subprocess, time
from pathlib import Path

run_dir = Path({remote_dir!r})
run_dir.mkdir(parents=True, exist_ok=True)
log_path = Path("/content/v10_runs/train_v10.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v10_ouroboros64.ouroboros_train.trainer "
    f"--out_dir {remote_dir!r} {extra_args} "
    "> /content/v10_runs/train_v10.log 2>&1 < /dev/null &"
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
print(subprocess.getoutput("ps -ef | grep ouroboros | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/v10_runs/train_v10.log || true"))
"""
    safe_colab_exec(session, launch_script, timeout=60, retries=3, label="launch_training")


def launch_m3(session: str, remote_dir: str, extra_args: str) -> None:
    """在同一热态 A100 实例上后台启动 Milestone 3 虫洞 MCTS 与前沿基准评测。"""
    script = f"""
import subprocess, time
from pathlib import Path

Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v10_ouroboros64.ouroboros_bench.m3_frontier_bench "
    f"--out_dir {remote_dir!r} {extra_args} "
    "> /content/v10_runs/m3_bench.log 2>&1 < /dev/null &"
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
print("--- ALIVE PIDS ---")
print(subprocess.getoutput("ps -ef | grep m3_frontier_bench | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/v10_runs/m3_bench.log || true"))
"""
    safe_colab_exec(session, script, timeout=60, retries=3, label="launch_m3")


def launch_m3_train(session: str, remote_dir: str, extra_args: str) -> None:
    """在同一热态 GPU 实例上后台启动 Milestone 3 在线 GRPO + MCTS 自对弈训练。"""
    script = f"""
import subprocess, time
from pathlib import Path

Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v10_ouroboros64.ouroboros_train.m3_trainer "
    f"--out_dir {remote_dir!r} {extra_args} "
    "> /content/v10_runs/m3_train.log 2>&1 < /dev/null &"
)
subprocess.Popen(
    ["bash", "-lc", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
time.sleep(4.0)
print("--- ALIVE PIDS ---")
print(subprocess.getoutput("ps -ef | grep m3_trainer | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/v10_runs/m3_train.log || true"))
"""
    safe_colab_exec(session, script, timeout=60, retries=3, label="launch_m3_train")


def upload_checkpoint(session: str, local_ckpt: Path, remote_ckpt: str) -> None:
    """将本地检查点文件完整上传至 Colab（大文件自动分片断点重传），合并后强校验字节数与 SHA-256。"""
    import hashlib
    import tempfile

    if not local_ckpt.exists():
        raise FileNotFoundError(f"本地检查点不存在: {local_ckpt}")
    raw = local_ckpt.read_bytes()
    expected_bytes = len(raw)
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    safe_colab_exec(
        session,
        f"from pathlib import Path; Path({remote_ckpt!r}).parent.mkdir(parents=True, exist_ok=True)",
        timeout=45,
        retries=3,
        label="mkdir_ckpt",
    )
    print(f"[Upload Ckpt] {local_ckpt} ({expected_bytes:,} bytes, sha256={expected_sha256[:12]}) -> {session}:{remote_ckpt}", flush=True)

    chunk_size = 10 * 1024 * 1024  # 10 MB 分片，彻底规避单连接 TLS 10053 中断
    if expected_bytes <= chunk_size:
        subprocess.run(
            [str(COLAB_EXE), "upload", "-s", session, str(local_ckpt), remote_ckpt],
            timeout=180,
            check=True,
        )
    else:
        n_parts = (expected_bytes + chunk_size - 1) // chunk_size
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            safe_colab_exec(
                session,
                "from pathlib import Path; Path('/tmp/v10_ckpt_parts').mkdir(parents=True, exist_ok=True)",
                timeout=45,
                retries=3,
                label="mkdir_parts",
            )
            for idx in range(n_parts):
                part_bytes = raw[idx * chunk_size : (idx + 1) * chunk_size]
                local_part = td_path / f"part_{idx:03d}.bin"
                local_part.write_bytes(part_bytes)
                remote_part = f"/tmp/v10_ckpt_parts/part_{idx:03d}.bin"
                last_exc: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        subprocess.run(
                            [str(COLAB_EXE), "upload", "-s", session, str(local_part), remote_part],
                            timeout=120,
                            check=True,
                        )
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        print(f"[Upload Part {idx+1}/{n_parts} RETRY {attempt}/3] {exc}", flush=True)
                        time.sleep(2.0 * attempt)
                if last_exc is not None:
                    raise last_exc
                print(f"[Upload Part {idx+1}/{n_parts}] OK ({len(part_bytes):,} bytes)", flush=True)
        merge_py = f"""
import hashlib, shutil
from pathlib import Path
out_p = Path({remote_ckpt!r})
parts = sorted(Path('/tmp/v10_ckpt_parts').glob('part_*.bin'))
assert len(parts) == {n_parts}, f'expected {n_parts} parts, got {{len(parts)}}'
with open(out_p, 'wb') as f:
    for pt in parts:
        f.write(pt.read_bytes())
shutil.rmtree('/tmp/v10_ckpt_parts', ignore_errors=True)
data = out_p.read_bytes()
print('REMOTE_CKPT_BYTES=', len(data))
assert len(data) == {expected_bytes}, f'size mismatch {{len(data)}} != {expected_bytes}'
sha = hashlib.sha256(data).hexdigest()
print('REMOTE_CKPT_SHA256=', sha[:12])
assert sha == {expected_sha256!r}, 'sha256 mismatch'
"""
        safe_colab_exec(session, merge_py, timeout=90, retries=3, label="merge_verify_ckpt")
        return

    verify_py = (
        f"from pathlib import Path; p=Path({remote_ckpt!r}); "
        f"sz=p.stat().st_size; print('REMOTE_CKPT_BYTES=', sz); "
        f"assert sz == {expected_bytes}, f'size mismatch {{sz}} != {expected_bytes}'"
    )
    safe_colab_exec(session, verify_py, timeout=45, retries=3, label="verify_ckpt")


def launch_m3_v2_train(session: str, remote_dir: str, extra_args: str) -> None:
    """在热态 GPU 实例后台启动 Milestone 3-V2 搜中动态 GRPO + MCTS 训练。"""
    script = f"""
import subprocess, time
from pathlib import Path

Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v10_ouroboros64.ouroboros_train.m3_v2_trainer "
    f"--out_dir {remote_dir!r} {extra_args} "
    "> /content/v10_runs/m3v2_train.log 2>&1 < /dev/null &"
)
subprocess.Popen(
    ["bash", "-lc", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
time.sleep(4.0)
print("--- ALIVE PIDS ---")
print(subprocess.getoutput("ps -ef | grep m3_v2_trainer | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/v10_runs/m3v2_train.log || true"))
"""
    safe_colab_exec(session, script, timeout=60, retries=3, label="launch_m3_v2_train")


def launch_m3_v2_bench(session: str, remote_dir: str, extra_args: str) -> None:
    """在热态 GPU 实例后台启动 Milestone 3-V2 五方 N=16 配对统计终审基准。"""
    script = f"""
import subprocess, time
from pathlib import Path

Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
cmd = (
    "cd /content && PYTHONPATH=/content "
    "setsid nohup python3 -u -m v10_ouroboros64.ouroboros_bench.m3_v2_paired_bench "
    f"--out_dir {remote_dir!r} {extra_args} "
    "> /content/v10_runs/m3v2_bench.log 2>&1 < /dev/null &"
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
print(subprocess.getoutput("ps -ef | grep m3_v2_paired_bench | grep -v grep || true"))
print("--- LOG HEAD ---")
print(subprocess.getoutput("head -n 30 /content/v10_runs/m3v2_bench.log || true"))
"""
    safe_colab_exec(session, script, timeout=60, retries=3, label="launch_m3_v2_bench")


def check_status(session: str, remote_dir: str) -> None:
    """打印远端 GPU 状态、存活进程、遥测关键指标与日志尾。"""
    status_script = f"""
import json, subprocess
from pathlib import Path

print("--- GPU ---")
print(subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"))
print("--- PIDS ---")
print(subprocess.getoutput("ps -ef | grep -E 'ouroboros|m3_frontier|m3_trainer|m3_v2_' | grep -v grep || true"))
tel_p = Path({remote_dir!r}) / "telemetry_v10.json"
if tel_p.exists():
    d = json.loads(tel_p.read_text(encoding="utf-8"))
    meta = d.get("meta", {{}})
    hist = d.get("history", [])
    last = hist[-1] if hist else {{}}
    print("--- TELEMETRY ---")
    print("META:", json.dumps(meta, ensure_ascii=False))
    print("LAST:", json.dumps(last, ensure_ascii=False))
    heal = (d.get("defect_healing_eval") or {{}}).get("latest")
    if heal:
        print("HEAL:", json.dumps({{
            "step": heal.get("step"),
            "E0": heal.get("initial_energy_mean"),
            "walksat_end": heal.get("walksat_1bit", {{}}).get("final_energy_mean"),
            "walksat_pct": round(heal.get("walksat_1bit", {{}}).get("reduction_pct", 0.0), 2),
            "wolff_end": heal.get("neural_wolff", {{}}).get("final_energy_mean"),
            "wolff_pct": round(heal.get("neural_wolff", {{}}).get("reduction_pct", 0.0), 2),
            "tunnel_events": heal.get("neural_wolff", {{}}).get("tunneling_events"),
            "pair_acc": heal.get("pair_joint_acc"),
            "kl_gain": heal.get("nll_gain_over_rank1"),
        }}, ensure_ascii=False))
    blind = (d.get("blind_inversion_eval") or {{}}).get("latest")
    if blind:
        print("BLIND:", json.dumps(blind.get("cdcl_benchmarks"), ensure_ascii=False))
else:
    print("TELEMETRY: <not yet created>")
print("--- TRAIN LOG TAIL ---")
print(subprocess.getoutput("tail -n 18 /content/v10_runs/train_v10.log 2>/dev/null || true"))
print("--- M3 TRAIN LOG TAIL ---")
print(subprocess.getoutput("tail -n 18 /content/v10_runs/m3_train.log 2>/dev/null || true"))
print("--- M3-V2 TRAIN LOG TAIL ---")
print(subprocess.getoutput("tail -n 18 /content/v10_runs/m3v2_train.log 2>/dev/null || true"))
print("--- M3-V2 BENCH LOG TAIL ---")
print(subprocess.getoutput("tail -n 15 /content/v10_runs/m3v2_bench.log 2>/dev/null || true"))
"""
    safe_colab_exec(session, status_script, timeout=45, retries=3, label="status")


def sync_dir(session: str, remote_dir: str, local_dir: Path) -> None:
    """从远端打包拉取权重、分析报告与日志到本地目录。"""
    pack_script = f"""
import zipfile
from pathlib import Path

rdir = Path({remote_dir!r})
out_zip = Path("/tmp/v10_sync.zip")
copied = []
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(rdir.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".pt", ".png", ".log"):
            arc = p.relative_to(rdir).as_posix()
            zf.write(p, arcname=arc)
            copied.append(arc)
    for extra in (
        "/content/v10_runs/train_v10.log",
        "/content/v10_runs/m3_bench.log",
        "/content/v10_runs/m3_train.log",
        "/content/v10_runs/m3v2_train.log",
        "/content/v10_runs/m3v2_bench.log",
    ):
        ep = Path(extra)
        if ep.exists():
            zf.write(ep, arcname=ep.name)
            copied.append(ep.name)
print("SYNCED FILES:", copied, "zip_bytes=", out_zip.stat().st_size)
"""
    safe_colab_exec(session, pack_script, timeout=60, retries=3, label="pack_sync")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_zip = local_dir / "v10_sync.zip"
    subprocess.run(
        [str(COLAB_EXE), "download", "-s", session, "/tmp/v10_sync.zip", str(local_zip)],
        timeout=120,
        check=True,
    )
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(local_dir)
    local_zip.unlink(missing_ok=True)
    print(f"[Sync] 已解压同步至 {local_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Colab A100 部署与监督脚本")
    p.add_argument("--session", type=str, default="phase-sha-v10-a100")
    p.add_argument("--gpu", type=str, default="A100")
    p.add_argument("--remote_dir", type=str, default="/content/v10_runs/a100_m2")
    p.add_argument("--create", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--skip_pytest", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--launch_m3", action="store_true")
    p.add_argument("--launch_m3_train", action="store_true")
    p.add_argument("--launch_m3_v2_train", action="store_true")
    p.add_argument("--launch_m3_v2_bench", action="store_true")
    p.add_argument("--upload_ckpt", type=str, default="")
    p.add_argument("--remote_ckpt", type=str, default="/content/v10_runs/a100_m2/checkpoints/ouroboros_a100_step4000.pt")
    p.add_argument("--extra_args", type=str, default="")
    p.add_argument("--status", action="store_true")
    p.add_argument("--sync_dir", type=str, default="")
    p.add_argument("--exec_py", type=str, default="")
    args = p.parse_args()

    if args.exec_py:
        safe_colab_exec(
            args.session,
            Path(args.exec_py).read_text(encoding="utf-8"),
            timeout=90,
            retries=3,
            label=Path(args.exec_py).name,
        )
        return

    if args.create:
        create_session(args.session, gpu=args.gpu)
    if args.upload:
        upload_and_verify(args.session, run_pytest=not args.skip_pytest)
    if args.upload_ckpt:
        upload_checkpoint(args.session, Path(args.upload_ckpt), args.remote_ckpt)
    if args.launch:
        launch_training(args.session, args.remote_dir, args.extra_args)
    if args.launch_m3:
        launch_m3(args.session, args.remote_dir, args.extra_args)
    if args.launch_m3_train:
        launch_m3_train(args.session, args.remote_dir, args.extra_args)
    if args.launch_m3_v2_train:
        launch_m3_v2_train(args.session, args.remote_dir, args.extra_args)
    if args.launch_m3_v2_bench:
        launch_m3_v2_bench(args.session, args.remote_dir, args.extra_args)
    if args.status:
        check_status(args.session, args.remote_dir)
    if args.sync_dir:
        sync_dir(args.session, args.remote_dir, Path(args.sync_dir))


if __name__ == "__main__":
    main()
