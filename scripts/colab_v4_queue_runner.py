"""Unattended sequential Colab A100 training queue runner for Phase-SHA V4 (V4C -> V4A -> V4B)."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WORK_DIR = Path("/content/colab_bundle_v4")
RUNS_DIR = Path("/content/runs")
STATE_PATH = RUNS_DIR / "queue_state.json"
COMPLETE_PATH = RUNS_DIR / "QUEUE_COMPLETE.json"

QUEUE: List[Tuple[str, str, str, int]] = [
    ("v4c", "configs/v4c_r18_colab.yaml", "v4c_r18_colab", 10000),
    ("v4a", "configs/v4a_r18_colab.yaml", "v4a_r18_colab", 10000),
    ("v4b", "configs/v4b_r18_colab.yaml", "v4b_r18_colab", 10000),
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pid_is_alive_train(pid: int, run_dir: Path, variant: str) -> bool:
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        return False
    try:
        stat_txt = (proc_dir / "stat").read_text(encoding="utf-8", errors="replace")
        rparen = stat_txt.rfind(")")
        state = stat_txt[rparen + 2 : rparen + 3] if rparen != -1 else "?"
        if state == "Z":
            return False
        cmdline = (
            (proc_dir / "cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", errors="replace")
        )
        return "train_v4.py" in cmdline and (
            str(run_dir) in cmdline or f"--variant {variant}" in cmdline
        )
    except OSError:
        return False


def _find_running_train_pid(run_dir: Path, variant: str) -> Optional[int]:
    pid_file = run_dir / "train.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0])
            if _pid_is_alive_train(pid, run_dir, variant):
                return pid
        except (OSError, ValueError, IndexError):
            pass

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _pid_is_alive_train(pid, run_dir, variant):
            return pid
    return None


def _read_max_step(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    max_step = 0
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                val = (row.get("step") or "").strip()
                if not val:
                    continue
                try:
                    s = int(float(val))
                    if s > max_step:
                        max_step = s
                except ValueError:
                    continue
    except OSError:
        return 0
    return max_step


def _is_run_complete(run_dir: Path, target_steps: int) -> bool:
    return _read_max_step(run_dir / "metrics.csv") >= target_steps and (run_dir / "latest.pt").exists()


def _wait_for_external_pid(pid: int, run_dir: Path, variant: str, state: Dict[str, Any]) -> None:
    print(f"[queue] Adopting active {variant.upper()} process PID={pid} ({run_dir})", flush=True)
    while _pid_is_alive_train(pid, run_dir, variant):
        state["updated_at"] = _utc_now()
        state["variants"][variant]["step"] = _read_max_step(run_dir / "metrics.csv")
        state["variants"][variant]["pid"] = pid
        _write_json_atomic(STATE_PATH, state)
        time.sleep(10.0)


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not (WORK_DIR / "train_v4.py").exists():
        raise FileNotFoundError(f"Missing train_v4.py under {WORK_DIR}")

    state: Dict[str, Any] = {
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "queue_pid": os.getpid(),
        "current_variant": None,
        "status": "running",
        "variants": {
            var: {
                "run_name": rname,
                "target_steps": tsteps,
                "status": "pending",
                "step": _read_max_step(RUNS_DIR / rname / "metrics.csv"),
                "pid": None,
                "attempts": 0,
            }
            for var, _, rname, tsteps in QUEUE
        },
    }
    _write_json_atomic(STATE_PATH, state)

    max_retries = 2
    for variant, cfg_rel, run_name, target_steps in QUEUE:
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "metrics.csv"
        state["current_variant"] = variant
        vstate = state["variants"][variant]

        if _is_run_complete(run_dir, target_steps):
            vstate["status"] = "completed"
            vstate["step"] = _read_max_step(csv_path)
            vstate["pid"] = None
            state["updated_at"] = _utc_now()
            _write_json_atomic(STATE_PATH, state)
            print(f"[queue] {variant.upper()} already complete at step={vstate['step']}, advancing.", flush=True)
            continue

        vstate["status"] = "running"
        _write_json_atomic(STATE_PATH, state)

        alive_pid = _find_running_train_pid(run_dir, variant)
        if alive_pid is not None:
            _wait_for_external_pid(alive_pid, run_dir, variant, state)
            if _is_run_complete(run_dir, target_steps):
                vstate["status"] = "completed"
                vstate["step"] = _read_max_step(csv_path)
                vstate["pid"] = None
                state["updated_at"] = _utc_now()
                _write_json_atomic(STATE_PATH, state)
                print(
                    f"[queue] Adopted {variant.upper()} finished cleanly at step={vstate['step']}; sleeping 15s for final sync.",
                    flush=True,
                )
                time.sleep(15.0)
                continue

        attempt = 0
        while not _is_run_complete(run_dir, target_steps):
            attempt += 1
            if attempt > (1 + max_retries):
                vstate["status"] = "failed"
                vstate["pid"] = None
                state["status"] = f"failed_on_{variant}"
                state["updated_at"] = _utc_now()
                _write_json_atomic(STATE_PATH, state)
                raise RuntimeError(
                    f"[queue] {variant.upper()} failed to reach {target_steps} steps after {attempt - 1} attempts."
                )

            vstate["attempts"] = attempt
            latest_ckpt = run_dir / "latest.pt"
            use_resume = latest_ckpt.exists() and latest_ckpt.stat().st_size > 0
            cmd = [
                sys.executable,
                "-u",
                "train_v4.py",
                "--variant",
                variant,
                "--config",
                cfg_rel,
                "--save_dir",
                str(run_dir),
            ]
            if use_resume:
                cmd.extend(["--resume", str(latest_ckpt)])
                log_mode = "a"
            else:
                cmd.extend(["--init_from", "checkpoints/stage3b_r18_calibrated_best.pt"])
                log_mode = "w"

            print(
                f"[queue] Starting {variant.upper()} attempt={attempt} (resume={use_resume}) | cmd={' '.join(cmd)}",
                flush=True,
            )
            with (run_dir / "train.log").open(log_mode, encoding="utf-8", buffering=1) as log_f:
                if use_resume:
                    log_f.write(
                        f"\n[queue] Resuming {variant.upper()} at {_utc_now()} from {latest_ckpt}\n"
                    )
                    log_f.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(WORK_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                (run_dir / "train.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
                vstate["pid"] = proc.pid
                while proc.poll() is None:
                    vstate["step"] = _read_max_step(csv_path)
                    state["updated_at"] = _utc_now()
                    _write_json_atomic(STATE_PATH, state)
                    time.sleep(10.0)
                rc = proc.wait()

            vstate["step"] = _read_max_step(csv_path)
            vstate["pid"] = None
            state["updated_at"] = _utc_now()
            _write_json_atomic(STATE_PATH, state)
            print(
                f"[queue] {variant.upper()} process exited rc={rc} at step={vstate['step']}",
                flush=True,
            )
            if _is_run_complete(run_dir, target_steps):
                vstate["status"] = "completed"
                _write_json_atomic(STATE_PATH, state)
                print(
                    f"[queue] {variant.upper()} completed 100%; sleeping 15s for relay sync before next variant.",
                    flush=True,
                )
                time.sleep(15.0)
                break
            time.sleep(5.0)

    summary = {
        "completed_at": _utc_now(),
        "variants": {
            var: {
                "run_name": rname,
                "final_step": _read_max_step(RUNS_DIR / rname / "metrics.csv"),
                "target_steps": tsteps,
                "files": sorted(
                    f"{p.name} ({round(p.stat().st_size / (1024.0 * 1024.0), 2)} MB)"
                    for p in (RUNS_DIR / rname).glob("*")
                    if p.is_file()
                ),
            }
            for var, _, rname, tsteps in QUEUE
        },
    }
    state["status"] = "completed"
    state["current_variant"] = None
    state["updated_at"] = _utc_now()
    _write_json_atomic(STATE_PATH, state)
    _write_json_atomic(COMPLETE_PATH, summary)
    print(f"[queue] ALL VARIANTS COMPLETED: {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
