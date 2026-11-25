"""CPU-only Live Web Dashboard & TensorBoard feeder for Phase-SHA V4 Colab runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional


TRAIN_TAGS = [
    "train_loss",
    "train_phys_loss",
    "train_bce_loss",
    "train_xor_loss",
    "train_comm_loss",
    "train_align_loss",
    "lr",
    "peak_vram_gb",
    "step_time_ms",
    "frontier_rounds",
    "inst_E_hom",
    "inst_E32",
    "inst_mean_E",
    "hist_min",
    "alpha",
    "mask_m",
    "S_ent",
    "gl_gap",
    "bifurc",
    "bifurc_mps_calls",
    "endgame_polish_runs",
    "lin_viol",
    "nl_err",
    "round_r",
    "r_c_neural",
    "joint_nll",
    "mean_reward",
]

EVAL_TAGS = [
    "eval_bce_loss",
    "min_phys_energy_m32",
    "best_global_m32_energy",
    "phys_energy_m1",
    "phys_energy_m32",
    "zero_w0_bit_acc",
    "w0_free_bit_acc",
    "zero_latch_ratio",
    "exact_preimage_rate@256",
    "unique_ratio@256",
    "zero_manifold_align",
    "zero_manifold_comm",
    "zero_w0_lsb03_acc",
    "zero_w0_lsb03_zscore",
    "exact_hits",
    "eval_n_digests",
    "eval_free_bits",
    "w0_zscore",
    "r12_zscore",
    "r13_zscore",
    "r16_zscore",
    "exact_pass_1",
    "exact_pass_64",
    "hash_bit_acc",
    "hash_bit_acc_best64",
    "w0_bit_acc",
]

_STATE_LOCK = threading.Lock()
_LAST_TB_STEP: Dict[str, int] = {}
_TB_WRITERS: Dict[str, Any] = {}


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "inf", "-inf", "none", "null"}:
        return None
    try:
        num = float(s)
    except ValueError:
        return None
    return num if math.isfinite(num) else None


def _read_metrics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                parsed: Dict[str, Any] = {}
                for k, v in raw_row.items():
                    if k is None:
                        continue
                    key = str(k).strip()
                    num = _safe_float(v)
                    if key == "step" and num is not None:
                        parsed[key] = int(num)
                    else:
                        parsed[key] = num
                if parsed.get("step") is not None:
                    rows.append(parsed)
    except OSError:
        return []
    return rows


def _read_log_tail(log_path: Path, max_lines: int = 80) -> List[str]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def _infer_target_steps(log_lines: List[str], default: int = 10000) -> int:
    pat_train = re.compile(r"step=\s*\d+/(\d+)")
    for line in reversed(log_lines):
        m = pat_train.search(line)
        if m:
            try:
                return max(1, int(m.group(1)))
            except ValueError:
                pass
    return default


def _is_pid_running(pid_file: Path) -> Dict[str, Any]:
    if not pid_file.exists():
        return {"pid": None, "running": False}
    try:
        pid = int(pid_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return {"pid": None, "running": False}
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        return {"pid": pid, "running": False}
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8", errors="replace")
        rparen = stat_text.rfind(")")
        state = stat_text[rparen + 2 : rparen + 3] if rparen != -1 else "?"
        if state == "Z":
            return {"pid": pid, "running": False}
    except OSError:
        return {"pid": pid, "running": False}
    return {"pid": pid, "running": True}


def _query_gpu() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=4.0).strip()
    except Exception as exc:
        return {"error": str(exc), "raw": ""}
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 6:
        return {"raw": out}
    return {
        "name": parts[0],
        "util_pct": _safe_float(parts[1]),
        "mem_used_mib": _safe_float(parts[2]),
        "mem_total_mib": _safe_float(parts[3]),
        "temp_c": _safe_float(parts[4]),
        "power_w": _safe_float(parts[5]),
        "raw": first,
    }


def _query_processes() -> List[str]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,pmem,etime,args"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=4.0,
        )
    except Exception:
        return []
    matched: List[str] = []
    for line in out.splitlines():
        if (
            "train_v6a.py" in line
            or "train_v5.py" in line
            or "train_v4.py" in line
            or "monitor_dashboard.py" in line
        ):
            matched.append(line.strip())
    return matched


def _collect_status(runs_dir: Path) -> Dict[str, Any]:
    preferred = [
        "v7a_colab",
        "v7b_colab",
        "v7c_colab",
        "v6a_v2_r18_colab",
        "v6a_frontier_push",
        "v6a_r18_colab",
        "v6a_calibrate_colab",
        "v6a_v1_r18_archive",
        "v5d_pure_r18_colab",
        "v5d_neural_r18_colab",
        "v5c_r18_colab",
        "v5b_r18_colab",
        "v5a_r18_colab",
        "v4c_r18_colab",
        "v4b_r18_colab",
        "v4a_r18_colab",
    ]
    discovered: List[Path] = []
    if runs_dir.exists():
        dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        name_map = {d.name: d for d in dirs}
        for name in preferred:
            if name in name_map:
                discovered.append(name_map.pop(name))
        for name in sorted(name_map):
            discovered.append(name_map[name])

    runs_payload: Dict[str, Any] = {}
    for run_path in discovered:
        rows = _read_metrics_csv(run_path / "metrics.csv")
        latest_row = rows[-1] if rows else None
        latest_eval_row = None
        for r in reversed(rows):
            if r.get("eval_bce_loss") is not None:
                latest_eval_row = r
                break

        log_tail = _read_log_tail(run_path / "train.log", max_lines=80)
        target_steps = _infer_target_steps(log_tail, default=10000)
        pid_info = _is_pid_running(run_path / "train.pid")

        step_times = [
            r["step_time_ms"]
            for r in rows
            if r.get("step_time_ms") is not None and r["step_time_ms"] > 0
        ]
        median_step_ms = statistics.median(step_times[-5:]) if step_times else None
        cur_step = int(latest_row["step"]) if latest_row and latest_row.get("step") is not None else 0
        eta_sec = None
        if median_step_ms is not None and target_steps > cur_step:
            eta_sec = (median_step_ms / 1000.0) * float(target_steps - cur_step)

        ckpts = []
        for pt in sorted(run_path.glob("*.pt")):
            try:
                st = pt.stat()
                ckpts.append(
                    {
                        "name": pt.name,
                        "size_mb": round(st.st_size / (1024.0 * 1024.0), 2),
                        "mtime": round(st.st_mtime, 1),
                    }
                )
            except OSError:
                pass

        runs_payload[run_path.name] = {
            "name": run_path.name,
            "pid": pid_info["pid"],
            "is_running": pid_info["running"],
            "target_steps": target_steps,
            "current_step": cur_step,
            "median_step_ms": median_step_ms,
            "eta_sec": eta_sec,
            "row_count": len(rows),
            "rows": rows,
            "latest_row": latest_row,
            "latest_eval_row": latest_eval_row,
            "log_tail": log_tail,
            "checkpoints": ckpts,
        }

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": _query_gpu(),
        "processes": _query_processes(),
        "runs": runs_payload,
    }


def _sync_tensorboard_once(runs_dir: Path, tb_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return

    if not runs_dir.exists():
        return

    for run_path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        csv_path = run_path / "metrics.csv"
        if not csv_path.exists():
            continue
        rows = _read_metrics_csv(csv_path)
        if not rows:
            continue
        run_name = run_path.name
        with _STATE_LOCK:
            last_step = _LAST_TB_STEP.get(run_name, -1)
            writer = _TB_WRITERS.get(run_name)
            if writer is None:
                log_dir = tb_dir / run_name
                log_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(log_dir=str(log_dir))
                _TB_WRITERS[run_name] = writer

        wrote = False
        for row in rows:
            step = row.get("step")
            if step is None or int(step) <= last_step:
                continue
            s_int = int(step)
            for tag in TRAIN_TAGS:
                val = row.get(tag)
                if val is not None:
                    writer.add_scalar(f"train/{tag}", float(val), s_int)
            for tag in EVAL_TAGS:
                val = row.get(tag)
                if val is not None:
                    writer.add_scalar(f"eval/{tag}", float(val), s_int)
            for r_idx in range(18):
                for prefix in ("per_round_acc_", "zero_round_acc_"):
                    k = f"{prefix}{r_idx}"
                    val = row.get(k)
                    if val is not None:
                        writer.add_scalar(f"rounds/{k}", float(val), s_int)
            last_step = s_int
            wrote = True

        if wrote:
            writer.flush()
            with _STATE_LOCK:
                _LAST_TB_STEP[run_name] = last_step


def _write_status_snapshot(runs_dir: Path) -> None:
    payload = json.dumps(_collect_status(runs_dir), ensure_ascii=False)
    tmp_path = runs_dir / "dashboard_status.json.tmp"
    dst_path = runs_dir / "dashboard_status.json"
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, dst_path)


def _tb_loop(runs_dir: Path, tb_dir: Path, poll_sec: float, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            _write_status_snapshot(runs_dir)
        except Exception as exc:
            print(f"[dashboard] snapshot write error: {exc}", flush=True)
        try:
            _sync_tensorboard_once(runs_dir, tb_dir)
        except Exception as exc:
            print(f"[dashboard] tensorboard sync error: {exc}", flush=True)
        stop_event.wait(poll_sec)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Phase-SHA V4/V5/V6A · Colab A100 实时监控仪表盘</title>
<style>
  :root {
    --bg: #0b101b;
    --card: #121a2b;
    --border: #23314d;
    --text: #e5ecf6;
    --muted: #8fa3c7;
    --accent: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 18px 22px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 14px;
  }
  header {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
  }
  .title-wrap { display: flex; align-items: center; gap: 12px; }
  .title { font-size: 20px; font-weight: 700; letter-spacing: 0.2px; }
  .badge {
    padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
    border: 1px solid var(--border); background: #172338; color: var(--muted);
  }
  .badge.ok { border-color: rgba(52,211,153,0.45); color: var(--green); background: rgba(52,211,153,0.12); }
  .badge.warn { border-color: rgba(251,191,36,0.45); color: var(--amber); background: rgba(251,191,36,0.12); }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  button, select {
    background: var(--card); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  .grid-kpi {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .kpi-label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  .kpi-val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .kpi-sub { color: var(--muted); font-size: 12px; margin-top: 6px; }
  .progress-bg {
    width: 100%; height: 8px; background: #1a253b; border-radius: 999px; overflow: hidden; margin-top: 8px;
  }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); width: 0%; transition: width 0.4s; }
  .grid-charts {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .chart-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
  .legend { font-size: 11px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  svg.chart { width: 100%; height: 210px; display: block; background: #0d1422; border-radius: 8px; }
  .bottom-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 12px; }
  @media (max-width: 1000px) { .bottom-grid { grid-template-columns: 1fr; } }
  pre.logbox {
    margin: 0; padding: 12px; background: #090e18; border: 1px solid var(--border); border-radius: 8px;
    color: #cbd5e1; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px; line-height: 1.45; height: 320px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
  }
  .ckpt-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
</style>
</head>
<body>
  <header>
    <div class="title-wrap">
      <div class="title">Phase-SHA V4/V5/V6A · Colab 实时监控面板</div>
      <span id="runBadge" class="badge">连接中...</span>
      <span id="tsText" style="color:var(--muted);font-size:12px;"></span>
    </div>
    <div class="controls">
      <label style="color:var(--muted);font-size:12px;">变体运行:</label>
      <select id="runSelect"></select>
      <button id="toggleAutoBtn">自动刷新: 开 (5s)</button>
      <button id="refreshBtn">立即刷新</button>
    </div>
  </header>

  <section id="v7Leaderboard" class="panel" style="display:none; margin-bottom:14px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <h2 style="margin:0; font-size:15px;">V7 三叉戟纯神经求逆实时排行榜（对比 V6A 经典同伦基线 R_c = 5~6）</h2>
      <span style="font-size:12px; color:#94a3b8;">铁血赛规：eval 100% 纯神经推理 · free_msg · allow_peel_back=False</span>
    </div>
    <table style="width:100%; border-collapse:collapse; font-size:12px;">
      <thead>
        <tr style="border-bottom:1px solid #334155; color:#94a3b8; text-align:left;">
          <th style="padding:4px 6px;">路线 / Run</th>
          <th style="padding:4px 6px;">状态</th>
          <th style="padding:4px 6px;">当前轮 R</th>
          <th style="padding:4px 6px;">pass@1</th>
          <th style="padding:4px 6px;">pass@64</th>
          <th style="padding:4px 6px;">hash_bit_acc</th>
          <th style="padding:4px 6px;">best64</th>
          <th style="padding:4px 6px;">nll / reward</th>
          <th style="padding:4px 6px;">R_c_neural</th>
          <th style="padding:4px 6px;">VRAM</th>
        </tr>
      </thead>
      <tbody id="v7LeaderboardBody"></tbody>
    </table>
  </section>

  <section class="grid-kpi">
    <div class="card">
      <div class="kpi-label">Colab GPU 硬件与显存 (FP32)</div>
      <div id="gpuName" class="kpi-val" style="font-size:17px;">--</div>
      <div id="gpuStats" class="kpi-sub">--</div>
    </div>
    <div class="card">
      <div class="kpi-label">训练进度与预计剩余时间 (ETA)</div>
      <div id="stepVal" class="kpi-val">--</div>
      <div class="progress-bg"><div id="stepBar" class="progress-fill"></div></div>
      <div id="etaSub" class="kpi-sub">--</div>
    </div>
    <div class="card">
      <div id="card3Label" class="kpi-label">Eval BCE 校准 (基准 ln2 = 0.6931)</div>
      <div id="bceVal" class="kpi-val">--</div>
      <div id="bceSub" class="kpi-sub">--</div>
    </div>
    <div class="card">
      <div id="card4Label" class="kpi-label">物理能量最值 (M32)</div>
      <div id="energyVal" class="kpi-val">--</div>
      <div id="energySub" class="kpi-sub">--</div>
    </div>
    <div class="card">
      <div id="card5Label" class="kpi-label">W0 比特准确率 / 锁存 / Pre@256</div>
      <div id="accVal" class="kpi-val">--</div>
      <div id="accSub" class="kpi-sub">--</div>
    </div>
  </section>

  <section class="grid-charts">
    <div class="card">
      <div class="chart-title">
        <span id="c1Title">训练损失曲线</span>
        <div id="c1Legend" class="legend">
          <span><span class="dot" style="background:var(--accent)"></span>train_loss</span>
          <span><span class="dot" style="background:var(--purple)"></span>train_phys_loss</span>
        </div>
      </div>
      <svg id="chartTrain" class="chart" viewBox="0 0 480 210"></svg>
    </div>
    <div class="card">
      <div class="chart-title">
        <span id="c2Title">评估 BCE 与 物理能量 M32</span>
        <div id="c2Legend" class="legend">
          <span><span class="dot" style="background:var(--green)"></span>eval_bce (虚线 ln2)</span>
          <span><span class="dot" style="background:var(--amber)"></span>min_E32(右轴缩放)</span>
        </div>
      </div>
      <svg id="chartEval" class="chart" viewBox="0 0 480 210"></svg>
    </div>
    <div class="card">
      <div class="chart-title">
        <span id="c3Title">破壁探索与锁存指标</span>
        <div id="c3Legend" class="legend">
          <span><span class="dot" style="background:var(--accent)"></span>zero_w0_acc</span>
          <span><span class="dot" style="background:var(--amber)"></span>latch_z</span>
          <span><span class="dot" style="background:var(--green)"></span>pre@256</span>
        </div>
      </div>
      <svg id="chartProbe" class="chart" viewBox="0 0 480 210"></svg>
    </div>
  </section>

  <section class="bottom-grid">
    <div class="card">
      <div class="chart-title">
        <span id="roundTitle">R=18 逐轮波前准确率 (r0..r17，高亮闭环死墙 r13 / r16)</span>
        <span id="ckptCount" style="color:var(--muted);font-size:12px;"></span>
      </div>
      <svg id="chartRounds" class="chart" style="height:260px;" viewBox="0 0 560 260"></svg>
      <div id="ckptRow" class="ckpt-row"></div>
    </div>
    <div class="card">
      <div class="chart-title">
        <span>实时训练日志 (train.log 末尾 80 行)</span>
        <span id="procCount" style="color:var(--muted);font-size:12px;"></span>
      </div>
      <pre id="logBox" class="logbox">等待日志输出...</pre>
    </div>
  </section>

<script>
(function() {
  let autoRefresh = true;
  let timerId = null;
  let selectedRun = null;

  function fmtNum(v, digits=4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return Number(v).toFixed(digits);
  }
  function fmtEta(sec) {
    if (sec === null || sec === undefined || sec < 0) return "--";
    const s = Math.round(sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    if (h > 0) return `${h}h ${m}m ${rem}s`;
    if (m > 0) return `${m}m ${rem}s`;
    return `${rem}s`;
  }

  function svgEl(tag, attrs, text) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function renderLineChart(svgId, seriesList, refLines=[], pulses=[]) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 480, H = 210, padL = 48, padR = 14, padT = 14, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const allPts = [];
    for (const s of seriesList) {
      for (const p of s.points) {
        if (p.y !== null && p.y !== undefined && Number.isFinite(p.y)) allPts.push(p);
      }
    }
    if (allPts.length === 0) {
      svg.appendChild(svgEl("text", {x: W/2, y: H/2, fill: "#8fa3c7", "text-anchor": "middle", "font-size": "13"}, "暂无数据点，等待采样..."));
      return;
    }

    const xs = allPts.map(p => p.x);
    let minX = Math.min(...xs), maxX = Math.max(...xs);
    if (maxX === minX) { minX = Math.max(0, minX - 1); maxX = maxX + 1; }

    // Support optional independent normalized scale for secondary axis series
    const primaryPts = [];
    for (const s of seriesList) {
      if (!s.secondary) {
        for (const p of s.points) if (Number.isFinite(p.y)) primaryPts.push(p.y);
      }
    }
    for (const r of refLines) primaryPts.push(r.y);
    let minY = primaryPts.length ? Math.min(...primaryPts) : 0;
    let maxY = primaryPts.length ? Math.max(...primaryPts) : 1;
    if (Math.abs(maxY - minY) < 1e-6) { minY -= 0.1; maxY += 0.1; }
    const spanY = maxY - minY;
    minY -= spanY * 0.08;
    maxY += spanY * 0.08;

    const xToPx = x => padL + ((x - minX) / (maxX - minX)) * plotW;
    const yToPx = (y, lo, hi) => padT + (1.0 - (y - lo) / Math.max(1e-9, hi - lo)) * plotH;

    // Grid
    for (let i = 0; i <= 4; i++) {
      const val = minY + (i / 4) * (maxY - minY);
      const py = yToPx(val, minY, maxY);
      svg.appendChild(svgEl("line", {x1: padL, y1: py, x2: W - padR, y2: py, stroke: "#1c2942", "stroke-width": 1}));
      svg.appendChild(svgEl("text", {x: padL - 6, y: py + 4, fill: "#8fa3c7", "font-size": "10", "text-anchor": "end"}, val.toFixed(3)));
    }
    svg.appendChild(svgEl("text", {x: padL, y: H - 7, fill: "#8fa3c7", "font-size": "10"}, `step ${Math.round(minX)}`));
    svg.appendChild(svgEl("text", {x: W - padR, y: H - 7, fill: "#8fa3c7", "font-size": "10", "text-anchor": "end"}, `step ${Math.round(maxX)}`));

    for (const r of refLines) {
      const py = yToPx(r.y, minY, maxY);
      if (py >= padT && py <= H - padB) {
        svg.appendChild(svgEl("line", {x1: padL, y1: py, x2: W - padR, y2: py, stroke: r.color || "#64748b", "stroke-dasharray": "4,4", "stroke-width": 1.2}));
        svg.appendChild(svgEl("text", {x: W - padR - 4, y: py - 4, fill: r.color || "#94a3b8", "font-size": "10", "text-anchor": "end"}, r.label));
      }
    }

    for (const s of seriesList) {
      const valid = s.points.filter(p => p.y !== null && p.y !== undefined && Number.isFinite(p.y));
      if (!valid.length) continue;
      let lo = minY, hi = maxY;
      if (s.secondary) {
        const vals = valid.map(p => p.y);
        lo = Math.min(...vals); hi = Math.max(...vals);
        if (Math.abs(hi - lo) < 1e-6) { lo -= 1; hi += 1; }
        const sp = hi - lo; lo -= sp * 0.08; hi += sp * 0.08;
      }
      if (valid.length >= 2) {
        let d = "";
        if (s.stepped) {
          d = `M${xToPx(valid[0].x).toFixed(1)},${yToPx(valid[0].y, lo, hi).toFixed(1)}`;
          for (let idx = 1; idx < valid.length; idx++) {
            const xPx = xToPx(valid[idx].x).toFixed(1);
            const yPx = yToPx(valid[idx].y, lo, hi).toFixed(1);
            d += ` H${xPx} V${yPx}`;
          }
        } else {
          d = valid.map((p, idx) => `${idx === 0 ? "M" : "L"}${xToPx(p.x).toFixed(1)},${yToPx(p.y, lo, hi).toFixed(1)}`).join(" ");
        }
        const pathAttrs = {d, fill: "none", stroke: s.color, "stroke-width": 2};
        if (s.dash) pathAttrs["stroke-dasharray"] = s.dash;
        svg.appendChild(svgEl("path", pathAttrs));
      }
      const last = valid[valid.length - 1];
      svg.appendChild(svgEl("circle", {cx: xToPx(last.x), cy: yToPx(last.y, lo, hi), r: 3.5, fill: s.color}));
    }

    for (const pl of pulses || []) {
      if (pl.x === null || pl.x === undefined || !Number.isFinite(pl.x)) continue;
      const px = xToPx(pl.x);
      const py = pl.y !== null && Number.isFinite(pl.y) ? yToPx(pl.y, minY, maxY) : padT + 14;
      svg.appendChild(svgEl("line", {x1: px, y1: padT, x2: px, y2: H - padB, stroke: "#f87171", "stroke-width": 1.3, "stroke-dasharray": "3,3"}));
      svg.appendChild(svgEl("circle", {cx: px, cy: py, r: 4.5, fill: "#ef4444", stroke: "#fecaca", "stroke-width": 1.2}));
      svg.appendChild(svgEl("text", {x: Math.min(W - padR - 14, px + 4), y: padT + 11, fill: "#fca5a5", "font-size": "9.5"}, pl.label || "⚡"));
    }
  }

  function renderRoundsBar(evalRow) {
    const svg = document.getElementById("chartRounds");
    svg.innerHTML = "";
    const W = 560, H = 260, padL = 38, padR = 14, padT = 18, padB = 34;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    if (!evalRow) {
      svg.appendChild(svgEl("text", {x: W/2, y: H/2, fill: "#8fa3c7", "text-anchor": "middle", "font-size": "13"}, "暂无 Eval 评估点..."));
      return;
    }

    let prefix = "zero_round_acc_";
    if (evalRow[prefix + "0"] === undefined || evalRow[prefix + "0"] === null) {
      prefix = "per_round_acc_";
    }
    const vals = [];
    for (let r = 0; r < 18; r++) {
      vals.push({r, v: evalRow[prefix + r]});
    }
    const labelPrefix = prefix === "zero_round_acc_" ? "zero_round_acc" : "per_round_acc";
    document.getElementById("roundTitle").textContent = `R=18 逐轮波前准确率 (${labelPrefix} @ step ${evalRow.step}，橙/红=闭环死墙 r13/r16)`;

    for (const mark of [0.0, 0.5, 0.75, 1.0]) {
      const py = padT + (1.0 - mark) * plotH;
      svg.appendChild(svgEl("line", {
        x1: padL, y1: py, x2: W - padR, y2: py,
        stroke: mark === 0.5 ? "#64748b" : "#1c2942",
        "stroke-dasharray": mark === 0.5 ? "4,4" : "none",
        "stroke-width": 1
      }));
      svg.appendChild(svgEl("text", {x: padL - 5, y: py + 4, fill: "#8fa3c7", "font-size": "10", "text-anchor": "end"}, mark.toFixed(2)));
    }

    const slot = plotW / 18;
    const barW = slot * 0.68;
    for (const item of vals) {
      const v = Number.isFinite(item.v) ? Math.max(0, Math.min(1, item.v)) : 0;
      const x = padL + item.r * slot + (slot - barW) / 2;
      const barH = v * plotH;
      const y = padT + plotH - barH;
      const isWall = (item.r === 13 || item.r === 16);
      const fill = isWall ? "#f87171" : (v >= 0.99 ? "#34d399" : "#38bdf8");
      svg.appendChild(svgEl("rect", {x, y, width: barW, height: Math.max(1, barH), rx: 3, fill}));
      svg.appendChild(svgEl("text", {x: x + barW/2, y: Math.max(11, y - 4), fill: "#cbd5e1", "font-size": "9", "text-anchor": "middle"}, v.toFixed(2)));
      svg.appendChild(svgEl("text", {x: x + barW/2, y: H - 12, fill: isWall ? "#fca5a5" : "#8fa3c7", "font-size": "10", "text-anchor": "middle"}, `r${item.r}`));
    }
  }

  function updateUI(data) {
    const gpu = data.gpu || {};
    if (gpu.name) {
      document.getElementById("gpuName").textContent = `${gpu.name} · ${gpu.util_pct ?? "--"}%`;
      const usedGb = gpu.mem_used_mib != null ? (gpu.mem_used_mib / 1024).toFixed(2) : "--";
      const totGb = gpu.mem_total_mib != null ? (gpu.mem_total_mib / 1024).toFixed(1) : "--";
      document.getElementById("gpuStats").textContent = `显存: ${usedGb} / ${totGb} GB | 温度: ${gpu.temp_c ?? "--"}°C | 功耗: ${gpu.power_w ?? "--"}W`;
    } else {
      document.getElementById("gpuName").textContent = "GPU 状态未知";
      document.getElementById("gpuStats").textContent = gpu.error || "--";
    }
    document.getElementById("tsText").textContent = `最近更新: ${new Date().toLocaleTimeString()}`;

    const runs = data.runs || {};
    const runKeys = Object.keys(runs);
    const sel = document.getElementById("runSelect");
    const prev = selectedRun || sel.value;
    sel.innerHTML = "";
    for (const k of runKeys) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${k} (${runs[k].is_running ? "运行中" : "已停止"})`;
      sel.appendChild(opt);
    }
    if (runKeys.length === 0) {
      document.getElementById("runBadge").textContent = "暂无运行目录";
      document.getElementById("runBadge").className = "badge warn";
      return;
    }
    if (prev && runs[prev]) selectedRun = prev;
    else selectedRun = runKeys[0];
    sel.value = selectedRun;

    const r = runs[selectedRun];
    const badge = document.getElementById("runBadge");
    if (r.is_running) {
      badge.textContent = `${r.name} · PID ${r.pid} 运行中`;
      badge.className = "badge ok";
    } else {
      badge.textContent = `${r.name} · 未在运行`;
      badge.className = "badge warn";
    }

    const pct = Math.min(100, (r.current_step / Math.max(1, r.target_steps)) * 100);
    document.getElementById("stepVal").textContent = `${r.current_step} / ${r.target_steps} (${pct.toFixed(1)}%)`;
    document.getElementById("stepBar").style.width = `${pct}%`;
    const stepMsText = r.median_step_ms != null ? `${r.median_step_ms.toFixed(1)} ms/step` : "--";
    document.getElementById("etaSub").textContent = `步速: ${stepMsText} | ETA: ${fmtEta(r.eta_sec)}`;

    const ev = r.latest_eval_row;
    const latest = r.latest_row;
    const rows = r.rows || [];
    const evalRows = rows.filter(x => x.eval_bce_loss !== null && x.eval_bce_loss !== undefined);
    const v7EvalRows = rows.filter(x => x.exact_pass_64 !== null && x.exact_pass_64 !== undefined);
    const v7Ev = v7EvalRows.length ? v7EvalRows[v7EvalRows.length - 1] : null;
    const isV7 = Boolean(latest && (
      (latest.round_r !== null && latest.round_r !== undefined) ||
      (latest.r_c_neural !== null && latest.r_c_neural !== undefined)
    ));
    const isV6A = Boolean(!isV7 && latest && latest.inst_E32 !== null && latest.inst_E32 !== undefined);

    // Populate V7 3-way leaderboard whenever any v7* run is present
    const lbPanel = document.getElementById('v7Leaderboard');
    const lbBody = document.getElementById('v7LeaderboardBody');
    const v7Entries = Object.entries(runs).filter(([k, _]) => /^v7[abc]/i.test(k));
    if (lbPanel && lbBody && v7Entries.length > 0) {
      lbPanel.style.display = 'block';
      const rowsHtml = v7Entries.map(([rname, info]) => {
        const lr = info.latest_row || {};
        const ers = (info.rows || []).filter(x => x.exact_pass_64 !== null && x.exact_pass_64 !== undefined);
        const er = ers.length ? ers[ers.length - 1] : {};
        const curR = lr.round_r !== null && lr.round_r !== undefined ? Math.round(lr.round_r) : '-';
        const p1 = er.exact_pass_1 !== null && er.exact_pass_1 !== undefined ? (er.exact_pass_1 * 100).toFixed(2) + '%' : '-';
        const p64 = er.exact_pass_64 !== null && er.exact_pass_64 !== undefined ? (er.exact_pass_64 * 100).toFixed(2) + '%' : '-';
        const hbit = er.hash_bit_acc !== null && er.hash_bit_acc !== undefined ? Number(er.hash_bit_acc).toFixed(4) : '-';
        const b64 = er.hash_bit_acc_best64 !== null && er.hash_bit_acc_best64 !== undefined ? Number(er.hash_bit_acc_best64).toFixed(4) : '-';
        const sig = lr.mean_reward !== null && lr.mean_reward !== undefined
          ? 'rew=' + Number(lr.mean_reward).toFixed(3)
          : (lr.joint_nll !== null && lr.joint_nll !== undefined ? 'nll=' + Number(lr.joint_nll).toFixed(2) : '-');
        const rcn = lr.r_c_neural !== null && lr.r_c_neural !== undefined ? Math.round(lr.r_c_neural) : 0;
        const vram = lr.peak_vram_gb !== null && lr.peak_vram_gb !== undefined ? Number(lr.peak_vram_gb).toFixed(2) + 'G' : '-';
        const st = info.is_running ? '<span style="color:#4ade80;">RUNNING</span>' : '<span style="color:#94a3b8;">IDLE</span>';
        return `<tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:4px 6px; font-weight:600;">${rname}</td>
          <td style="padding:4px 6px;">${st}</td>
          <td style="padding:4px 6px;">R=${curR}</td>
          <td style="padding:4px 6px;">${p1}</td>
          <td style="padding:4px 6px; color:#38bdf8; font-weight:600;">${p64}</td>
          <td style="padding:4px 6px;">${hbit}</td>
          <td style="padding:4px 6px;">${b64}</td>
          <td style="padding:4px 6px;">${sig}</td>
          <td style="padding:4px 6px; color:#facc15; font-weight:700;">R_c=${rcn}</td>
          <td style="padding:4px 6px;">${vram}</td>
        </tr>`;
      }).join('');
      const v6aRow = `<tr style="background:#0f172a; color:#94a3b8;">
        <td style="padding:4px 6px;">V6A_Classic_Ref</td>
        <td style="padding:4px 6px;">FROZEN</td>
        <td style="padding:4px 6px;">R=6</td>
        <td style="padding:4px 6px;">-</td>
        <td style="padding:4px 6px;">R=4~5 100% / R=6 minE=0</td>
        <td style="padding:4px 6px;">-</td>
        <td style="padding:4px 6px;">-</td>
        <td style="padding:4px 6px;">GL-RRR</td>
        <td style="padding:4px 6px;">R_c=5~6</td>
        <td style="padding:4px 6px;">-</td>
      </tr>`;
      lbBody.innerHTML = rowsHtml + v6aRow;
    } else if (lbPanel) {
      lbPanel.style.display = 'none';
    }

    if (isV7) {
      document.getElementById('kStep').textContent = `${latest.step ?? '-'} (R=${latest.round_r ?? '-'})`;
      const sigTxt = latest.mean_reward !== null && latest.mean_reward !== undefined
        ? `rew=${fmtNum(latest.mean_reward, 3)}`
        : `nll=${fmtNum(latest.joint_nll, 2)}`;
      document.getElementById('kLoss').textContent = `loss=${fmtNum(latest.train_loss, 4)} | ${sigTxt} | vram=${fmtNum(latest.peak_vram_gb, 2)}G`;

      document.getElementById('kE32').textContent = v7Ev ? `p@64=${fmtPct(v7Ev.exact_pass_64)} | p@1=${fmtPct(v7Ev.exact_pass_1)}` : 'await eval';
      document.getElementById('kPre').textContent = v7Ev ? `hits=${Math.round(v7Ev.exact_hits || 0)}/${Math.round(v7Ev.eval_n_digests || 0)} | best64=${fmtNum(v7Ev.hash_bit_acc_best64, 4)}` : '-';

      document.getElementById('kZ').textContent = v7Ev ? `hash=${fmtNum(v7Ev.hash_bit_acc, 4)} | w0_z=${fmtNum(v7Ev.w0_zscore, 2)}` : '-';
      document.getElementById('kManifold').textContent = `R_c_neural=${Math.round(latest.r_c_neural || 0)} (vs V6A 5~6)`;

      document.getElementById('kProgress').textContent = `evals=${v7EvalRows.length} | train_pts=${rows.length}`;
      const el = r.elapsed_sec;
      document.getElementById('kEta').textContent = `elapsed=${el ? (el / 60).toFixed(1) + 'm' : '-'}`;

      renderLineChart('chartBce', [
        { name: 'train_loss', color: '#60a5fa', points: rows.filter(x => x.train_loss !== null).map(x => ({ x: x.step, y: x.train_loss })) },
        { name: 'hash_bit_acc', color: '#4ade80', points: v7EvalRows.filter(x => x.hash_bit_acc !== null).map(x => ({ x: x.step, y: x.hash_bit_acc })) }
      ]);
      renderLineChart('chartZ', [
        { name: 'exact_pass@64', color: '#38bdf8', points: v7EvalRows.filter(x => x.exact_pass_64 !== null).map(x => ({ x: x.step, y: x.exact_pass_64 })) },
        { name: 'exact_pass@1', color: '#facc15', points: v7EvalRows.filter(x => x.exact_pass_1 !== null).map(x => ({ x: x.step, y: x.exact_pass_1 })) }
      ]);
      renderLineChart('chartPhy', [
        { name: 'r_c_neural', color: '#fb923c', points: rows.filter(x => x.r_c_neural !== null).map(x => ({ x: x.step, y: x.r_c_neural })) },
        { name: 'round_r', color: '#a78bfa', points: rows.filter(x => x.round_r !== null).map(x => ({ x: x.step, y: x.round_r })) }
      ]);
    } else if (isV6A) {
      document.getElementById("card3Label").textContent = "Homotopy α & 光圈 mask_m (V6A-v2)";
      const ac = latest.alpha_c !== null && latest.alpha_c !== undefined ? fmtNum(latest.alpha_c, 3) : "--";
      document.getElementById("bceVal").textContent = `α=${fmtNum(latest.alpha, 3)} | m=${Math.round(latest.mask_m ?? 256)}`;
      document.getElementById("bceSub").textContent =
        `bifurc=${Math.round(latest.bifurc ?? 0)} (mps=${Math.round(latest.bifurc_mps_calls ?? 0)}) | α_c=${ac} | polish=${Math.round(latest.endgame_polish_runs ?? 0)}`;

      document.getElementById("card4Label").textContent = "瞬时能量 inst_E32 (严禁混用 hist_min)";
      document.getElementById("energyVal").textContent = fmtNum(latest.inst_E32, 3);
      document.getElementById("energySub").textContent =
        `hist_min=${fmtNum(latest.hist_min, 3)} | inst_E_hom=${fmtNum(latest.inst_E_hom, 4)} | gMin=${fmtNum(latest.best_global_m32_energy, 3)}`;

      const zSpan = (lbl, zVal) => {
        if (zVal === null || zVal === undefined || !Number.isFinite(zVal)) return `${lbl}:--`;
        const zStr = (zVal >= 0 ? "+" : "") + Number(zVal).toFixed(2);
        if (Math.abs(zVal) > 3.5) {
          const col = zVal > 0 ? "#34d399" : "#fbbf24";
          return `<b style="color:${col}">${lbl}:z=${zStr}*</b>`;
        }
        return `<span style="color:var(--muted)">${lbl}:z=${zStr}</span>`;
      };
      const evStat = ev || latest;
      const nDig = Math.round(evStat.eval_n_digests ?? 0);
      const nFreeB = Math.round(evStat.eval_free_bits ?? 0);
      document.getElementById("card5Label").textContent = "MPS S_ent / GL-Gap / 二项 z-score(|z|>3.5显)";
      document.getElementById("accVal").textContent = `S=${fmtNum(latest.S_ent, 3)} | gap=${fmtNum(latest.gl_gap, 4)} | hit=${Math.round(latest.exact_hits ?? 0)}`;
      document.getElementById("accSub").innerHTML =
        `N=${nDig} (${nFreeB}b) | ${zSpan("w0", evStat.w0_zscore)} ${zSpan("r12", evStat.r12_zscore)} ${zSpan("r13", evStat.r13_zscore)} ${zSpan("r16", evStat.r16_zscore)}`;

      document.getElementById("c1Title").textContent = "V6A 真实能量与历史最小对比 (M32)";
      document.getElementById("c1Legend").innerHTML =
        '<span><span class="dot" style="background:var(--accent)"></span>inst_E32</span>' +
        '<span><span class="dot" style="background:var(--green)"></span>hist_min</span>' +
        '<span><span class="dot" style="background:var(--purple)"></span>inst_mean_E</span>';
      document.getElementById("c2Title").textContent = "2D 同伦推进 (α, mask_m 阶梯) & ⚡分岔脉冲";
      document.getElementById("c2Legend").innerHTML =
        '<span><span class="dot" style="background:var(--green)"></span>alpha</span>' +
        '<span><span class="dot" style="background:var(--accent)"></span>mask_m/256</span>' +
        '<span><span class="dot" style="background:var(--amber)"></span>inst_E_hom(右轴)</span>' +
        '<span><span class="dot" style="background:var(--red)"></span>⚡bifurc</span>';
      document.getElementById("c3Title").textContent = "MPS 纠缠熵 / GL-Gap / 命中率";
      document.getElementById("c3Legend").innerHTML =
        '<span><span class="dot" style="background:var(--accent)"></span>S_ent (虚线0.55)</span>' +
        '<span><span class="dot" style="background:var(--amber)"></span>gl_gap</span>' +
        '<span><span class="dot" style="background:var(--green)"></span>pre@256</span>';

      const pulses = [];
      for (let i = 0; i < rows.length; i++) {
        const curB = Math.round(rows[i].bifurc ?? 0);
        const prevB = i > 0 ? Math.round(rows[i - 1].bifurc ?? 0) : 0;
        if (curB > prevB) {
          pulses.push({x: rows[i].step, y: rows[i].alpha, label: `⚡${curB}`});
        }
      }

      renderLineChart("chartTrain", [
        {color: "#38bdf8", points: rows.map(x => ({x: x.step, y: x.inst_E32}))},
        {color: "#34d399", points: rows.map(x => ({x: x.step, y: x.hist_min}))},
        {color: "#a78bfa", points: rows.map(x => ({x: x.step, y: x.inst_mean_E}))},
      ]);
      renderLineChart("chartEval", [
        {color: "#34d399", points: rows.map(x => ({x: x.step, y: x.alpha}))},
        {color: "#38bdf8", stepped: true, dash: "5,3", points: rows.map(x => ({x: x.step, y: (x.mask_m ?? 256) / 256.0}))},
        {color: "#fbbf24", secondary: true, points: rows.map(x => ({x: x.step, y: x.inst_E_hom}))},
      ], [{y: 1.0, color: "#94a3b8", label: "1.0 / m=256"}], pulses);
      renderLineChart("chartProbe", [
        {color: "#38bdf8", points: rows.map(x => ({x: x.step, y: x.S_ent}))},
        {color: "#fbbf24", points: rows.map(x => ({x: x.step, y: x.gl_gap}))},
        {color: "#34d399", points: rows.map(x => ({x: x.step, y: x["exact_preimage_rate@256"]}))},
      ], [{y: 0.55, color: "#94a3b8", label: "S_min=0.55"}]);
      renderRoundsBar(ev || latest);
    } else {
      document.getElementById("card3Label").textContent = "Eval BCE 校准 (基准 ln2 = 0.6931)";
      document.getElementById("bceVal").textContent = ev ? fmtNum(ev.eval_bce_loss, 4) : "--";
      document.getElementById("bceSub").textContent = ev
        ? `最新评估 @ step ${ev.step} | train_loss=${fmtNum(latest?.train_loss, 4)}`
        : "等待首轮 eval...";

      document.getElementById("card4Label").textContent = "物理能量最值 (M32)";
      document.getElementById("energyVal").textContent = ev ? `${fmtNum(ev.min_phys_energy_m32, 2)}` : "--";
      document.getElementById("energySub").textContent = ev
        ? `global_min=${fmtNum(ev.best_global_m32_energy, 2)} | align=${fmtNum(ev.zero_manifold_align, 4)} | comm=${fmtNum(ev.zero_manifold_comm, 4)}`
        : "--";

      document.getElementById("card5Label").textContent = "W0 比特准确率 / 锁存 / Pre@256";
      document.getElementById("accVal").textContent = ev ? fmtNum(ev.zero_w0_bit_acc, 4) : "--";
      document.getElementById("accSub").textContent = ev
        ? `latch_z=${fmtNum(ev.zero_latch_ratio, 3)} | pre@256=${fmtNum(ev["exact_preimage_rate@256"], 4)} | uniq=${fmtNum(ev["unique_ratio@256"], 3)}`
        : "--";

      document.getElementById("c1Title").textContent = "训练损失曲线";
      document.getElementById("c1Legend").innerHTML =
        '<span><span class="dot" style="background:var(--accent)"></span>train_loss</span>' +
        '<span><span class="dot" style="background:var(--purple)"></span>train_phys_loss</span>';
      document.getElementById("c2Title").textContent = "评估 BCE 与 物理能量 M32";
      document.getElementById("c2Legend").innerHTML =
        '<span><span class="dot" style="background:var(--green)"></span>eval_bce (虚线 ln2)</span>' +
        '<span><span class="dot" style="background:var(--amber)"></span>min_E32(右轴缩放)</span>';
      document.getElementById("c3Title").textContent = "破壁探索与锁存指标";
      document.getElementById("c3Legend").innerHTML =
        '<span><span class="dot" style="background:var(--accent)"></span>zero_w0_acc</span>' +
        '<span><span class="dot" style="background:var(--amber)"></span>latch_z</span>' +
        '<span><span class="dot" style="background:var(--green)"></span>pre@256</span>';

      renderLineChart("chartTrain", [
        {color: "#38bdf8", points: rows.map(x => ({x: x.step, y: x.train_loss}))},
        {color: "#a78bfa", points: rows.map(x => ({x: x.step, y: x.train_phys_loss}))},
      ]);
      renderLineChart("chartEval", [
        {color: "#34d399", points: evalRows.map(x => ({x: x.step, y: x.eval_bce_loss}))},
        {color: "#fbbf24", secondary: true, points: evalRows.map(x => ({x: x.step, y: x.min_phys_energy_m32}))},
      ], [{y: 0.693147, color: "#94a3b8", label: "ln 2 = 0.6931"}]);
      renderLineChart("chartProbe", [
        {color: "#38bdf8", points: evalRows.map(x => ({x: x.step, y: x.zero_w0_bit_acc}))},
        {color: "#fbbf24", points: evalRows.map(x => ({x: x.step, y: x.zero_latch_ratio}))},
        {color: "#34d399", points: evalRows.map(x => ({x: x.step, y: x["exact_preimage_rate@256"]}))},
      ], [{y: 0.5, color: "#64748b", label: "0.50"}]);

      renderRoundsBar(ev);
    }

    const logBox = document.getElementById("logBox");
    const wasBottom = Math.abs(logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight) < 30;
    logBox.textContent = (r.log_tail || []).join("\n") || "暂无日志...";
    if (wasBottom) logBox.scrollTop = logBox.scrollHeight;

    document.getElementById("procCount").textContent = `活跃进程: ${(data.processes || []).length}`;
    const ckpts = r.checkpoints || [];
    document.getElementById("ckptCount").textContent = `检查点 (${ckpts.length})`;
    document.getElementById("ckptRow").innerHTML = ckpts
      .map(c => `<span class="badge">${c.name} (${c.size_mb} MB)</span>`)
      .join("");
  }

  async function fetchStatus() {
    try {
      const resp = await fetch("api/status", {cache: "no-store"});
      const data = await resp.json();
      updateUI(data);
    } catch (err) {
      const badge = document.getElementById("runBadge");
      badge.textContent = "轮询异常，等待重连...";
      badge.className = "badge warn";
    }
  }

  document.getElementById("runSelect").addEventListener("change", e => {
    selectedRun = e.target.value;
    fetchStatus();
  });
  document.getElementById("refreshBtn").addEventListener("click", fetchStatus);
  document.getElementById("toggleAutoBtn").addEventListener("click", () => {
    autoRefresh = !autoRefresh;
    document.getElementById("toggleAutoBtn").textContent = autoRefresh ? "自动刷新: 开 (5s)" : "自动刷新: 关";
  });

  fetchStatus();
  timerId = setInterval(() => { if (autoRefresh) fetchStatus(); }, 5000);
})();
</script>
</body>
</html>
"""


def make_handler(runs_dir: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                payload = json.dumps(_collect_status(runs_dir), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-SHA V4 Colab Monitor Dashboard")
    parser.add_argument("--runs_dir", type=str, default="/content/runs")
    parser.add_argument("--tb_dir", type=str, default="/content/tb_logs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll_sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    tb_dir = Path(args.tb_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    feeder = threading.Thread(
        target=_tb_loop,
        args=(runs_dir, tb_dir, float(args.poll_sec), stop_event),
        daemon=True,
    )
    feeder.start()

    server = ThreadingHTTPServer(("0.0.0.0", int(args.port)), make_handler(runs_dir))
    print(
        f"[dashboard] serving on http://0.0.0.0:{args.port} | runs_dir={runs_dir} | tb_dir={tb_dir}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
