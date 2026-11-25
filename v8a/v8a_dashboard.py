"""Clean-room live web monitoring dashboard for Phase-SHA V8A."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse


REMOTE_SYNC_SNIPPET = """
import json, os, subprocess
from pathlib import Path
root = Path({remote_dir!r})
payload = {{"queue_status": None, "runs": {{}}, "queue_log": "", "gpu": {{}}}}
qs = root / "queue_status.json"
if qs.exists():
    payload["queue_status"] = json.loads(qs.read_text(encoding="utf-8"))
ql = root / "queue.log"
if ql.exists():
    lines = ql.read_text(encoding="utf-8", errors="replace").splitlines()
    payload["queue_log"] = "\\n".join(lines[-40:])
if root.exists():
    for d in sorted(root.iterdir()):
        if d.is_dir():
            tel = d / "telemetry.json"
            met = d / "metrics.jsonl"
            item = {{}}
            if tel.exists():
                item["telemetry"] = json.loads(tel.read_text(encoding="utf-8"))
            if met.exists():
                mlines = met.read_text(encoding="utf-8", errors="replace").splitlines()
                item["metrics"] = [json.loads(x) for x in mlines[-300:] if x.strip()]
            if item:
                payload["runs"][d.name] = item
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=3
    ).strip()
    if out:
        p = [x.strip() for x in out.splitlines()[0].split(",")]
        payload["gpu"] = {{"gpu_name": p[0], "gpu_util_pct": float(p[1]), "smi_vram_used_mb": float(p[2]), "smi_vram_total_mb": float(p[3]), "gpu_temp_c": float(p[4])}}
except Exception:
    pass
print("<<<V8A_SYNC_JSON>>>")
print(json.dumps(payload, ensure_ascii=False))
print("<<<V8A_SYNC_END>>>")
"""


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Phase-SHA V8A 环面解缠实时监控台</title>
<style>
  :root {
    --bg: #0b101b; --card: #121928; --border: #233048;
    --text: #e6edf7; --muted: #8ea0bc; --cyan: #38bdf8;
    --emerald: #34d399; --amber: #fbbf24; --rose: #fb7185; --violet: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 20px; border-bottom: 1px solid var(--border); background: #0e1524; }
  .title { font-size: 18px; font-weight: 700; letter-spacing: 0.3px; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; }
  .badge { padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--muted); }
  .badge.live { color: var(--emerald); border-color: rgba(52,211,153,0.4); }
  .badge.fp32 { color: var(--cyan); border-color: rgba(56,189,248,0.4); }
  select, button { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
  button.active { border-color: var(--cyan); color: var(--cyan); background: rgba(56,189,248,0.1); }
  .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; padding: 16px 20px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .span12 { grid-column: span 12; }
  .span6 { grid-column: span 6; }
  @media (max-width: 1050px) { .span6 { grid-column: span 12; } }
  .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 8px; flex-wrap: wrap; }
  .card-title { font-size: 14px; font-weight: 600; color: var(--text); }
  .card-sub { font-size: 12px; color: var(--muted); }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; }
  .kpi { background: #0d1422; border: 1px solid #1d283d; border-radius: 8px; padding: 10px 12px; }
  .kpi-label { font-size: 11px; color: var(--muted); }
  .kpi-val { font-size: 20px; font-weight: 700; margin-top: 3px; font-variant-numeric: tabular-nums; }
  canvas { width: 100%; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 7px 8px; border-bottom: 1px solid #1d283d; text-align: right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  tr.selected { background: rgba(56,189,248,0.08); }
  .bar-wrap { height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 6px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--cyan), var(--emerald)); transition: width 0.3s; }
  pre { margin: 0; background: #090d16; border: 1px solid #1d283d; border-radius: 8px; padding: 10px; font-size: 11px; line-height: 1.45; color: #b9c7dc; max-height: 210px; overflow: auto; white-space: pre-wrap; }
</style>
</head>
<body>
<header>
  <div>
    <div class="title">Phase-SHA V8A — 全 256 位保体积环面流解缠监控台</div>
    <div class="card-sub" id="subHeader">正在连接遥测数据...</div>
  </div>
  <div class="badges">
    <span class="badge live" id="syncBadge">SYNC: INIT</span>
    <span class="badge fp32" id="fp32Badge">TF32=False · Pure FP32</span>
    <span class="badge" id="bijBadge">Bijection err: --</span>
    <label class="card-sub">当前视图:
      <select id="runSelect"></select>
    </label>
  </div>
</header>

<div class="grid">
  <div class="card span12">
    <div class="kpis" id="kpiRow"></div>
  </div>

  <div class="card span12">
    <div class="card-head">
      <div>
        <span class="card-title">1. 256 通道圆周聚拢度天际线（W0–W7 八个 32-bit 寄存器分组）</span>
        <span class="card-sub" id="skyStats"></span>
      </div>
      <div class="badges">
        <button class="active" data-sky="R1_vec">R1 (一阶聚拢度)</button>
        <button data-sky="R2_vec">R2 (二阶谐波)</button>
        <button data-sky="R_disc_vec">R_disc (去量化关闭)</button>
        <button data-sky="per_channel_bits">单通道抽干比特</button>
      </div>
    </div>
    <canvas id="skyCanvas" height="230"></canvas>
  </div>

  <div class="card span6">
    <div class="card-head">
      <div>
        <div class="card-title">2. 净抽取冗余熵曲线（Train vs Unseen Holdout）</div>
        <div class="card-sub" id="entSub">虚线: 原始输入冗余基线 & 理论封顶 (256 - k)</div>
      </div>
    </div>
    <canvas id="entCanvas" height="230"></canvas>
  </div>

  <div class="card span6">
    <div class="card-head">
      <div>
        <div class="card-title">3. 冻结流原像破壁曲线（W Bit Acc & Exact Pass）</div>
        <div class="card-sub" id="probeSub">独立前向 SHA-256 验真</div>
      </div>
    </div>
    <canvas id="probeCanvas" height="230"></canvas>
  </div>

  <div class="card span6">
    <div class="card-head">
      <div class="card-title">V8A-T1 vs V8B-T1 vs V8B-T2 轮次相变对照总表</div>
      <div class="card-sub" id="queueMeta"></div>
    </div>
    <div style="overflow:auto; max-height:270px;">
      <table>
        <thead>
          <tr><th>Run</th><th>Arch</th><th>Phase/Mode</th><th>R</th><th>Base</th><th>Eval</th><th>Gain</th><th>ch55</th><th>w_acc</th><th>pass@1</th><th>pass@64</th><th>状态</th></tr>
        </thead>
        <tbody id="summaryBody"></tbody>
      </table>
    </div>
  </div>

  <div class="card span6">
    <div class="card-head">
      <div class="card-title">4. A100 硬件遥测、双射校验与实时日志</div>
      <div class="card-sub" id="hwMeta"></div>
    </div>
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;font-size:12px;">
        <span id="vramText">VRAM: --</span>
        <span id="utilText">GPU Util: --</span>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="vramBar" style="width:0%"></div></div>
    </div>
    <pre id="logBox">等待日志...</pre>
  </div>
</div>

<script>
let selectedRun = null;
let skyMode = "R1_vec";
let lastData = null;

document.querySelectorAll("button[data-sky]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("button[data-sky]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    skyMode = btn.dataset.sky;
    if (lastData) renderAll(lastData);
  });
});

document.getElementById("runSelect").addEventListener("change", (e) => {
  selectedRun = e.target.value;
  if (lastData) renderAll(lastData);
});

function resizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  const w = Math.max(300, Math.floor(rect.width * dpr));
  const h = Math.floor(parseInt(c.getAttribute("height"), 10) * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return { ctx: c.getContext("2d"), w, h, dpr };
}

function drawSkyline(tel) {
  const c = document.getElementById("skyCanvas");
  const { ctx, w, h, dpr } = resizeCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const padL = 42 * dpr, padR = 14 * dpr, padT = 16 * dpr, padB = 24 * dpr;
  const pw = w - padL - padR, ph = h - padT - padB;
  const vec = (tel && tel[skyMode]) || new Array(256).fill(0);
  const yMax = skyMode === "per_channel_bits" ? 1.05 : 1.0;

  // 8 register background bands
  for (let g = 0; g < 8; g++) {
    const x0 = padL + (g * 32 / 256) * pw;
    const gw = (32 / 256) * pw;
    ctx.fillStyle = g % 2 === 0 ? "rgba(255,255,255,0.02)" : "rgba(56,189,248,0.03)";
    ctx.fillRect(x0, padT, gw, ph);
    ctx.fillStyle = "#8ea0bc";
    ctx.font = `${11 * dpr}px sans-serif`;
    ctx.fillText("W" + g, x0 + gw * 0.42, h - 6 * dpr);
  }

  // Reference lines
  const refs = skyMode === "per_channel_bits"
    ? [{ v: 0.93, lbl: "0.93b (ideal box)", col: "#34d399" }]
    : [
        { v: 0.35, lbl: "0.35", col: "#fbbf24" },
        { v: 0.55, lbl: "0.55", col: "#34d399" },
        { v: 2 / Math.PI, lbl: "2/π=0.637", col: "#a78bfa" }
      ];
  refs.forEach(r => {
    const y = padT + ph * (1 - r.v / yMax);
    ctx.save();
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.strokeStyle = r.col;
    ctx.lineWidth = 1.2 * dpr;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillStyle = r.col;
    ctx.font = `${10 * dpr}px sans-serif`;
    ctx.fillText(r.lbl, padL + 4 * dpr, y - 3 * dpr);
    ctx.restore();
  });

  // Bars
  const bw = pw / 256;
  for (let i = 0; i < 256; i++) {
    const val = Math.max(0, Math.min(yMax, vec[i] || 0));
    const bh = (val / yMax) * ph;
    const x = padL + i * bw;
    const y = padT + ph - bh;
    if (skyMode === "per_channel_bits") {
      ctx.fillStyle = val > 0.7 ? "#34d399" : (val > 0.35 ? "#38bdf8" : "#475569");
    } else {
      ctx.fillStyle = val > 0.55 ? "#34d399" : (val > 0.35 ? "#fbbf24" : "#38bdf8");
    }
    ctx.fillRect(x, y, Math.max(1, bw - 0.5 * dpr), bh);
  }
  ctx.fillStyle = "#8ea0bc";
  ctx.font = `${10 * dpr}px sans-serif`;
  ctx.fillText(yMax.toFixed(1), 6 * dpr, padT + 10 * dpr);
  ctx.fillText("0.0", 10 * dpr, padT + ph);
}

function drawLines(canvasId, series, opts) {
  const c = document.getElementById(canvasId);
  const { ctx, w, h, dpr } = resizeCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const padL = 44 * dpr, padR = 14 * dpr, padT = 16 * dpr, padB = 24 * dpr;
  const pw = w - padL - padR, ph = h - padT - padB;

  const xs = opts.steps;
  if (!xs || xs.length === 0) {
    ctx.fillStyle = "#8ea0bc";
    ctx.font = `${12 * dpr}px sans-serif`;
    ctx.fillText("等待首个评估节点数据...", padL + 10 * dpr, padT + ph / 2);
    return;
  }
  const xMin = xs[0], xMax = Math.max(xs[xs.length - 1], xMin + 1);
  const yMin = opts.yMin, yMax = opts.yMax;

  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1 * dpr;
  for (let k = 0; k <= 4; k++) {
    const yy = padT + (k / 4) * ph;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    const val = yMax - (k / 4) * (yMax - yMin);
    ctx.fillStyle = "#8ea0bc";
    ctx.font = `${10 * dpr}px sans-serif`;
    ctx.fillText(val.toFixed(opts.yDec || 0), 4 * dpr, yy + 3 * dpr);
  }

  (opts.hlines || []).forEach(hl => {
    const yy = padT + ph * (1 - (hl.v - yMin) / (yMax - yMin));
    if (yy >= padT && yy <= padT + ph) {
      ctx.save();
      ctx.setLineDash([5 * dpr, 4 * dpr]);
      ctx.strokeStyle = hl.col;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      ctx.fillStyle = hl.col;
      ctx.font = `${10 * dpr}px sans-serif`;
      ctx.fillText(hl.lbl, padL + 6 * dpr, yy - 3 * dpr);
      ctx.restore();
    }
  });

  if (opts.vMarker && opts.vMarker > xMin && opts.vMarker < xMax) {
    const vx = padL + ((opts.vMarker - xMin) / (xMax - xMin)) * pw;
    ctx.save();
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.strokeStyle = "#a78bfa";
    ctx.beginPath(); ctx.moveTo(vx, padT); ctx.lineTo(vx, padT + ph); ctx.stroke();
    ctx.fillStyle = "#a78bfa";
    ctx.fillText("Phase1->2", vx + 4 * dpr, padT + 12 * dpr);
    ctx.restore();
  }

  series.forEach(s => {
    ctx.strokeStyle = s.col;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    s.vals.forEach((v, idx) => {
      const px = padL + ((xs[idx] - xMin) / (xMax - xMin)) * pw;
      const py = padT + ph * (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin));
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });

  // Legend
  let lx = padL;
  series.forEach(s => {
    ctx.fillStyle = s.col;
    ctx.fillRect(lx, h - 12 * dpr, 10 * dpr, 3 * dpr);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = `${10 * dpr}px sans-serif`;
    ctx.fillText(s.lbl, lx + 14 * dpr, h - 8 * dpr);
    lx += (s.lbl.length * 7 + 34) * dpr;
  });
}

function renderAll(data) {
  lastData = data;
  const runs = data.runs || {};
  const names = Object.keys(runs);
  const qs = data.queue_status || {};
  const activeRun = qs.current_run && runs[qs.current_run] ? qs.current_run : (names[names.length - 1] || "");
  if (!selectedRun || !runs[selectedRun]) selectedRun = activeRun;

  const sel = document.getElementById("runSelect");
  const prevOpts = Array.from(sel.options).map(o => o.value).join(",");
  if (prevOpts !== names.join(",")) {
    sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("");
  }
  if (selectedRun) sel.value = selectedRun;

  document.getElementById("syncBadge").textContent = data.source_mode || "LOCAL";
  document.getElementById("subHeader").textContent =
    `队列状态: ${qs.status || "idle"} · 已完成 ${ (qs.completed_runs || []).length }/${ (qs.planned_runs || []).length } 个任务` +
    (qs.critical_round ? ` · 自动探测临界轮 R_c = ${qs.critical_round}` : "");

  const rObj = runs[selectedRun] || {};
  const tel = rObj.telemetry || {};
  const metrics = rObj.metrics || [];
  const probe = tel.probe || {};

  const bij = tel.bijection_max_err !== undefined ? tel.bijection_max_err.toExponential(2) : "--";
  document.getElementById("bijBadge").textContent = `Bijection err: ${bij}`;
  const gpu = tel.gpu || data.gpu || {};
  if (gpu.tf32_matmul === false && gpu.tf32_cudnn === false) {
    document.getElementById("fp32Badge").textContent = "TF32=False · Pure FP32 ✓";
  }

  const kpis = [
    { l: "当前轮数 / 架构", v: tel.rounds ? `R=${tel.rounds} (${tel.arch || "v8a"})` : "--" },
    { l: "阶段 / Step", v: tel.total_steps ? `${tel.phase === "Phase-1 (round_transport)" ? "P1" : (tel.phase === "Phase-2 (e2e_unsupervised)" ? "P2" : "T1")} · ${tel.step}/${tel.total_steps}` : "--" },
    { l: "留出集抽取冗余 (bits)", v: tel.eval_extracted_bits !== undefined ? `${tel.eval_extracted_bits.toFixed(2)} / ${tel.theory_max_bits || 224}` : "--" },
    { l: "净流解缠增益", v: tel.flow_gain_bits !== undefined ? `${tel.flow_gain_bits >= 0 ? "+" : ""}${tel.flow_gain_bits.toFixed(2)} b` : "--" },
    { l: "坍缩通道 ch55 / ch35", v: tel.collapsed_ch_55 !== undefined ? `${tel.collapsed_ch_55} / ${tel.collapsed_ch_35}` : "--" },
    { l: "原像位准确率 (z)", v: probe.w_bit_acc !== undefined ? `${(probe.w_bit_acc * 100).toFixed(2)}% (z=${probe.z_score.toFixed(1)})` : "--" },
    { l: "独立前向 pass@1 / @64", v: probe.exact_pass_1 !== undefined ? `${(probe.exact_pass_1 * 100).toFixed(2)}% / ${(probe.exact_pass_64 * 100).toFixed(2)}%` : "--" },
    { l: "输运保真 tf / roll", v: tel.mode === "round_transport" && tel.tf_bit_acc !== undefined ? `${(tel.tf_bit_acc * 100).toFixed(1)}% / ${(tel.rollout_bit_acc * 100).toFixed(1)}%` : (tel.param_count ? `${(tel.param_count / 1e6).toFixed(1)}M · ${tel.steps_per_sec || 0} it/s` : "--") },
  ];
  document.getElementById("kpiRow").innerHTML = kpis.map(k =>
    `<div class="kpi"><div class="kpi-label">${k.l}</div><div class="kpi-val">${k.v}</div></div>`
  ).join("");

  document.getElementById("skyStats").textContent = tel.max_R1 !== undefined
    ? `max_R1=${tel.max_R1.toFixed(3)} · max_R2=${tel.max_R2.toFixed(3)} · disc90=${tel.collapsed_ch_disc_90}`
    : "";
  drawSkyline(tel);

  const steps = metrics.map(m => m.step);
  drawLines("entCanvas", [
    { lbl: "Train bits", col: "#38bdf8", vals: metrics.map(m => m.train_extracted_bits) },
    { lbl: "Eval Holdout bits", col: "#34d399", vals: metrics.map(m => m.eval_extracted_bits) },
  ], {
    steps, yMin: 0, yMax: Math.max(230, (tel.theory_max_bits || 224) + 8), yDec: 0,
    hlines: [
      { v: tel.baseline_input_bits || 0, lbl: `Base ${(tel.baseline_input_bits || 0).toFixed(1)}b`, col: "#fbbf24" },
      { v: tel.theory_max_bits || 224, lbl: `Ceiling ${tel.theory_max_bits || 224}b`, col: "#fb7185" },
    ],
    vMarker: tel.mode === "round_transport" ? tel.transport_steps : null,
  });

  drawLines("probeCanvas", [
    { lbl: "w_bit_acc", col: "#38bdf8", vals: metrics.map(m => m.w_bit_acc || 0.5) },
    { lbl: "exact_pass@1", col: "#34d399", vals: metrics.map(m => m.exact_pass_1 || 0.0) },
    { lbl: "exact_pass@64", col: "#a78bfa", vals: metrics.map(m => m.exact_pass_64 || 0.0) },
  ], {
    steps, yMin: 0, yMax: 1.02, yDec: 2,
    hlines: [{ v: 0.5, lbl: "Random 0.50", col: "#64748b" }],
    vMarker: tel.mode === "round_transport" ? tel.transport_steps : null,
  });

  // Summary table across all discovered runs
  const rowsHtml = names.map(n => {
    const rt = (runs[n] && runs[n].telemetry) || {};
    const rp = rt.probe || {};
    const cls = n === selectedRun ? "selected" : "";
    const phaseText = rt.phase && rt.phase !== "completed" ? rt.phase : (rt.mode || "--");
    return `<tr class="${cls}" onclick="selectedRun='${n}'; renderAll(lastData);" style="cursor:pointer">
      <td>${n}</td>
      <td>${rt.arch || "v8a"}</td>
      <td>${phaseText}</td>
      <td>${rt.rounds || "--"}</td>
      <td>${rt.baseline_input_bits !== undefined ? rt.baseline_input_bits.toFixed(1) : "--"}</td>
      <td>${rt.eval_extracted_bits !== undefined ? rt.eval_extracted_bits.toFixed(1) : "--"}</td>
      <td>${rt.flow_gain_bits !== undefined ? (rt.flow_gain_bits >= 0 ? "+" : "") + rt.flow_gain_bits.toFixed(1) : "--"}</td>
      <td>${rt.collapsed_ch_55 !== undefined ? rt.collapsed_ch_55 : "--"}</td>
      <td>${rp.w_bit_acc !== undefined ? rp.w_bit_acc.toFixed(4) : "--"}</td>
      <td>${rp.exact_pass_1 !== undefined ? rp.exact_pass_1.toFixed(4) : "--"}</td>
      <td>${rp.exact_pass_64 !== undefined ? rp.exact_pass_64.toFixed(4) : "--"}</td>
      <td>${rt.status || "--"}</td>
    </tr>`;
  }).join("");
  document.getElementById("summaryBody").innerHTML = rowsHtml;

  const usedMb = gpu.smi_vram_used_mb || (gpu.vram_alloc_gb ? gpu.vram_alloc_gb * 1024 : 0);
  const totMb = gpu.smi_vram_total_mb || (gpu.vram_total_gb ? gpu.vram_total_gb * 1024 : 40960);
  const pct = totMb > 0 ? Math.min(100, (usedMb / totMb) * 100) : 0;
  document.getElementById("hwMeta").textContent = `${gpu.gpu_name || "CUDA"} · ${gpu.gpu_temp_c ? gpu.gpu_temp_c + "°C" : ""}`;
  document.getElementById("vramText").textContent = `VRAM: ${(usedMb / 1024).toFixed(2)} / ${(totMb / 1024).toFixed(1)} GB (${pct.toFixed(1)}%)`;
  document.getElementById("utilText").textContent = `GPU Util: ${gpu.gpu_util_pct !== undefined ? gpu.gpu_util_pct + "%" : "--"} · ETA: ${tel.eta_sec ? Math.round(tel.eta_sec) + "s" : "--"}`;
  document.getElementById("vramBar").style.width = `${pct.toFixed(1)}%`;

  const logs = (tel.log_tail && tel.log_tail.length) ? tel.log_tail.join("\n") : (data.queue_log || "暂无日志");
  document.getElementById("logBox").textContent = logs;
}

async function poll() {
  try {
    const res = await fetch("/api/state");
    if (res.ok) renderAll(await res.json());
  } catch (e) {}
}
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


class DashboardState:
    def __init__(
        self,
        local_dir: Path,
        colab_session: str,
        colab_exe: str,
        remote_dir: str,
        poll_sec: float,
    ) -> None:
        self.local_dir = local_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.colab_session = colab_session.strip()
        self.colab_exe = colab_exe
        self.remote_dir = remote_dir
        self.poll_sec = float(poll_sec)
        self.lock = threading.Lock()
        self.source_mode = "LOCAL"
        self.remote_gpu: Dict[str, Any] = {}
        self.queue_log: str = ""
        self.stop_event = threading.Event()

    def _sync_once(self) -> None:
        if not self.colab_session:
            return
        snippet = REMOTE_SYNC_SNIPPET.format(remote_dir=self.remote_dir)
        try:
            proc = subprocess.run(
                [self.colab_exe, "exec", "-s", self.colab_session, "--timeout", "18"],
                input=snippet,
                capture_output=True,
                text=True,
                timeout=25,
            )
            out = proc.stdout or ""
            start_tok = "<<<V8A_SYNC_JSON>>>"
            end_tok = "<<<V8A_SYNC_END>>>"
            if start_tok not in out or end_tok not in out:
                with self.lock:
                    self.source_mode = "LOCAL_MIRROR (Colab offline/idle)"
                return
            raw_json = out.split(start_tok, 1)[1].split(end_tok, 1)[0].strip()
            data = json.loads(raw_json)
            if data.get("queue_status") is not None:
                _atomic_write(self.local_dir / "queue_status.json", json.dumps(data["queue_status"], ensure_ascii=False, indent=2))
            if data.get("queue_log"):
                _atomic_write(self.local_dir / "queue.log", data["queue_log"])
            for rname, ritem in (data.get("runs") or {}).items():
                rdir = self.local_dir / rname
                rdir.mkdir(parents=True, exist_ok=True)
                if "telemetry" in ritem:
                    _atomic_write(rdir / "telemetry.json", json.dumps(ritem["telemetry"], ensure_ascii=False, indent=2))
                if "metrics" in ritem:
                    lines = "\n".join(json.dumps(m, ensure_ascii=False) for m in ritem["metrics"]) + "\n"
                    _atomic_write(rdir / "metrics.jsonl", lines)
            with self.lock:
                self.source_mode = f"LIVE_A100 ({self.colab_session})"
                self.remote_gpu = data.get("gpu") or {}
                self.queue_log = data.get("queue_log") or ""
        except Exception:
            with self.lock:
                self.source_mode = "LOCAL_MIRROR"

    def start_sync_thread(self) -> None:
        if not self.colab_session:
            return

        def _loop() -> None:
            while not self.stop_event.is_set():
                self._sync_once()
                self.stop_event.wait(self.poll_sec)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def build_api_state(self) -> Dict[str, Any]:
        qs_path = self.local_dir / "queue_status.json"
        queue_status = None
        if qs_path.exists():
            try:
                queue_status = json.loads(qs_path.read_text(encoding="utf-8"))
            except Exception:
                queue_status = None

        runs: Dict[str, Any] = {}
        for d in sorted(self.local_dir.iterdir()):
            if not d.is_dir():
                continue
            item: Dict[str, Any] = {}
            tel_p = d / "telemetry.json"
            met_p = d / "metrics.jsonl"
            if tel_p.exists():
                try:
                    item["telemetry"] = json.loads(tel_p.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if met_p.exists():
                try:
                    item["metrics"] = [
                        json.loads(line)
                        for line in met_p.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip()
                    ]
                except Exception:
                    pass
            if item:
                runs[d.name] = item

        qlog_p = self.local_dir / "queue.log"
        qlog = self.queue_log
        if not qlog and qlog_p.exists():
            try:
                qlog = "\n".join(qlog_p.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            except Exception:
                qlog = ""

        with self.lock:
            return {
                "source_mode": self.source_mode,
                "queue_status": queue_status,
                "runs": runs,
                "gpu": self.remote_gpu,
                "queue_log": qlog,
            }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                body = json.dumps(state.build_api_state(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-SHA V8A Web Monitoring Dashboard")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--local_dir", type=str, default=str(Path(__file__).resolve().parent / "colab_runs"))
    p.add_argument("--colab_session", type=str, default="")
    p.add_argument("--colab_exe", type=str, default=r"g:\SHA-LM\.colab_venv\Scripts\colab.exe")
    p.add_argument("--remote_dir", type=str, default="/content/v8a_runs")
    p.add_argument("--poll_sec", type=float, default=10.0)
    args = p.parse_args()

    state = DashboardState(
        local_dir=Path(args.local_dir),
        colab_session=args.colab_session,
        colab_exe=args.colab_exe,
        remote_dir=args.remote_dir,
        poll_sec=args.poll_sec,
    )
    state.start_sync_thread()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"[V8A Dashboard] Listening at http://127.0.0.1:{args.port} (mirror: {state.local_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
