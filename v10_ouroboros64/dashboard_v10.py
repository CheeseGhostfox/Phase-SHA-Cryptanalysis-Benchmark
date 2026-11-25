"""
Phase-SHA V10「衔尾蛇-64」8789 端口实时遥测战术看板（纯简体中文、零外部 CDN 依赖）。

严格履行双域物理评测隔离契约：
  - 左侧域 A：64×9 时空晶体缺陷热力图 + Neural Wolff vs 1-bit WalkSAT 跨垒愈合曲线；
  - 右侧域 B：零泄漏盲求逆隔离榜（仅限 S0=IV 与 SR=H-IV，绝不混淆）；
  - 下方域 C：Tensor-BP 纠缠熵 S_ent 演化、GRPO 隧穿奖励轨迹、FP32 精度锁定与实时日志尾流。
支持本地 `--local_dir` 与远端 `--remote_url` 双模式轮询。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_EXE = REPO_ROOT / ".colab_venv" / "Scripts" / "colab.exe"
STATE_LOCK = threading.Lock()
LIVE_STATE: Dict[str, Any] = {
    "meta": {"phase": "WAIT", "step": 0, "total_steps": 0, "status": "waiting"},
    "poll_source": "INIT",
    "poll_error": "",
}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Phase-SHA V10「衔尾蛇-64」实时遥测控制台</title>
  <style>
    :root {
      --bg: #080d14;
      --panel: #0f1724;
      --border: #1e2d45;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --cyan: #22d3ee;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #a78bfa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 14px 18px;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      font-size: 13px;
    }
    .topbar {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 14px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 10px;
      margin-bottom: 12px; flex-wrap: wrap; gap: 10px;
    }
    .title { font-size: 16px; font-weight: 700; color: var(--cyan); letter-spacing: 0.4px; }
    .badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .badge {
      padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
      border: 1px solid var(--border); background: #142033; color: var(--text);
    }
    .badge.ok { border-color: #059669; color: #6ee7b7; background: #062c22; }
    .badge.warn { border-color: #d97706; color: #fcd34d; background: #2d1e07; }
    .badge.iso { border-color: #7c3aed; color: #ddd6fe; background: #22143d; }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.15fr 1.1fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .grid-bottom {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
    }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; padding: 12px; display: flex; flex-direction: column;
    }
    .card-hdr {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px; border-bottom: 1px solid var(--border); padding-bottom: 6px;
    }
    .card-title { font-weight: 700; font-size: 13.5px; color: #f8fafc; }
    .card-sub { font-size: 11px; color: var(--muted); }

    .kpis {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px;
    }
    .kpi {
      background: #0a111c; border: 1px solid #18253a; border-radius: 8px; padding: 7px 9px;
    }
    .kpi-lbl { font-size: 10.5px; color: var(--muted); }
    .kpi-val { font-size: 15px; font-weight: 700; margin-top: 2px; font-family: ui-monospace, monospace; }

    /* 64x9 时空晶体热力图 */
    .hm-wrap {
      display: flex; flex-direction: column; gap: 3px;
    }
    .hm-weld {
      font-size: 11px; font-family: ui-monospace, monospace; padding: 4px 8px;
      border-radius: 6px; text-align: center; font-weight: 700;
      background: #062c22; border: 1px solid #10b981; color: #6ee7b7;
    }
    .hm-cols {
      display: grid; grid-template-columns: 28px repeat(9, 1fr); gap: 2px;
      font-size: 10px; color: var(--muted); text-align: center; font-family: ui-monospace, monospace;
    }
    .hm-rows {
      display: flex; flex-direction: column; gap: 1.5px; max-height: 336px; overflow-y: auto;
    }
    .hm-row {
      display: grid; grid-template-columns: 28px repeat(9, 1fr); gap: 2px; align-items: center;
    }
    .hm-ridx { font-size: 9.5px; color: var(--muted); text-align: right; padding-right: 4px; font-family: ui-monospace, monospace; }
    .hm-cell {
      height: 5px; border-radius: 1.5px; background: #132238;
    }

    table { width: 100%; border-collapse: collapse; font-size: 11.5px; margin-top: 6px; }
    th, td {
      border-bottom: 1px solid #1a273d; padding: 5px 6px; text-align: right;
      font-family: ui-monospace, monospace;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; font-size: 11px; }

    .logbox {
      background: #060a10; border: 1px solid #18253a; border-radius: 8px;
      padding: 8px 10px; height: 220px; overflow-y: auto;
      font-family: Consolas, ui-monospace, monospace; font-size: 11px; line-height: 1.45;
      color: #cbd5e1; white-space: pre-wrap;
    }
    .svg-box { width: 100%; height: 185px; background: #09101a; border: 1px solid #18253a; border-radius: 8px; }
    .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); margin-top: 5px; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <div class="title">Phase-SHA V10「衔尾蛇-64」边界焊死三维时空晶体求解控制台</div>
      <div class="card-sub" id="subtitle">等待遥测数据接入...</div>
    </div>
    <div class="badges" id="badges">
      <span class="badge ok">纯 FP32 锁死 (TF32=OFF)</span>
      <span class="badge iso">双评测域物理绝对隔离</span>
      <span class="badge" id="statusBadge">STATUS: --</span>
    </div>
  </div>

  <div class="grid">
    <!-- 面板 1: 64x9 时空晶体缺陷热力图 -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">1. 64×9 时空晶体实时缺陷热力图</span>
        <span class="card-sub" id="hmEnergy">E_defect = --</span>
      </div>
      <div class="hm-wrap">
        <div class="hm-weld" id="weldTop">顶部焊死边界 S[0] ≡ IV（不可变）</div>
        <div class="hm-cols">
          <span>t</span><span>W</span><span>a</span><span>b</span><span>c</span><span>d</span><span>e</span><span>f</span><span>g</span><span>h</span>
        </div>
        <div class="hm-rows" id="hmGrid"></div>
        <div class="hm-weld" id="weldBot">底部焊死边界 S[64] ≡ (H - IV) mod 2^32（不可变）</div>
      </div>
    </div>

    <!-- 面板 2: 评测域 A - 缺陷愈合与 Wolff 跨垒曲线 -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">2. Wolff vs WalkSAT 缺陷愈合曲线</span>
        <span class="badge warn">域A：已知内部缺陷（非盲求逆）</span>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-lbl">初始缺陷能量</div><div class="kpi-val" id="healE0">--</div></div>
        <div class="kpi"><div class="kpi-lbl">1-bit WalkSAT</div><div class="kpi-val" style="color:var(--amber)" id="healWs">--</div></div>
        <div class="kpi"><div class="kpi-lbl">V10 Wolff 终能</div><div class="kpi-val" style="color:var(--green)" id="healWf">--</div></div>
        <div class="kpi"><div class="kpi-lbl">协同跨垒隧穿率</div><div class="kpi-val" style="color:var(--cyan)" id="healTun">--</div></div>
      </div>
      <svg class="svg-box" id="svgHealing" viewBox="0 0 460 185"></svg>
      <div class="legend">
        <span><i class="dot" style="background:var(--amber)"></i>1-bit WalkSAT（单比特基线）</span>
        <span><i class="dot" style="background:var(--green)"></i>V10 Neural Wolff（协同团簇）</span>
      </div>
    </div>

    <!-- 面板 3: 评测域 B - 严格盲求逆隔离榜 -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">3. 盲求逆严格隔离评测榜</span>
        <span class="badge iso">域B：零内部泄漏（verify_preimage核验）</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>目标基准</th><th>自由位</th><th>纯 CaDiCaL</th><th>V10 引导 CaDiCaL</th><th>冲突对比</th>
          </tr>
        </thead>
        <tbody id="cdclBody">
          <tr><td colspan="5" style="text-align:center;color:var(--muted)">暂无盲求逆数据</td></tr>
        </tbody>
      </table>
      <div style="margin-top:12px;border-top:1px solid var(--border);padding-top:8px">
        <div class="card-title" style="font-size:12.5px;margin-bottom:6px">原生 R=64（512-bit 全盲）双端焊死能量演化</div>
        <div class="kpis" style="grid-template-columns: repeat(2, 1fr);">
          <div class="kpi"><div class="kpi-lbl">纯盲 1-bit WalkSAT 终能</div><div class="kpi-val" id="blindWs64">--</div></div>
          <div class="kpi"><div class="kpi-lbl">纯盲 Neural Wolff 终能</div><div class="kpi-val" id="blindWf64">--</div></div>
        </div>
        <div class="card-sub" id="blindSat64" style="color:var(--amber)">R=64 盲精确原像破解率（Exact SAT）：0.0%</div>
      </div>
    </div>
  </div>

  <div class="grid-bottom">
    <!-- 面板 4A: Tensor-BP 纠缠熵、Loss 与 GRPO 奖励轨迹 -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">4. Tensor-BP 纠缠熵 S_ent 与训练 / GRPO 奖励轨迹</span>
        <span class="card-sub" id="bpStats">S_ent_carry=-- | S_ent_time=-- | VRAM=--</span>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-lbl">当前阶段/步数</div><div class="kpi-val" id="kpiStep">--</div></div>
        <div class="kpi"><div class="kpi-lbl">进位键熵 S_ent(c)</div><div class="kpi-val" style="color:var(--purple)" id="kpiEntC">--</div></div>
        <div class="kpi"><div class="kpi-lbl">Wolff 降能命中</div><div class="kpi-val" style="color:var(--green)" id="kpiHit">--</div></div>
        <div class="kpi"><div class="kpi-lbl">GRPO 组均值奖励</div><div class="kpi-val" style="color:var(--cyan)" id="kpiRew">--</div></div>
      </div>
      <svg class="svg-box" id="svgTrain" viewBox="0 0 540 165"></svg>
      <div class="legend">
        <span><i class="dot" style="background:var(--cyan)"></i>总损失 Total Loss</span>
        <span><i class="dot" style="background:var(--purple)"></i>进位纠缠熵 S_ent(carry)</span>
        <span><i class="dot" style="background:var(--green)"></i>Wolff 命中率 / GRPO 奖励</span>
      </div>
    </div>

    <!-- 面板 4B: 实时训练日志流 -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">5. 零静默实时训练遥测日志流</span>
        <span class="card-sub" id="lastUpdate">--</span>
      </div>
      <div class="logbox" id="logBox">正在连接后端 /api/state ...</div>
    </div>
  </div>

  <script>
    function drawPolyline(svgEl, seriesList, width, height) {
      if (!svgEl) return;
      const padL = 38, padR = 12, padT = 12, padB = 22;
      const w = width - padL - padR, h = height - padT - padB;
      let allVals = [];
      seriesList.forEach(s => s.vals.forEach(v => { if (v !== null && Number.isFinite(v)) allVals.push(v); }));
      if (allVals.length === 0) {
        svgEl.innerHTML = `<text x="${width/2}" y="${height/2}" fill="#64748b" font-size="12" text-anchor="middle">等待轨迹数据...</text>`;
        return;
      }
      let minV = Math.min(...allVals), maxV = Math.max(...allVals);
      if (Math.abs(maxV - minV) < 1e-6) { maxV = minV + 1; minV = Math.max(0, minV - 1); }

      let parts = [];
      // 网格
      for (let i = 0; i <= 4; i++) {
        const y = padT + (h * i / 4);
        const val = maxV - (maxV - minV) * (i / 4);
        parts.push(`<line x1="${padL}" y1="${y}" x2="${padL+w}" y2="${y}" stroke="#192538" stroke-width="1"/>`);
        parts.push(`<text x="${padL-5}" y="${y+3}" fill="#64748b" font-size="9.5" text-anchor="end" font-family="monospace">${val.toFixed(1)}</text>`);
      }
      seriesList.forEach(s => {
        const pts = [];
        const n = s.vals.length;
        for (let i = 0; i < n; i++) {
          const v = s.vals[i];
          if (v === null || !Number.isFinite(v)) continue;
          const x = padL + (n <= 1 ? w / 2 : (i / (n - 1)) * w);
          const y = padT + (1.0 - (v - minV) / (maxV - minV)) * h;
          pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        }
        if (pts.length >= 2) {
          parts.push(`<polyline fill="none" stroke="${s.color}" stroke-width="2.2" points="${pts.join(' ')}"/>`);
        } else if (pts.length === 1) {
          const [cx, cy] = pts[0].split(',');
          parts.push(`<circle cx="${cx}" cy="${cy}" r="3.5" fill="${s.color}"/>`);
        }
      });
      svgEl.innerHTML = parts.join("");
    }

    function cellColor(val) {
      if (val <= 0) return "#102033";
      if (val <= 2) return "#1d4ed8";
      if (val <= 6) return "#0891b2";
      if (val <= 12) return "#d97706";
      return "#dc2626";
    }

    function renderHeatmap(hm) {
      if (!hm || !hm.matrix) return;
      document.getElementById("hmEnergy").textContent = `快照总能量 E = ${hm.E_total}`;
      document.getElementById("weldTop").textContent = `顶部焊死边界 S[0] ≡ IV（${hm.welded_S0_ok ? "校验锁定OK" : "告警!"}）`;
      document.getElementById("weldBot").textContent = `底部焊死边界 S[${hm.rounds}] ≡ H-IV（${hm.welded_SR_ok ? "校验锁定OK" : "告警!"}）`;
      const grid = document.getElementById("hmGrid");
      const rows = [];
      for (let t = 0; t < hm.matrix.length; t++) {
        const r = hm.matrix[t];
        const cells = r.map(v => `<div class="hm-cell" title="t=${t}, err_bits=${v}" style="background:${cellColor(v)}"></div>`).join("");
        rows.push(`<div class="hm-row"><span class="hm-ridx">${t}</span>${cells}</div>`);
      }
      grid.innerHTML = rows.join("");
    }

    function render(data) {
      if (!data || !data.meta) return;
      const m = data.meta;
      document.getElementById("subtitle").textContent =
        `预设=${m.preset} | 设备=${m.gpu_name || m.device} | 参数量=${(m.n_params||0).toLocaleString()} | 轮数 R=${m.rounds} | 耗时=${m.elapsed_sec}s`;
      const sb = document.getElementById("statusBadge");
      sb.textContent = `${m.phase} (${m.step}/${m.total_steps})`;
      sb.className = "badge " + (m.status === "completed" ? "ok" : "warn");

      renderHeatmap(data.crystal_heatmap);

      // 域 A: 缺陷愈合
      const heal = data.defect_healing_eval && data.defect_healing_eval.latest;
      if (heal) {
        document.getElementById("healE0").textContent = heal.initial_energy_mean.toFixed(1);
        document.getElementById("healWs").textContent =
          `${heal.walksat_1bit.final_energy_mean.toFixed(1)} (-${heal.walksat_1bit.reduction_pct.toFixed(0)}%)`;
        document.getElementById("healWf").textContent =
          `${heal.neural_wolff.final_energy_mean.toFixed(1)} (-${heal.neural_wolff.reduction_pct.toFixed(0)}%)`;
        document.getElementById("healTun").textContent =
          `${(heal.neural_wolff.tunneling_ratio * 100).toFixed(1)}% (${heal.neural_wolff.tunneling_events}次)`;
        drawPolyline(
          document.getElementById("svgHealing"),
          [
            { vals: heal.walksat_1bit.energy_curve || [], color: "#f59e0b" },
            { vals: heal.neural_wolff.energy_curve || [], color: "#10b981" },
          ],
          460, 185
        );
      }

      // 域 B: 零泄漏盲求逆
      const blind = data.blind_inversion_eval && data.blind_inversion_eval.latest;
      if (blind) {
        const rows = (blind.cdcl_benchmarks || []).map(row => {
          const p = row.pure_cadical, g = row.v10_guided_cadical;
          const diff = g.mean_conflicts - p.mean_conflicts;
          const diffTxt = diff === 0 ? "持平" : (diff < 0 ? `${diff.toFixed(0)} 优` : `+${diff.toFixed(0)}`);
          const diffCol = diff <= 0 ? "#6ee7b7" : "#fca5a5";
          const modeTag = row.pos_mode ? ` (${row.pos_mode})` : "";
          return `<tr>
            <td>R=${row.rounds}</td>
            <td>${row.free_bits}b${modeTag}</td>
            <td>${p.sat}/${row.cases} (${p.mean_conflicts.toFixed(0)}冲突)</td>
            <td>${g.sat}/${row.cases} (${g.mean_conflicts.toFixed(0)}冲突)</td>
            <td style="color:${diffCol}">${diffTxt}</td>
          </tr>`;
        });
        document.getElementById("cdclBody").innerHTML = rows.join("");
        const r64 = blind.r64_full_blind || {};
        if (r64.initial_energy_mean !== undefined) {
          document.getElementById("blindWs64").textContent = `${r64.walksat_1bit_final_energy.toFixed(1)} (初始 ${r64.initial_energy_mean.toFixed(0)})`;
          document.getElementById("blindWf64").textContent = `${r64.neural_wolff_final_energy.toFixed(1)} (初始 ${r64.initial_energy_mean.toFixed(0)})`;
          document.getElementById("blindSat64").textContent =
            `R=64 盲精确原像破解率：WalkSAT=${(r64.walksat_exact_sat_rate*100).toFixed(1)}% | Neural Wolff=${(r64.neural_wolff_exact_sat_rate*100).toFixed(1)}%（严格 verify_preimage 核验）`;
        }
      }

      // 域 C: Tensor-BP 与训练轨迹
      const hist = data.history || [];
      const lastH = hist.length > 0 ? hist[hist.length - 1] : {};
      document.getElementById("kpiStep").textContent = `${m.step}/${m.total_steps}`;
      document.getElementById("kpiEntC").textContent = lastH.s_ent_carry !== undefined ? Number(lastH.s_ent_carry).toFixed(3) : "--";
      document.getElementById("kpiHit").textContent = (lastH.wolff_hit_rate !== null && lastH.wolff_hit_rate !== undefined) ? `${(lastH.wolff_hit_rate * 100).toFixed(1)}%` : "--";
      document.getElementById("kpiRew").textContent = (lastH.grpo_reward !== null && lastH.grpo_reward !== undefined) ? Number(lastH.grpo_reward).toFixed(2) : "--";
      document.getElementById("bpStats").textContent =
        `S_ent_carry=${lastH.s_ent_carry ?? "--"} | S_ent_time=${lastH.s_ent_time ?? "--"} | 峰值VRAM=${m.vram_peak_mb} MB`;

      drawPolyline(
        document.getElementById("svgTrain"),
        [
          { vals: hist.map(x => x.total_loss), color: "#22d3ee" },
          { vals: hist.map(x => x.s_ent_carry), color: "#a78bfa" },
          { vals: hist.map(x => x.grpo_reward !== null && x.grpo_reward !== undefined ? x.grpo_reward : x.wolff_hit_rate), color: "#10b981" },
        ],
        540, 165
      );

      document.getElementById("logBox").textContent = (data.log_tail || []).join("\n");
      document.getElementById("lastUpdate").textContent = `最新步 ${m.step} | 已运行 ${m.elapsed_sec}s`;
    }

    async function poll() {
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        if (res.ok) render(await res.json());
      } catch (e) {}
    }
    poll();
    setInterval(poll, 2500);
  </script>
</body>
</html>
"""


def _poll_colab_once(session_name: str, remote_dir: str, cache_dir: Path) -> None:
    remote_script = f"""
import json, subprocess
from pathlib import Path
root = Path({remote_dir!r})
tel_p = root / "telemetry_v10.json"
tel = json.loads(tel_p.read_text(encoding="utf-8")) if tel_p.exists() else None
log_p = root.parent / "train_v10.log"
tail = subprocess.getoutput(f"tail -n 45 {{str(log_p)}}").splitlines() if log_p.exists() else []
print("===V10_TELEM_START===")
print(json.dumps({{"telemetry": tel, "log_tail": tail}}, ensure_ascii=False))
print("===V10_TELEM_END===")
"""
    try:
        res = subprocess.run(
            [str(COLAB_EXE), "exec", "-s", session_name, "--timeout", "20"],
            input=remote_script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        txt = res.stdout or ""
        s_tag, e_tag = "===V10_TELEM_START===", "===V10_TELEM_END==="
        if s_tag in txt and e_tag in txt:
            raw_json = txt.split(s_tag, 1)[1].split(e_tag, 1)[0].strip()
            parsed = json.loads(raw_json)
            tel = parsed.get("telemetry")
            if isinstance(tel, dict):
                if parsed.get("log_tail"):
                    tel["log_tail"] = parsed["log_tail"]
                tel["poll_source"] = f"COLAB:{session_name}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "telemetry_v10.json").write_text(
                    json.dumps(tel, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with STATE_LOCK:
                    LIVE_STATE.clear()
                    LIVE_STATE.update(tel)
        else:
            with STATE_LOCK:
                LIVE_STATE["poll_error"] = (res.stderr or "empty response")[:160]
    except Exception as exc:
        with STATE_LOCK:
            LIVE_STATE["poll_error"] = str(exc)[:160]


def _read_state(
    local_file: Optional[Path],
    remote_url: Optional[str],
    colab_session: Optional[str] = None,
) -> Dict[str, Any]:
    if colab_session:
        with STATE_LOCK:
            if LIVE_STATE.get("meta", {}).get("step", 0) > 0 or LIVE_STATE.get("meta", {}).get("phase") != "WAIT":
                return dict(LIVE_STATE)
    if local_file is not None and local_file.exists():
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    if remote_url:
        try:
            with urllib.request.urlopen(remote_url, timeout=3.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return {"error": str(exc)}
    with STATE_LOCK:
        return dict(LIVE_STATE)


def make_handler(
    local_file: Optional[Path],
    remote_url: Optional[str],
    colab_session: Optional[str] = None,
):
    class V10Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path.startswith("/api/state"):
                payload = json.dumps(
                    _read_state(local_file, remote_url, colab_session=colab_session),
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                body = HTML_TEMPLATE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return V10Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-SHA V10 8789 实时遥测看板")
    p.add_argument("--port", type=int, default=8789)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--local_dir", type=str, default="v10_ouroboros64/runs/local_smoke")
    p.add_argument("--remote_url", type=str, default="")
    p.add_argument("--colab_session", type=str, default="")
    p.add_argument("--remote_dir", type=str, default="/content/v10_runs/a100_m2")
    p.add_argument("--poll_sec", type=float, default=15.0)
    args = p.parse_args()

    colab_session = args.colab_session.strip() or None
    local_dir_path = Path(args.local_dir) if args.local_dir else Path("v10_ouroboros64/runs/a100_m2")
    local_file = local_dir_path / "telemetry_v10.json" if args.local_dir else None
    remote_url = args.remote_url.strip() or None

    if colab_session:
        cache_dir = Path("v10_ouroboros64/runs/a100_m2")
        def _bg_poller() -> None:
            while True:
                _poll_colab_once(colab_session, args.remote_dir, cache_dir)
                time.sleep(max(5.0, float(args.poll_sec)))
        threading.Thread(target=_bg_poller, daemon=True).start()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(local_file, remote_url, colab_session=colab_session),
    )
    print(
        f"V10 看板已启动: http://{args.host}:{args.port} | local={local_file} | colab={colab_session}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
