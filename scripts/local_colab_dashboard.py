"""Local REST-relay Web Dashboard & live file sync for Colab Phase-SHA V4 runs."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from colab_cli.auth import AuthProvider  # type: ignore  # noqa: E402
from colab_cli.common import state  # type: ignore  # noqa: E402
from colab_cli.contents import ContentsClient  # type: ignore  # noqa: E402

from colab_monitor_dashboard import HTML_PAGE  # noqa: E402


class RelayCache:
    """Thread-safe cache for remote Colab telemetry snapshots and sync tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: Dict[str, Any] = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu": {},
            "processes": [],
            "runs": {},
            "relay_status": "initializing",
        }
        self.last_row_counts: Dict[str, int] = {}
        self.last_file_sync_ts: float = 0.0
        self.ckpt_mtimes: Dict[str, float] = {}
        self.last_keepalive_ts: float = 0.0
        self.keepalive_lock = threading.Lock()

    def update_ok(self, snapshot: Dict[str, Any]) -> None:
        enriched = dict(snapshot)
        enriched["relay_status"] = "connected"
        enriched["relay_fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        enriched.pop("relay_error", None)
        with self._lock:
            self._payload = enriched

    def set_error(self, err_msg: str) -> None:
        with self._lock:
            copy_payload = dict(self._payload)
            copy_payload["relay_status"] = "retrying"
            copy_payload["relay_error"] = err_msg
            self._payload = copy_payload

    def get_bytes(self) -> bytes:
        with self._lock:
            return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")


def _decode_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    content = data.get("content", "")
    fmt = data.get("format")
    if fmt == "json" and isinstance(content, dict):
        return content
    if fmt == "base64" and isinstance(content, str):
        raw_bytes = base64.b64decode(content)
        return json.loads(raw_bytes.decode("utf-8"))
    if isinstance(content, dict):
        return content
    return json.loads(str(content))


def _atomic_download(contents: ContentsClient, remote_path: str, local_dst: Path) -> None:
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_dst.with_suffix(local_dst.suffix + ".tmp")
    contents.download(remote_path, str(tmp_path))
    os.replace(tmp_path, local_dst)


def _maybe_spawn_keepalive(session_name: str, runs: Dict[str, Any], cache: RelayCache, now_mono: float) -> None:
    if (now_mono - cache.last_keepalive_ts) < 180.0:
        return
    if not any(bool(info.get("is_running")) for info in runs.values()):
        return
    colab_exe = Path(sys.executable).parent / "colab.exe"
    if not colab_exe.exists():
        return
    if not cache.keepalive_lock.acquire(blocking=False):
        return
    cache.last_keepalive_ts = now_mono

    def _worker() -> None:
        try:
            subprocess.run(
                [str(colab_exe), "--auth=oauth2", "exec", "-s", session_name, "--timeout", "15"],
                input="_ka=1\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=25,
            )
        except Exception:
            pass
        finally:
            cache.keepalive_lock.release()

    threading.Thread(target=_worker, daemon=True).start()


def _sync_run_files(
    contents: ContentsClient,
    runs: Dict[str, Any],
    sync_dir: Path,
    cache: RelayCache,
    now_mono: float,
) -> None:
    need_log_sync = (now_mono - cache.last_file_sync_ts) >= 30.0
    if not need_log_sync:
        for run_name, info in runs.items():
            cur_rows = int(info.get("row_count") or 0)
            if cur_rows > cache.last_row_counts.get(run_name, -1):
                need_log_sync = True
                break

    for run_name, info in runs.items():
        dst_dir = sync_dir / run_name
        if need_log_sync:
            for fname in ("metrics.csv", "train.log", "v7_summary.json"):
                remote_file = f"/content/runs/{run_name}/{fname}"
                try:
                    _atomic_download(contents, remote_file, dst_dir / fname)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    print(f"[relay] file sync warning ({run_name}/{fname}): {type(exc).__name__}", flush=True)
            cache.last_row_counts[run_name] = int(info.get("row_count") or 0)

        for ckpt in info.get("checkpoints") or []:
            cname = str(ckpt.get("name") or "")
            if not cname.endswith(".pt"):
                continue
            cmtime = float(ckpt.get("mtime") or 0.0)
            ckey = f"{run_name}/{cname}"
            local_ckpt = dst_dir / cname
            prev_mtime = cache.ckpt_mtimes.get(ckey, -1.0)
            if (cmtime > prev_mtime) or (not local_ckpt.exists()):
                remote_ckpt = f"/content/runs/{run_name}/{cname}"
                try:
                    _atomic_download(contents, remote_ckpt, local_ckpt)
                    cache.ckpt_mtimes[ckey] = cmtime
                    print(f"[relay] synced checkpoint {ckey} ({ckpt.get('size_mb', '?')} MB)", flush=True)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    print(f"[relay] ckpt sync warning ({ckey}): {type(exc).__name__}", flush=True)

    if need_log_sync:
        cache.last_file_sync_ts = now_mono


def _poller_loop(
    session_name: str,
    poll_sec: float,
    sync_dir: Path,
    cache: RelayCache,
    stop_event: threading.Event,
) -> None:
    state.auth_provider = AuthProvider.OAUTH2
    session_list = [s.strip() for s in session_name.split(",") if s.strip()]
    if not session_list:
        session_list = [session_name]
    last_logged_err = ""
    while not stop_event.is_set():
        try:
            merged_runs: Dict[str, Any] = {}
            merged_procs: list[Any] = []
            gpus_by_session: Dict[str, Any] = {}
            primary_gpu: Dict[str, Any] = {}
            last_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            now_mono = time.monotonic()

            for s_idx, s_name in enumerate(session_list):
                sess = state.get_session(s_name, ignore_missing_session=True)
                if sess is None:
                    if len(session_list) == 1:
                        raise RuntimeError(f"Colab session '{s_name}' not active")
                    continue
                contents = ContentsClient(sess)
                data = contents._request(
                    "GET",
                    "/content/runs/dashboard_status.json",
                    params={"content": "1"},
                )
                snap = _decode_snapshot(data)
                if s_idx == 0:
                    primary_gpu = snap.get("gpu") or {}
                    last_timestamp = snap.get("timestamp") or last_timestamp
                gpus_by_session[s_name] = snap.get("gpu") or {}
                for p in snap.get("processes") or []:
                    proc_item = dict(p) if isinstance(p, dict) else {"cmd": str(p)}
                    proc_item["session"] = s_name
                    merged_procs.append(proc_item)

                s_runs = snap.get("runs") or {}
                if isinstance(s_runs, dict) and s_runs:
                    for rname, rinfo in s_runs.items():
                        # In multi-VM mode, keep v7a/b/c runs from their respective host VMs
                        if len(session_list) > 1 and rname in merged_runs and not bool(rinfo.get("is_running")):
                            continue
                        enriched_r = dict(rinfo)
                        enriched_r["session"] = s_name
                        merged_runs[rname] = enriched_r
                    _sync_run_files(contents, s_runs, sync_dir, cache, now_mono)
                    _maybe_spawn_keepalive(s_name, s_runs, cache, now_mono)

            if not gpus_by_session:
                raise RuntimeError(f"No active Colab sessions among {session_list}")

            combined_snapshot = {
                "timestamp": last_timestamp,
                "gpu": primary_gpu,
                "gpus_by_session": gpus_by_session,
                "processes": merged_procs,
                "runs": merged_runs,
                "sessions": session_list,
            }
            cache.update_ok(combined_snapshot)
            last_logged_err = ""
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            cache.set_error(err_msg)
            if err_msg != last_logged_err:
                print(f"[relay] poll retry ({err_msg})", flush=True)
                last_logged_err = err_msg
        stop_event.wait(poll_sec)


def make_relay_handler(cache: RelayCache):
    class RelayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                payload = cache.get_bytes()
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

    return RelayHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Colab Dashboard Relay")
    parser.add_argument("--session", type=str, default="phase-sha-v4")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll_sec", type=float, default=5.0)
    parser.add_argument("--sync_dir", type=str, default=r"g:\SHA-LM\runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state.auth_provider = AuthProvider.OAUTH2
    sync_dir = Path(args.sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)

    cache = RelayCache()
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poller_loop,
        args=(args.session, float(args.poll_sec), sync_dir, cache, stop_event),
        daemon=True,
    )
    poller.start()

    server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), make_relay_handler(cache))
    print(
        f"[relay] listening on http://127.0.0.1:{args.port} | session={args.session} | sync_dir={sync_dir}",
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
