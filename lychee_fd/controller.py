"""
lychee_fd.controller
====================

Frontend static server plus backend controller process:
- Serves Vue static files (dist/ by default) and /admin/* control endpoints.
- /admin/* starts scripts/start_backend.sh on behalf of the browser, allowing
  the frontend to select a model and restart the backend.

Start through the project script:
    bash scripts/start_frontend_dev.sh prod public

Or run directly:
    python -m lychee_fd.controller \
        --static-dir /path/to/frontend/dist \
        --host 0.0.0.0 --port 8084 \
        --backend-script /abs/path/scripts/start_backend.sh \
        --presets-file /abs/path/model_presets_dev.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


# --------------------------- logging ----------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][controller][%(levelname)s] %(message)s",
)
log = logging.getLogger("lychee_fd.controller")


# --------------------------- backend supervisor -----------------------------

class BackendSupervisor:
    """
    Manage one backend subprocess (scripts/start_backend.sh).

    - start_new_session=True creates an independent process group so killpg can
      stop vLLM workers together.
    - Shutdown sequence: SIGTERM -> grace wait -> SIGKILL.
    - After shutdown, wait briefly for GPU memory to settle before restart.
    """

    def __init__(
        self,
        backend_script: Path,
        backend_root: Path,
        backend_health_url: str,
        log_dir: Path,
        startup_timeout_s: int = 360,
        kill_grace_s: int = 8,
        gpu_settle_s: int = 3,
    ) -> None:
        self.backend_script = backend_script
        self.backend_root = backend_root
        self.backend_health_url = backend_health_url
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout_s = startup_timeout_s
        self.kill_grace_s = kill_grace_s
        self.gpu_settle_s = gpu_settle_s

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None
        self._current_model_path: Optional[str] = None
        self._current_backend_type: str = "vllm"   # vllm / hf
        self._current_mode: str = "stable"         # stable / aggressive
        self._state: str = "idle"                  # idle/starting/ready/stopping/error
        self._last_error: str = ""
        self._last_log_path: Optional[Path] = None
        self._extra_env: dict = {}

    # ------------- public API ----------------

    def status(self) -> dict:
        proc_alive = bool(self._proc and self._proc.poll() is None)
        return {
            "state": self._state,
            "alive": proc_alive,
            "pid": self._proc.pid if self._proc else None,
            "pgid": self._pgid,
            "model_path": self._current_model_path,
            "backend_type": self._current_backend_type,
            "mode": self._current_mode,
            "backend_health_url": self.backend_health_url,
            "log_path": str(self._last_log_path) if self._last_log_path else None,
            "last_error": self._last_error,
            "extra_env": dict(self._extra_env),
        }

    def restart(
        self,
        model_path: str,
        backend_type: str = "vllm",
        mode: str = "stable",
        extra_env: Optional[dict] = None,
        wait_ready: bool = True,
    ) -> dict:
        with self._lock:
            self._last_error = ""
            self._stop_locked()
            self._spawn_locked(model_path, backend_type, mode, extra_env or {})
        if wait_ready:
            ok, detail = self._wait_until_ready(self.startup_timeout_s)
            if not ok:
                self._last_error = detail
                self._state = "error"
                return {"ok": False, "error": detail, "status": self.status()}
            self._state = "ready"
        return {"ok": True, "status": self.status()}

    def stop(self) -> dict:
        with self._lock:
            self._stop_locked()
        return {"ok": True, "status": self.status()}

    def tail_log(self, n: int = 200) -> str:
        if not self._last_log_path or not self._last_log_path.exists():
            return ""
        try:
            with open(self._last_log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read_size = min(size, 256 * 1024)
                f.seek(size - read_size)
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            return "\n".join(lines[-n:])
        except Exception as exc:  # noqa: BLE001
            return f"[tail_log error] {exc}"

    # ------------- internal ----------------

    def _spawn_locked(
        self,
        model_path: str,
        backend_type: str,
        mode: str,
        extra_env: dict,
    ) -> None:
        if not self.backend_script.is_file():
            raise RuntimeError(f"backend script not found: {self.backend_script}")
        if backend_type not in ("vllm", "hf"):
            raise ValueError(f"backend_type must be vllm/hf, got {backend_type!r}")
        if mode not in ("stable", "aggressive"):
            raise ValueError(f"mode must be stable/aggressive, got {mode!r}")

        env = os.environ.copy()
        env["LYCHEEFD_MODEL_PATH"] = model_path
        # Forward selected runtime env overrides, such as hop size or lookahead.
        for k, v in (extra_env or {}).items():
            if v is None:
                continue
            env[str(k)] = str(v)

        log_path = self.log_dir / f"backend_dev_{int(time.time())}.log"
        self._last_log_path = log_path
        log.info(
            "spawning backend: script=%s mode=%s backend=%s model=%s log=%s",
            self.backend_script, mode, backend_type, model_path, log_path,
        )
        self._state = "starting"

        # start_new_session=True makes the subprocess a session/process-group
        # leader, keeping vLLM worker processes under the same pgid.
        proc = subprocess.Popen(
            ["bash", str(self.backend_script), mode, backend_type],
            cwd=str(self.backend_root),
            env=env,
            stdout=open(log_path, "ab", buffering=0),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._proc = proc
        try:
            self._pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            self._pgid = proc.pid
        self._current_model_path = model_path
        self._current_backend_type = backend_type
        self._current_mode = mode
        self._extra_env = dict(extra_env or {})

    def _stop_locked(self) -> None:
        if not self._proc:
            return
        if self._proc.poll() is not None:
            log.info("previous backend already exited rc=%s", self._proc.returncode)
            self._proc = None
            self._pgid = None
            self._state = "idle"
            return

        self._state = "stopping"
        pgid = self._pgid or self._proc.pid
        log.info("stopping backend pid=%s pgid=%s (SIGTERM)", self._proc.pid, pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        # wait grace
        deadline = time.time() + self.kill_grace_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                break
            time.sleep(0.2)

        if self._proc.poll() is None:
            log.warning("backend not exit after SIGTERM, sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.error("backend still alive after SIGKILL?")

        self._proc = None
        self._pgid = None
        self._state = "idle"

        # Give the GPU driver a short best-effort window to release memory.
        if self.gpu_settle_s > 0:
            time.sleep(self.gpu_settle_s)

    def _wait_until_ready(self, timeout_s: int) -> tuple[bool, str]:
        """
        Probe the host:port from backend_health_url until the backend listens.
        A reachable TCP port is treated as ready; application-level warmup can
        still be checked separately by the frontend.
        """
        host, port = _parse_host_port(self.backend_health_url)
        log.info("waiting backend ready on %s:%s (timeout=%ss)", host, port, timeout_s)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                rc = self._proc.returncode
                tail = self.tail_log(80)
                return False, f"backend exited early rc={rc}\n--- log tail ---\n{tail}"
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    log.info("backend port reachable: %s:%s", host, port)
                    return True, "ok"
            except OSError:
                time.sleep(1.0)
        tail = self.tail_log(80)
        return False, f"timeout waiting backend ready\n--- log tail ---\n{tail}"


def _parse_host_port(url: str) -> tuple[str, int]:
    # Minimal URL parsing is enough for controller health probes.
    s = url.replace("http://", "").replace("https://", "")
    if "/" in s:
        s = s.split("/", 1)[0]
    if ":" in s:
        host, port = s.split(":", 1)
        return host or "127.0.0.1", int(port)
    return s or "127.0.0.1", 80


# --------------------------- presets ----------------------------------------

class PresetsStore:
    def __init__(self, presets_file: Optional[Path], allowed_root: Optional[Path]) -> None:
        self.presets_file = presets_file
        self.allowed_root = allowed_root.resolve() if allowed_root else None

    def load(self) -> list[dict]:
        if not self.presets_file or not self.presets_file.is_file():
            return []
        try:
            data = json.loads(self.presets_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "presets" in data:
                data = data["presets"]
            if not isinstance(data, list):
                return []
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                mp = item.get("model_path")
                if not mp:
                    continue
                out.append({
                    "name": str(item.get("name") or mp),
                    "model_path": str(mp),
                    "backend_type": str(item.get("backend_type", "vllm")),
                    "mode": str(item.get("mode", "stable")),
                    "note": str(item.get("note", "")),
                    "extra_env": dict(item.get("extra_env") or {}),
                    "frontend_config": dict(item.get("frontend_config") or item.get("frontend") or {}),
                })
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to load presets: %s", exc)
            return []

    def is_path_allowed(self, model_path: str) -> bool:
        # 1. Preset paths are always allowed.
        for p in self.load():
            if os.path.realpath(p["model_path"]) == os.path.realpath(model_path):
                return True
        # 2. Otherwise, require the path to be under allowed_root when configured.
        if self.allowed_root is None:
            return True
        try:
            real = Path(model_path).resolve()
            return str(real).startswith(str(self.allowed_root) + os.sep) or real == self.allowed_root
        except Exception:
            return False


# --------------------------- FastAPI app ------------------------------------

def build_app(
    static_dir: Path,
    supervisor: BackendSupervisor,
    presets: PresetsStore,
    admin_token: Optional[str],
) -> FastAPI:
    app = FastAPI(title="lychee-fd controller", version="0.1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _check_token(req: Request) -> None:
        if not admin_token:
            return
        provided = req.headers.get("x-admin-token") or req.query_params.get("token")
        if provided != admin_token:
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.get("/admin/health")
    def admin_health() -> dict:
        return {"ok": True, "ts": time.time()}

    @app.get("/admin/status")
    def admin_status() -> dict:
        return supervisor.status()

    @app.get("/admin/presets")
    def admin_presets() -> dict:
        return {"presets": presets.load()}

    @app.get("/admin/logs")
    def admin_logs(tail: int = 200) -> PlainTextResponse:
        tail = max(1, min(int(tail or 200), 5000))
        return PlainTextResponse(supervisor.tail_log(tail))

    @app.post("/admin/restart")
    async def admin_restart(req: Request) -> JSONResponse:
        _check_token(req)
        try:
            payload = await req.json()
        except Exception:
            payload = {}
        model_path = (payload or {}).get("model_path") or ""
        backend_type = (payload or {}).get("backend_type") or "vllm"
        mode = (payload or {}).get("mode") or "stable"
        extra_env = (payload or {}).get("extra_env") or {}
        wait_ready = bool((payload or {}).get("wait_ready", True))

        if not model_path:
            raise HTTPException(status_code=400, detail="model_path required")
        if not presets.is_path_allowed(model_path):
            raise HTTPException(
                status_code=403,
                detail=f"model_path not allowed by whitelist: {model_path}",
            )

        try:
            result = supervisor.restart(
                model_path=model_path,
                backend_type=backend_type,
                mode=mode,
                extra_env=extra_env,
                wait_ready=wait_ready,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("restart failed")
            raise HTTPException(status_code=500, detail=str(exc))
        code = 200 if result.get("ok") else 500
        return JSONResponse(result, status_code=code)

    @app.post("/admin/stop")
    async def admin_stop(req: Request) -> dict:
        _check_token(req)
        return supervisor.stop()

    # Mount Vue dist last so /admin/* routes remain reachable.
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:
        log.warning("static dir not found, skip mount: %s", static_dir)

    return app


# --------------------------- entrypoint -------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="lychee-fd controller (DEV)")
    parser.add_argument("--host", default=os.environ.get("FRONTEND_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FRONTEND_PORT", "8084")))
    parser.add_argument("--static-dir", required=True, help="Path to Vue build dist/")
    parser.add_argument("--backend-script", required=True, help="Absolute path to scripts/start_backend.sh")
    parser.add_argument("--backend-root", default=None, help="Working dir for backend script (default: same as script dir)")
    parser.add_argument("--backend-health-url", default=os.environ.get("BACKEND_HEALTH_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--presets-file", default=None)
    parser.add_argument("--allowed-root", default=os.environ.get("ALLOWED_MODEL_ROOT", ""))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", ""))
    parser.add_argument(
        "--startup-timeout-s",
        type=int,
        default=int(os.environ.get("BACKEND_STARTUP_TIMEOUT_S", "360")),
        help="Seconds to wait for backend port readiness after spawning.",
    )
    parser.add_argument("--auto-load-default", action="store_true", help="Boot first preset automatically on startup")
    args = parser.parse_args()

    backend_script = Path(args.backend_script).resolve()
    backend_root = Path(args.backend_root).resolve() if args.backend_root else backend_script.parent
    static_dir = Path(args.static_dir).resolve()
    presets_file = Path(args.presets_file).resolve() if args.presets_file else None
    allowed_root = Path(args.allowed_root).resolve() if args.allowed_root else None
    log_dir = Path(args.log_dir).resolve() if args.log_dir else (backend_root / "runtime_logs" / "controller")

    supervisor = BackendSupervisor(
        backend_script=backend_script,
        backend_root=backend_root,
        backend_health_url=args.backend_health_url,
        log_dir=log_dir,
        startup_timeout_s=args.startup_timeout_s,
    )
    presets = PresetsStore(presets_file=presets_file, allowed_root=allowed_root)

    log.info("controller config:")
    log.info("  static_dir       = %s", static_dir)
    log.info("  backend_script   = %s", backend_script)
    log.info("  backend_root     = %s", backend_root)
    log.info("  backend_health   = %s", args.backend_health_url)
    log.info("  presets_file     = %s", presets_file)
    log.info("  allowed_root     = %s", allowed_root)
    log.info("  admin_token      = %s", "<set>" if args.admin_token else "<empty>")
    log.info("  log_dir          = %s", log_dir)
    log.info("  startup_timeout = %ss", args.startup_timeout_s)

    app = build_app(
        static_dir=static_dir,
        supervisor=supervisor,
        presets=presets,
        admin_token=(args.admin_token or None),
    )

    # Graceful shutdown.
    def _graceful(signum, _frame):
        log.info("controller received signal=%s, stopping backend...", signum)
        try:
            supervisor.stop()
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    if args.auto_load_default:
        plist = presets.load()
        if plist:
            p = plist[0]
            try:
                supervisor.restart(
                    model_path=p["model_path"],
                    backend_type=p.get("backend_type", "vllm"),
                    mode=p.get("mode", "stable"),
                    extra_env=p.get("extra_env") or {},
                    wait_ready=False,
                )
            except Exception:  # noqa: BLE001
                log.exception("auto-load default preset failed")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
