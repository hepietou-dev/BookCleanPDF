"""OCR worker 子进程传输层：JSON-lines over stdio（M6.0 §6.3）。

职责：进程生命周期、请求/响应、页级超时、崩溃检测与一次重启、取消/关闭。
worker 只写 stdout/stderr，所有落盘由主进程完成（规避受限环境写入问题）。
单线程使用（Provider 批处理循环）；本层不做业务重试策略以外的决策。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger("bookclean")

_EOF = object()


class WorkerError(Exception):
    """传输/协议层失败（进程崩溃、超时、初始化失败、畸形响应）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _creationflags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class WorkerClient:
    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        page_timeout_s: float = 120.0,
        init_timeout_s: float = 180.0,
        restart_budget: int = 1,
    ):
        self.command = [str(c) for c in command]
        self.cwd = str(cwd) if cwd else None
        self.page_timeout_s = float(page_timeout_s)
        self.init_timeout_s = float(init_timeout_s)
        self._restart_budget = int(restart_budget)
        self._restarts_left = int(restart_budget)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._stderr_tail: deque = deque(maxlen=40)
        self._init_params: dict | None = None
        self._threads: list[threading.Thread] = []

    # ---- 生命周期 -------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        self._q = queue.Queue()
        self._stderr_tail.clear()
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self.cwd,
            creationflags=_creationflags(),
        )
        self._threads = [
            threading.Thread(target=self._read_stdout, name="ocr-worker-out", daemon=True),
            threading.Thread(target=self._read_stderr, name="ocr-worker-err", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        q = self._q
        try:
            assert proc is not None and proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            q.put(_EOF)

    def _read_stderr(self) -> None:
        proc = self._proc
        try:
            assert proc is not None and proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    self._stderr_tail.append(line)
                    logger.debug("[ocr-worker] %s", line)
        except (OSError, ValueError):
            pass

    def initialize(self, params: dict) -> dict:
        """发送 init 并等待 ready；失败抛 WorkerError。保存 params 供崩溃重启。"""
        self._init_params = dict(params)
        if not self.alive:
            self.start()
        resp = self._request({"cmd": "init", "params": params}, self.init_timeout_s)
        if resp.get("type") != "ready":
            err = resp.get("error") or {}
            raise WorkerError(
                str(resp.get("code") or err.get("code") or "INIT_FAILED"),
                str(resp.get("message") or err.get("message") or f"意外初始化响应: {resp}"),
            )
        return resp

    def recognize(self, page_no: int, image_path: str | Path) -> dict:
        """单页请求。崩溃/超时 → 重启 worker 并重试一次（预算内）；仍失败抛 WorkerError。"""
        try:
            return self._recognize_once(page_no, image_path)
        except WorkerError as exc:
            if exc.code in ("WORKER_CRASHED", "TIMEOUT") and self._restarts_left > 0 \
                    and self._init_params is not None:
                self._restarts_left -= 1
                logger.warning("OCR worker 异常（%s），重启并重试第 %s 页", exc.code, page_no)
                self._restart()
                return self._recognize_once(page_no, image_path)
            raise

    def _recognize_once(self, page_no: int, image_path: str | Path) -> dict:
        resp = self._request(
            {"cmd": "recognize", "page_no": page_no, "image": str(image_path)},
            self.page_timeout_s,
            expect_page=page_no,
        )
        if resp.get("type") != "page":
            raise WorkerError("BAD_RESPONSE", f"意外页面响应: {resp}")
        return resp

    def _restart(self) -> None:
        self.kill()
        self.start()
        assert self._init_params is not None
        self.initialize(self._init_params)

    def cancel(self) -> None:
        """协作取消：页边界生效（worker 单线程，当前页完成后读到 cancel）。"""
        self._send({"cmd": "cancel"})

    def shutdown(self, grace_s: float = 5.0) -> None:
        if not self.alive:
            self._cleanup()
            return
        try:
            self._send({"cmd": "shutdown"})
        except WorkerError:
            pass  # worker 已死/管道已断：直接清理，shutdown 绝不抛出
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and self.alive:
            time.sleep(0.05)
        self.kill()

    def kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
            except OSError:
                pass
        self._cleanup()

    def _cleanup(self) -> None:
        proc = self._proc
        if proc is not None:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
        self._proc = None

    # ---- 请求/响应 ------------------------------------------------------

    def _send(self, obj: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise WorkerError("WORKER_CRASHED", self._crash_message())
        try:
            proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise WorkerError("WORKER_CRASHED", self._crash_message(str(exc))) from exc

    def _request(self, obj: dict, timeout_s: float, expect_page: int | None = None) -> dict:
        self._send(obj)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("TIMEOUT", f"等待响应超时（{timeout_s:g}s）: cmd={obj.get('cmd')}")
            try:
                item = self._q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if not self.alive:
                    raise WorkerError("WORKER_CRASHED", self._crash_message())
                continue
            if item is _EOF:
                raise WorkerError("WORKER_CRASHED", self._crash_message())
            try:
                resp = json.loads(item)
            except (json.JSONDecodeError, TypeError):
                logger.warning("OCR worker 输出畸形 JSON（已忽略）: %.120s", str(item))
                continue
            if not isinstance(resp, dict):
                continue
            rtype = resp.get("type")
            if expect_page is not None:
                if rtype == "page" and resp.get("page_no") == expect_page:
                    return resp
                if rtype == "error":
                    return resp
                continue  # 陈旧/无关消息
            if rtype in ("ready", "error", "page", "cancelled", "bye"):
                return resp

    def _crash_message(self, extra: str = "") -> str:
        rc = self._proc.poll() if self._proc is not None else None
        tail = " | ".join(list(self._stderr_tail)[-5:])
        parts = [f"OCR worker 进程异常（exit={rc}）"]
        if extra:
            parts.append(extra)
        if tail:
            parts.append(f"stderr: {tail}")
        return "；".join(parts)