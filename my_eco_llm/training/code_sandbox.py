from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FutureTimeout
import contextlib
import io
import multiprocessing as mp
import time
import traceback


@dataclass
class CodeSandboxResult:
    exit_code: int
    timed_out: bool
    stdout: str
    stderr: str
    error: str
    wall_time_s: float


def _run_code_worker(code: str) -> tuple[int, str, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            scope = {"__name__": "__main__"}
            compiled = compile(code, "<generated_code>", "exec")
            exec(compiled, scope, scope)
        return 0, out.getvalue(), err.getvalue(), ""
    except Exception:
        return 1, out.getvalue(), err.getvalue(), traceback.format_exc(limit=8)


class CodeSandbox:
    def __init__(self, timeout_sec: float = 4.0, max_workers: int = 1) -> None:
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
        self.timeout_sec = float(timeout_sec)
        self._startup_timeout_sec = max(15.0, self.timeout_sec * 5.0)
        self._executor = None
        self._process_mode = False
        self._workers = max(1, int(max_workers))
        try:
            ctx = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=ctx,
            )
            self._process_mode = True
        except Exception as exc:
            # Restricted environments can deny process-spawn primitives (WinError 5).
            # Fallback keeps runtime functional for verifier/reward path.
            print(f"[sigma][warn] code_sandbox_process_pool_unavailable={type(exc).__name__}: {exc}")
            self._executor = ThreadPoolExecutor(max_workers=self._workers)
            self._process_mode = False
        try:
            self._warm_executor()
        except Exception as exc:
            if self._process_mode:
                print(f"[sigma][warn] code_sandbox_process_pool_warmup_failed={type(exc).__name__}: {exc}")
                if self._executor is not None:
                    self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = ThreadPoolExecutor(max_workers=self._workers)
                self._process_mode = False
                self._warm_executor()
            else:
                raise

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "CodeSandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def _warm_executor(self) -> None:
        if self._executor is None:
            return
        future = self._executor.submit(_run_code_worker, "pass")
        exit_code, _, _, error = future.result(timeout=self._startup_timeout_sec)
        if int(exit_code) != 0:
            raise RuntimeError(f"sandbox_warmup_failed: {error}")

    def run_code(self, code: str) -> CodeSandboxResult:
        if self._executor is None:
            return CodeSandboxResult(
                exit_code=2,
                timed_out=False,
                stdout="",
                stderr="",
                error="sandbox_unavailable",
                wall_time_s=0.0,
            )
        started = time.perf_counter()
        future = self._executor.submit(_run_code_worker, str(code))
        try:
            exit_code, stdout, stderr, error = future.result(timeout=self.timeout_sec)
            return CodeSandboxResult(
                exit_code=int(exit_code),
                timed_out=False,
                stdout=stdout,
                stderr=stderr,
                error=error,
                wall_time_s=max(0.0, time.perf_counter() - started),
            )
        except FutureTimeout:
            future.cancel()
            return CodeSandboxResult(
                exit_code=124,
                timed_out=True,
                stdout="",
                stderr="",
                error=f"timeout>{self.timeout_sec:.3f}s",
                wall_time_s=max(0.0, time.perf_counter() - started),
            )

    def reward(self, generated_code: str, success_reward: float = 2.0, fail_reward: float = -1.0) -> float:
        res = self.run_code(generated_code)
        return float(success_reward if res.exit_code == 0 else fail_reward)
