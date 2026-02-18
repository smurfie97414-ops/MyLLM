from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
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
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=max(1, int(max_workers)),
            mp_context=ctx,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "CodeSandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def run_code(self, code: str) -> CodeSandboxResult:
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

