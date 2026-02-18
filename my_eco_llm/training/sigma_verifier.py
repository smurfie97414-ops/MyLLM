from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import queue
import random
import re
import time
from typing import Any


_INT_RE = re.compile(r"-?\d+")


@dataclass
class SigmaTask:
    task_id: str
    domain: str  # math|code
    prompt: str
    expected: str


@dataclass
class VerificationResult:
    passed: bool
    score: float
    parsed_answer: str
    latency_ms: float
    error_type: str = ""


def parse_int_answer(text: str) -> str:
    matches = _INT_RE.findall(text)
    return matches[-1] if matches else ""


def build_math_task(rng: random.Random, *, min_v: int = 100, max_v: int = 999) -> SigmaTask:
    a = rng.randint(min_v, max_v)
    b = rng.randint(min_v, max_v)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        expected = str(a + b)
    elif op == "-":
        expected = str(a - b)
    else:
        expected = str(a * b)
    prompt = (
        "Solve and return only one integer at the end.\n"
        f"Problem: {a} {op} {b}\n"
        "Answer:"
    )
    return SigmaTask(task_id=f"math_{a}_{op}_{b}", domain="math", prompt=prompt, expected=expected)


def build_code_task(rng: random.Random, *, min_v: int = 2, max_v: int = 30) -> SigmaTask:
    x = rng.randint(min_v, max_v)
    y = rng.randint(min_v, max_v)
    z = rng.randint(min_v, max_v)
    expected_val = ((x * y) + z) % (x + 3)
    prompt = (
        "Read this Python snippet and return only the final integer value of out.\n"
        "```python\n"
        f"x = {x}\n"
        f"y = {y}\n"
        f"z = {z}\n"
        "out = ((x * y) + z) % (x + 3)\n"
        "print(out)\n"
        "```\n"
        "Answer:"
    )
    return SigmaTask(task_id=f"code_eval_{x}_{y}_{z}", domain="code", prompt=prompt, expected=str(expected_val))


def verify_answer(task: SigmaTask, completion_text: str) -> VerificationResult:
    start = time.perf_counter()
    parsed = parse_int_answer(completion_text)
    passed = parsed == task.expected
    score = 1.0 if passed else 0.0
    return VerificationResult(
        passed=passed,
        score=score,
        parsed_answer=parsed,
        latency_ms=float((time.perf_counter() - start) * 1000.0),
        error_type="" if passed else "wrong_answer",
    )


def _exec_code_with_tests_worker(code: str, tests: list[tuple[Any, Any]], out_q: mp.Queue) -> None:
    try:
        scope: dict[str, Any] = {}
        exec(code, {"__builtins__": {}}, scope)
        fn = scope.get("solve", None)
        if fn is None or not callable(fn):
            out_q.put(("missing_solve", 0))
            return
        ok = 0
        for inp, expected in tests:
            if fn(inp) == expected:
                ok += 1
        out_q.put(("ok", ok))
    except Exception:
        out_q.put(("exec_error", 0))


def verify_code_function(
    code: str,
    tests: list[tuple[Any, Any]],
    *,
    timeout_ms: int = 2500,
) -> VerificationResult:
    start = time.perf_counter()
    q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=_exec_code_with_tests_worker, args=(code, tests, q))
    p.start()
    p.join(timeout=max(0.1, timeout_ms / 1000.0))
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return VerificationResult(
            passed=False,
            score=0.0,
            parsed_answer="",
            latency_ms=float((time.perf_counter() - start) * 1000.0),
            error_type="timeout",
        )
    try:
        status, ok = q.get_nowait()
    except queue.Empty:
        status, ok = ("exec_error", 0)
    passed = status == "ok" and ok == len(tests)
    score = float(ok / max(len(tests), 1))
    return VerificationResult(
        passed=passed,
        score=score,
        parsed_answer="",
        latency_ms=float((time.perf_counter() - start) * 1000.0),
        error_type="" if passed else str(status),
    )


def load_task_manifest(path: str) -> list[SigmaTask]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise RuntimeError(f"Task manifest must be a list, got {type(payload)!r}")
    tasks: list[SigmaTask] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid manifest item at {idx}: {type(item)!r}")
        task_id = str(item.get("task_id", f"manifest_{idx}"))
        domain = str(item.get("domain", "math"))
        prompt = str(item.get("prompt", ""))
        expected = str(item.get("expected", ""))
        if not prompt or not expected:
            raise RuntimeError(f"Manifest item {task_id} missing prompt/expected.")
        tasks.append(SigmaTask(task_id=task_id, domain=domain, prompt=prompt, expected=expected))
    if not tasks:
        raise RuntimeError("Task manifest is empty.")
    return tasks
