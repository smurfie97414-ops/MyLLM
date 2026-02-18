from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from model import LivingLLM, TTTLinearState
from .living_data import TiktokenTokenizer


@dataclass
class CurriculumConfig:
    amplify_every_steps: int = 16
    tasks_per_round: int = 6
    thought_loop_max_iters: int = 8
    problem_gen_tokens: int = 120
    answer_gen_tokens: int = 80
    generation_temperature: float = 0.7
    generation_top_k: int = 40
    buffer_capacity: int = 50000
    buffer_mix_ratio: float = 0.45
    immediate_train_repeats: int = 2
    prioritized_replay: bool = True
    priority_temperature: float = 0.8
    dynamic_amplification: bool = True
    plateau_patience: int = 36
    plateau_min_delta: float = 0.003
    amplification_burst_rounds: int = 2


@dataclass
class LivingTrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    mixed_precision: bool = True
    precision: str = "auto"  # auto|bf16|fp16
    output_dir: str = "runs/living"
    metrics_file: str = "metrics.jsonl"
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints_living"
    max_checkpoints: int = 5
    resume_from_latest: bool = True
    strict_fail_fast: bool = False
    max_seq_len: int = 512
    batch_size: int = 4
    seed: int = 1337
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    consistency_weight: float = 0.02
    consistency_interval: int = 4
    diff_attention_reg_weight: float = 0.01
    moe_aux_weight: float = 0.003


@dataclass
class GeneratedTask:
    task_type: str
    question: str
    canonical: str
    answer: str | None = None


class AmplificationBuffer:
    def __init__(self, capacity: int) -> None:
        self._items: deque[tuple[str, float]] = deque(maxlen=max(1, int(capacity)))

    def __len__(self) -> int:
        return len(self._items)

    def add(self, sample_text: str, priority: float = 1.0) -> None:
        self._items.append((sample_text, max(1e-4, float(priority))))

    def sample(
        self,
        n: int,
        rng: random.Random,
        prioritized: bool,
        temperature: float = 1.0,
    ) -> list[str]:
        if n <= 0 or not self._items:
            return []
        n = min(n, len(self._items))
        if not prioritized or len(self._items) <= 1:
            idxs = rng.sample(range(len(self._items)), k=n)
            return [self._items[i][0] for i in idxs]
        temp = max(1e-3, float(temperature))
        weights = [(p ** (1.0 / temp)) for _, p in self._items]
        idxs = [rng.choices(range(len(self._items)), weights=weights, k=1)[0] for _ in range(n)]
        return [self._items[i][0] for i in idxs]

    def average_priority(self) -> float:
        if not self._items:
            return 0.0
        return float(sum(p for _, p in self._items) / len(self._items))


class DeterministicTaskSolver:
    _JSON_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)
    _FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", flags=re.IGNORECASE)

    def __init__(self) -> None:
        try:
            import sympy  # noqa: F401
        except Exception as exc:
            raise RuntimeError("SymPy is required for deterministic math verification.") from exc

    def parse_task(self, text: str) -> GeneratedTask | None:
        matches = self._JSON_RE.findall(text)
        for block in matches:
            try:
                obj = json.loads(block)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            task_type = str(obj.get("type", "")).strip().lower()
            question = str(obj.get("question", "")).strip()
            canonical = str(obj.get("canonical", "")).strip()
            answer = obj.get("answer")
            if task_type in {"math", "code"} and question and canonical:
                ans = str(answer).strip() if answer is not None else None
                return GeneratedTask(task_type=task_type, question=question, canonical=canonical, answer=ans)
        return None

    def extract_answer(self, text: str) -> str:
        m = self._FINAL_RE.search(text)
        if m:
            return m.group(1).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    def _safe_eval_code_expr(self, expr: str) -> Any:
        import ast

        allowed = {
            ast.Expression,
            ast.Constant,
            ast.Num,
            ast.Str,
            ast.Name,
            ast.Load,
            ast.Tuple,
            ast.List,
            ast.Dict,
            ast.Set,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Call,
            ast.keyword,
            ast.ListComp,
            ast.comprehension,
            ast.GeneratorExp,
            ast.IfExp,
            ast.Subscript,
            ast.Slice,
            ast.Index,
        }
        safe_builtins = {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "abs": abs,
            "round": round,
            "range": range,
            "all": all,
            "any": any,
        }
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if type(n) not in allowed:
                raise ValueError(f"Unsupported code AST node: {type(n).__name__}")
            if isinstance(n, ast.Name) and n.id not in safe_builtins:
                raise ValueError(f"Unsupported identifier: {n.id}")
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name) or n.func.id not in safe_builtins:
                    raise ValueError("Only safe builtin function calls are allowed.")
        return eval(compile(node, "<safe_expr>", "eval"), {"__builtins__": {}}, safe_builtins)

    def solve(self, task: GeneratedTask) -> str:
        if task.task_type == "math":
            import sympy

            return str(sympy.simplify(task.canonical))
        if task.task_type == "code":
            return str(self._safe_eval_code_expr(task.canonical))
        raise ValueError(f"Unsupported task_type: {task.task_type}")

    def is_correct(self, task: GeneratedTask, predicted: str, expected: str) -> bool:
        if task.task_type == "math":
            import sympy

            try:
                return sympy.simplify(f"({predicted})-({expected})") == 0
            except Exception:
                return predicted.strip() == expected.strip()
        return predicted.strip() == expected.strip()


class LivingTrainer:
    def __init__(
        self,
        model: LivingLLM,
        optimizer: torch.optim.Optimizer,
        tokenizer: TiktokenTokenizer,
        seed_batch_stream,
        train_config: LivingTrainConfig,
        curriculum_config: CurriculumConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.seed_batch_stream = seed_batch_stream
        self.train_config = train_config
        self.curriculum_config = curriculum_config
        self.device = torch.device(train_config.device)
        self.rng = random.Random(train_config.seed)

        self.buffer = AmplificationBuffer(capacity=curriculum_config.buffer_capacity)
        self.solver = DeterministicTaskSolver()

        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / train_config.metrics_file
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = Path.cwd() / self.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        self.model.to(self.device)
        self.model.train()

        self._compile_disabled_reason: str | None = None
        self._compiled_active = False
        if train_config.compile_model and hasattr(torch, "compile"):
            if self._compile_sanity_check():
                try:
                    self.model = torch.compile(self.model, mode=train_config.compile_mode)  # type: ignore[assignment]
                    self._compiled_active = True
                except Exception as exc:
                    if train_config.strict_fail_fast:
                        raise RuntimeError(f"torch.compile failed in strict mode: {exc}") from exc
                    self._disable_compile_runtime(f"compile init failed: {exc}")
            else:
                self._disable_compile_runtime("compile pre-check failed")

        self.autocast_dtype = self._resolve_autocast_dtype()
        scaler_enabled = self.device.type == "cuda" and self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self._seed_iter = iter(self.seed_batch_stream)
        self._tokens_seen = 0
        self._amp_successes = 0
        self._amp_attempts = 0
        self._global_step = 0
        self._loss_ema: float | None = None
        self._best_loss_ema: float = float("inf")
        self._plateau_steps = 0
        self._last_train_aux: dict[str, float] = {}

        print(f"[living-trainer] algorithms={self.algorithm_stack()}")

    def _disable_compile_runtime(self, reason: str) -> None:
        self._compile_disabled_reason = reason
        maybe_orig = getattr(self.model, "_orig_mod", None)
        if maybe_orig is not None:
            self.model = maybe_orig
            self.model.to(self.device)
        self._compiled_active = False
        print(f"[living-trainer] compile disabled at runtime: {reason}")

    def _is_compile_error(self, exc: BaseException) -> bool:
        msg = str(exc)
        patterns = [
            "torch._inductor",
            "BackendCompilerFailed",
            "InductorError",
            "Compiler: cl is not found",
            "Cannot find a working triton installation",
            "TritonMissing",
        ]
        return any(p in msg for p in patterns)

    def _compile_sanity_check(self) -> bool:
        if self.device.type == "cpu":
            has_cl = shutil.which("cl") is not None
            has_gpp = shutil.which("g++") is not None
            if not (has_cl or has_gpp):
                return False
        try:
            tiny = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 16))
            tiny.to(self.device)
            compiled = torch.compile(tiny, mode=self.train_config.compile_mode)
            x = torch.randn(2, 16, device=self.device)
            _ = compiled(x)
            return True
        except Exception:
            return False

    def algorithm_stack(self) -> dict[str, int]:
        model_ref = getattr(self.model, "_orig_mod", self.model)
        adaptive_ttt = 0
        living_moe = 0
        try:
            if getattr(model_ref, "blocks", None):
                adaptive_ttt = int(bool(model_ref.blocks[0].ttt.adaptive_lr))
                living_moe = int(bool(model_ref.blocks[0].moe is not None))
        except Exception:
            adaptive_ttt = 0
            living_moe = 0
        return {
            "ttt_linear": 1,
            "diff_attention": 1,
            "nested_learning_memory": 1,
            "living_moe": living_moe,
            "adaptive_ttt_lr": adaptive_ttt,
            "prioritized_replay": int(self.curriculum_config.prioritized_replay),
            "dynamic_amplification": int(self.curriculum_config.dynamic_amplification),
            "consistency_regularization": int(self.train_config.consistency_weight > 0),
            "compile_model": int(self._compiled_active),
            "memory_slots_enabled": int(getattr(model_ref.memory, "slots", 0) > 0),
        }

    def _resolve_autocast_dtype(self):
        if not self.train_config.mixed_precision or self.device.type != "cuda":
            return None
        precision = self.train_config.precision.lower()
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "auto":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        raise ValueError(f"Unsupported precision mode: {self.train_config.precision}")

    def _autocast_ctx(self):
        if self.autocast_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=True)

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"living_checkpoint_step_{step:08d}.pt"

    def _save_checkpoint(self, step: int) -> None:
        payload = {
            "step": step,
            "model": getattr(self.model, "_orig_mod", self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "tokens_seen": self._tokens_seen,
            "amp_successes": self._amp_successes,
            "amp_attempts": self._amp_attempts,
            "loss_ema": self._loss_ema,
            "best_loss_ema": self._best_loss_ema,
            "plateau_steps": self._plateau_steps,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        path = self._checkpoint_path(step)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        checkpoints = sorted(self.checkpoint_dir.glob("living_checkpoint_step_*.pt"))
        while len(checkpoints) > self.train_config.max_checkpoints:
            checkpoints.pop(0).unlink(missing_ok=True)
        print(f"[living-trainer] checkpoint saved: {path}")

    def _resume_if_available(self) -> int:
        if not self.train_config.resume_from_latest:
            return 0
        checkpoints = sorted(self.checkpoint_dir.glob("living_checkpoint_step_*.pt"))
        if not checkpoints:
            return 0
        model_ref = getattr(self.model, "_orig_mod", self.model)
        for ckpt in reversed(checkpoints):
            try:
                payload = torch.load(ckpt, map_location="cpu")
                model_ref.load_state_dict(payload["model"], strict=True)
                self.optimizer.load_state_dict(payload["optimizer"])
                for state in self.optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                if self.scaler.is_enabled() and payload.get("scaler") is not None:
                    self.scaler.load_state_dict(payload["scaler"])
                self._tokens_seen = int(payload.get("tokens_seen", 0))
                self._amp_successes = int(payload.get("amp_successes", 0))
                self._amp_attempts = int(payload.get("amp_attempts", 0))
                self._loss_ema = payload.get("loss_ema")
                self._best_loss_ema = float(payload.get("best_loss_ema", float("inf")))
                self._plateau_steps = int(payload.get("plateau_steps", 0))
                step = int(payload.get("step", 0))
                print(f"[living-trainer] resumed from {ckpt} step={step}")
                return step
            except Exception as exc:
                bad = ckpt.with_name(ckpt.name + ".corrupt")
                ckpt.replace(bad)
                print(f"[living-trainer] corrupted checkpoint quarantined: {ckpt} -> {bad} error={exc}")
        return 0

    def _write_metrics(self, rec: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def _next_seed_batch(self) -> dict[str, torch.Tensor]:
        while True:
            try:
                return next(self._seed_iter)
            except StopIteration:
                self._seed_iter = iter(self.seed_batch_stream)
            except Exception as exc:
                if self.train_config.strict_fail_fast:
                    raise RuntimeError(f"Seed stream failed in strict mode: {exc}") from exc
                print(f"[living-trainer] seed stream transient error={exc}; retrying")
                time.sleep(1.0)

    def _truncate_tokens(self, ids: list[int]) -> list[int]:
        if len(ids) <= self.train_config.max_seq_len:
            return ids
        return ids[-self.train_config.max_seq_len :]

    def _batch_from_buffer(self) -> dict[str, torch.Tensor] | None:
        texts = self.buffer.sample(
            self.train_config.batch_size,
            self.rng,
            prioritized=self.curriculum_config.prioritized_replay,
            temperature=self.curriculum_config.priority_temperature,
        )
        if not texts:
            return None

        seq_len = self.train_config.max_seq_len
        inputs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) < 2:
                continue
            ids = self._truncate_tokens(ids)
            if len(ids) < seq_len + 1:
                ids = ids + [self.tokenizer.eos_token_id] * (seq_len + 1 - len(ids))
            elif len(ids) > seq_len + 1:
                start_max = max(0, len(ids) - (seq_len + 1))
                start = self.rng.randint(0, start_max)
                ids = ids[start : start + seq_len + 1]
            inputs.append(torch.tensor(ids[:-1], dtype=torch.long))
            labels.append(torch.tensor(ids[1:], dtype=torch.long))
        if not inputs:
            return None
        return {"input_ids": torch.stack(inputs, dim=0), "labels": torch.stack(labels, dim=0)}

    def _train_step(self, batch: dict[str, torch.Tensor], step: int) -> tuple[float, int, dict[str, float]]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        tokens = int(input_ids.numel())
        self.optimizer.zero_grad(set_to_none=True)

        try:
            with self._autocast_ctx():
                logits, _, stats = self.model(
                    input_ids,
                    ttt_states=None,
                    update_ttt=True,
                    update_nested_memory=True,
                    return_state=True,
                )
            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            diff_reg = stats["diff_attn_abs_mean"].to(main_loss.dtype)
            loss = main_loss + (self.train_config.diff_attention_reg_weight * diff_reg)
            moe_aux = stats.get("moe_aux_z_loss")
            if moe_aux is not None and self.train_config.moe_aux_weight > 0:
                loss = loss + (self.train_config.moe_aux_weight * moe_aux.to(main_loss.dtype))

                consistency_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
                if (
                    self.train_config.consistency_weight > 0
                    and self.train_config.consistency_interval > 0
                    and (step % self.train_config.consistency_interval) == 0
                ):
                    keep_t = min(64, logits.size(1))
                    teacher_probs = F.softmax(logits[:, -keep_t:, :].detach().float(), dim=-1)
                    static_logits = self.model(
                        input_ids,
                        ttt_states=None,
                        update_ttt=False,
                        update_nested_memory=False,
                        return_state=False,
                    )
                    static_logp = F.log_softmax(static_logits[:, -keep_t:, :].float(), dim=-1)
                    consistency_loss = F.kl_div(static_logp, teacher_probs, reduction="batchmean").to(loss.dtype)
                    loss = loss + (self.train_config.consistency_weight * consistency_loss)
        except Exception as exc:
            if self._compiled_active and self._is_compile_error(exc):
                self._disable_compile_runtime(str(exc))
                return self._train_step(batch, step=step)
            raise

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if self.train_config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            self.optimizer.step()

        aux = {
            "main_loss": float(main_loss.detach().item()),
            "diff_attn_abs_mean": float(stats["diff_attn_abs_mean"].detach().item()),
            "diff_lambda_mean": float(stats["diff_lambda_mean"].detach().item()),
            "ttt_adaptive_lr_scale": float(stats.get("ttt_adaptive_lr_scale", torch.tensor(1.0)).detach().item()),
            "ttt_input_novelty": float(stats.get("ttt_input_novelty", torch.tensor(0.0)).detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "moe_aux_z_loss": float(stats.get("moe_aux_z_loss", torch.tensor(0.0)).detach().item()),
            "moe_usage_imbalance": float(stats.get("moe_usage_imbalance", torch.tensor(0.0)).detach().item()),
            "moe_gate_scale": float(stats.get("moe_gate_scale", torch.tensor(0.0)).detach().item()),
        }
        return float(loss.detach().item()), tokens, aux

    def _encode_prompt(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        if not ids:
            ids = [self.tokenizer.eos_token_id]
        ids = self._truncate_tokens(ids)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        ttt_states: list[TTTLinearState] | None = None,
    ) -> tuple[str, list[TTTLinearState]]:
        prompt_ids = self._encode_prompt(prompt)
        try:
            out_ids, next_states, _ = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                ttt_states=ttt_states,
                update_ttt=True,
                update_nested_memory=True,
            )
        except Exception as exc:
            if self._compiled_active and self._is_compile_error(exc):
                self._disable_compile_runtime(str(exc))
                out_ids, next_states, _ = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    ttt_states=ttt_states,
                    update_ttt=True,
                    update_nested_memory=True,
                )
            else:
                raise
        full = out_ids[0].tolist()
        return self.tokenizer.decode(full[prompt_ids.size(1) :]), next_states

    def _task_prompt(self) -> str:
        return (
            "Generate one strict JSON task with keys type, question, canonical. "
            "type must be math or code. canonical must be directly verifiable."
        )

    def _solve_prompt(self, task: GeneratedTask, feedback: str = "") -> str:
        return (
            f"{feedback}\n"
            f"Solve this {task.task_type} task.\n"
            f"Question: {task.question}\n"
            f"Canonical: {task.canonical}\n"
            "Return exactly: FINAL_ANSWER: <value>"
        )

    def _build_verified_training_text(self, task: GeneratedTask, expected: str) -> str:
        return (
            "### VERIFIED_TASK\n"
            f"type: {task.task_type}\n"
            f"question: {task.question}\n"
            f"canonical: {task.canonical}\n"
            f"final_answer: {expected}\n"
            "### END\n"
        )

    def _train_immediate_on_text(self, text: str, repeats: int, step: int) -> None:
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return
        seq_len = self.train_config.max_seq_len
        if len(ids) < seq_len + 1:
            ids = ids + [self.tokenizer.eos_token_id] * (seq_len + 1 - len(ids))
        else:
            ids = ids[: seq_len + 1]
        batch = {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0),
            "labels": torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0),
        }
        for _ in range(max(1, repeats)):
            self._train_step(batch, step=step)

    def _run_amplification_round(self, step: int) -> dict[str, float]:
        successes = 0
        attempts = 0
        thought_steps_total = 0
        avg_solve_iters = 0.0
        for _ in range(self.curriculum_config.tasks_per_round):
            attempts += 1
            try:
                task_raw, task_states = self._generate_text(
                    prompt=self._task_prompt(),
                    max_new_tokens=self.curriculum_config.problem_gen_tokens,
                    temperature=self.curriculum_config.generation_temperature,
                    top_k=self.curriculum_config.generation_top_k,
                    ttt_states=None,
                )
                task = self.solver.parse_task(task_raw)
                if task is None:
                    continue
                expected = self.solver.solve(task)

                feedback = ""
                answer_states = task_states
                solved = False
                solve_iter = self.curriculum_config.thought_loop_max_iters
                for it in range(1, self.curriculum_config.thought_loop_max_iters + 1):
                    thought_steps_total += 1
                    ans_raw, answer_states = self._generate_text(
                        prompt=self._solve_prompt(task, feedback=feedback),
                        max_new_tokens=self.curriculum_config.answer_gen_tokens,
                        temperature=0.2,
                        top_k=20,
                        ttt_states=answer_states,
                    )
                    answer = self.solver.extract_answer(ans_raw)
                    if self.solver.is_correct(task, predicted=answer, expected=expected):
                        solved = True
                        solve_iter = it
                        break
                    feedback = (
                        f"Incorrect answer '{answer}'. Re-evaluate canonical exactly. "
                        f"Verifier target form: {expected}."
                    )
                if solved:
                    sample = self._build_verified_training_text(task, expected=expected)
                    difficulty = 1.0 + (float(solve_iter - 1) / max(self.curriculum_config.thought_loop_max_iters, 1))
                    self.buffer.add(sample, priority=difficulty)
                    extra_repeats = int(max(0, solve_iter - 1) // 2)
                    repeats = self.curriculum_config.immediate_train_repeats + extra_repeats
                    self._train_immediate_on_text(sample, repeats=repeats, step=step)
                    successes += 1
                    avg_solve_iters += float(solve_iter)
            except Exception as exc:
                print(f"[living-trainer] amplification task error={exc}")
        self._amp_attempts += attempts
        self._amp_successes += successes
        if successes > 0:
            avg_solve_iters /= successes
        return {
            "amp_round_attempts": float(attempts),
            "amp_round_successes": float(successes),
            "amp_round_success_rate": float(successes / max(attempts, 1)),
            "amp_round_thought_steps": float(thought_steps_total),
            "amp_round_avg_solve_iters": float(avg_solve_iters),
        }

    def _update_plateau_state(self, loss: float) -> None:
        beta = 0.98
        if self._loss_ema is None:
            self._loss_ema = loss
            self._best_loss_ema = loss
            self._plateau_steps = 0
            return
        self._loss_ema = (beta * self._loss_ema) + ((1.0 - beta) * loss)
        min_delta = self.curriculum_config.plateau_min_delta
        if self._loss_ema < (self._best_loss_ema - min_delta):
            self._best_loss_ema = self._loss_ema
            self._plateau_steps = 0
        else:
            self._plateau_steps += 1

    def _buffer_mix_probability(self) -> float:
        base = self.curriculum_config.buffer_mix_ratio
        amp_sr = float(self._amp_successes / max(self._amp_attempts, 1))
        plateau_scale = min(1.0, self._plateau_steps / max(self.curriculum_config.plateau_patience, 1))
        prob = base + (0.20 * amp_sr) + (0.15 * plateau_scale)
        return float(min(0.90, max(0.05, prob)))

    def train(self) -> None:
        start_step = self._resume_if_available()
        start_time = time.perf_counter()
        self.model.train()

        for step in range(start_step + 1, self.train_config.steps + 1):
            self._global_step = step
            step_start = time.perf_counter()
            mix_prob = self._buffer_mix_probability()
            use_buffer = len(self.buffer) > 0 and self.rng.random() < mix_prob

            if use_buffer:
                batch = self._batch_from_buffer()
                source = "amplified" if batch is not None else "seed"
            else:
                batch = None
                source = "seed"
            if batch is None:
                batch = self._next_seed_batch()

            loss, tokens, aux = self._train_step(batch, step=step)
            self._last_train_aux = aux
            self._tokens_seen += tokens
            self._update_plateau_state(loss)

            amp_metrics: dict[str, float] = {}
            run_amp = (step % max(1, self.curriculum_config.amplify_every_steps)) == 0
            if run_amp:
                amp_metrics = self._run_amplification_round(step=step)
            if (
                self.curriculum_config.dynamic_amplification
                and self._plateau_steps >= self.curriculum_config.plateau_patience
            ):
                burst: dict[str, float] = {}
                for _ in range(max(1, self.curriculum_config.amplification_burst_rounds)):
                    round_m = self._run_amplification_round(step=step)
                    for k, v in round_m.items():
                        burst[k] = burst.get(k, 0.0) + float(v)
                for k in list(burst.keys()):
                    burst[k] /= max(1, self.curriculum_config.amplification_burst_rounds)
                amp_metrics.update({f"burst_{k}": v for k, v in burst.items()})
                self._plateau_steps = 0

            step_time = time.perf_counter() - step_start
            rec: dict[str, Any] = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "step": step,
                "source": source,
                "loss": loss,
                "main_loss": aux["main_loss"],
                "consistency_loss": aux["consistency_loss"],
                "perplexity": float(math.exp(min(loss, 20.0))),
                "tokens": tokens,
                "tokens_seen_total": self._tokens_seen,
                "step_time_s": step_time,
                "tokens_per_s": tokens / max(step_time, 1e-6),
                "buffer_size": len(self.buffer),
                "buffer_avg_priority": self.buffer.average_priority(),
                "buffer_mix_probability": mix_prob,
                "amp_total_attempts": self._amp_attempts,
                "amp_total_successes": self._amp_successes,
                "amp_total_success_rate": float(self._amp_successes / max(self._amp_attempts, 1)),
                "loss_ema": float(self._loss_ema if self._loss_ema is not None else loss),
                "best_loss_ema": float(self._best_loss_ema),
                "plateau_steps": int(self._plateau_steps),
                "diff_attn_abs_mean": aux["diff_attn_abs_mean"],
                "diff_lambda_mean": aux["diff_lambda_mean"],
                "ttt_adaptive_lr_scale": aux["ttt_adaptive_lr_scale"],
                "ttt_input_novelty": aux["ttt_input_novelty"],
                "moe_aux_z_loss": aux["moe_aux_z_loss"],
                "moe_usage_imbalance": aux["moe_usage_imbalance"],
                "moe_gate_scale": aux["moe_gate_scale"],
                "cpu_util_percent": float(psutil.cpu_percent(interval=None)),
                "ram_used_percent": float(psutil.virtual_memory().percent),
            }
            rec.update({f"alg_{k}": v for k, v in self.algorithm_stack().items()})
            if hasattr(self.optimizer, "metrics"):
                try:
                    opt_metrics = self.optimizer.metrics()
                    if isinstance(opt_metrics, dict):
                        rec.update({k: float(v) for k, v in opt_metrics.items()})
                except Exception:
                    pass
            if self.device.type == "cuda":
                rec["gpu_mem_alloc_gb"] = float(torch.cuda.memory_allocated(self.device) / (1024**3))
                rec["gpu_mem_reserved_gb"] = float(torch.cuda.memory_reserved(self.device) / (1024**3))
                rec["gpu_mem_peak_gb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
                try:
                    rec["gpu_util_percent"] = float(torch.cuda.utilization(self.device))
                except Exception:
                    pass
            rec.update(amp_metrics)
            self._write_metrics(rec)

            if step % self.train_config.log_interval == 0:
                print(
                    f"step={step} loss={loss:.4f} tok/s={rec['tokens_per_s']:.2f} "
                    f"main={aux['main_loss']:.4f} cons={aux['consistency_loss']:.4f} "
                    f"buffer={len(self.buffer)} amp_sr={rec['amp_total_success_rate']:.3f} "
                    f"gpu={rec.get('gpu_util_percent', 0.0):.1f}%"
                )

            if self.train_config.save_interval > 0 and (
                step % self.train_config.save_interval == 0 or step == self.train_config.steps
            ):
                self._save_checkpoint(step)

        total = time.perf_counter() - start_time
        print(
            f"[living-trainer] training complete steps={self.train_config.steps} "
            f"elapsed_s={total:.2f} amp_success_rate={self._amp_successes / max(self._amp_attempts, 1):.4f}"
        )
