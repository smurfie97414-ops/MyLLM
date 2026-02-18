from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from .code_sandbox import CodeSandbox


def _safe_kwargs(callable_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(callable_obj)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in payload.items() if k in valid}


def _extract_completion_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "content" in item:
            return str(item.get("content", ""))
        if "text" in item:
            return str(item.get("text", ""))
        if "completion" in item:
            return str(item.get("completion", ""))
    return str(item)


@dataclass
class TRLGRPOBridgeConfig:
    num_generations: int = 8
    max_new_tokens: int = 64
    temperature: float = 0.7
    code_timeout_sec: float = 4.0
    success_reward: float = 2.0
    fail_reward: float = -1.0


class TRLGRPOBridge:
    """
    Bridge layer wiring TRL GRPOTrainer/GRPOConfig to executable ground-truth code rewards.
    """

    def __init__(self, model: Any, tokenizer: Any, config: TRLGRPOBridgeConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TRLGRPOBridgeConfig()
        self.sandbox = CodeSandbox(timeout_sec=self.config.code_timeout_sec)

        try:
            from trl import GRPOConfig, GRPOTrainer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TRL GRPO is required but unavailable. Install a TRL version exposing "
                "GRPOConfig and GRPOTrainer."
            ) from exc

        self._GRPOConfig = GRPOConfig
        self._GRPOTrainer = GRPOTrainer

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> "TRLGRPOBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def reward_func(self, prompts: Any, completions: Any, **kwargs: Any) -> list[float]:
        del prompts, kwargs
        rewards: list[float] = []
        for comp in completions:
            code = _extract_completion_text(comp)
            rewards.append(
                self.sandbox.reward(
                    generated_code=code,
                    success_reward=self.config.success_reward,
                    fail_reward=self.config.fail_reward,
                )
            )
        return rewards

    def build_trainer(
        self,
        *,
        train_dataset: Any,
        output_dir: str,
        learning_rate: float = 1e-6,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_steps: int = 100,
        **extra_args: Any,
    ) -> Any:
        cfg_payload = {
            "output_dir": output_dir,
            "learning_rate": float(learning_rate),
            "per_device_train_batch_size": int(per_device_train_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "max_steps": int(max_steps),
            "num_generations": int(self.config.num_generations),
            "temperature": float(self.config.temperature),
            "max_new_tokens": int(self.config.max_new_tokens),
            "max_completion_length": int(self.config.max_new_tokens),
        }
        cfg_payload.update(extra_args)
        cfg = self._GRPOConfig(**_safe_kwargs(self._GRPOConfig, cfg_payload))

        trainer_payload = {
            "model": self.model,
            "args": cfg,
            "train_dataset": train_dataset,
            "processing_class": self.tokenizer,
            "reward_funcs": [self.reward_func],
            "reward_func": self.reward_func,
        }
        return self._GRPOTrainer(**_safe_kwargs(self._GRPOTrainer, trainer_payload))

