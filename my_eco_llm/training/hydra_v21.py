from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bitlinear import BitLinear
from model.merger import MergeConfig, merge_adapter_states
from .sigma_verifier import SigmaTask, build_code_task, build_math_task, verify_answer


@dataclass
class HydraV21Config:
    enabled: bool = False
    domains: tuple[str, ...] = ("math", "code", "text")
    steps_per_phase: int = 5000
    n_candidates: int = 4
    unverified_cap_text: float = 0.20
    rollback_interval: int = 200
    rollback_threshold: float = 0.01
    update_interval: int = 16
    dpo_beta: float = 0.1
    sampo_verbosity_weight: float = 0.02
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_lr: float = 1e-4
    merge_method: str = "ties"
    merge_density: float = 1.0
    merge_fold_into_backbone: bool = False
    replay_capacity: int = 256
    anchor_capacity: int = 64
    seed: int = 20260216


class _LoRABranch(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(max(1, rank))
        self.alpha = float(alpha)
        self.scaling = float(self.alpha / float(self.rank))
        self.a = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.b = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5.0))
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.a)
        return F.linear(z, self.b) * self.scaling


class _LoRAInjectedBitLinear(nn.Module):
    def __init__(self, base: BitLinear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.adapters = nn.ModuleDict()
        self.active_adapter: str | None = None

    def add_adapter(self, name: str) -> None:
        key = str(name)
        if key in self.adapters:
            return
        branch = _LoRABranch(
            in_features=int(self.base.in_features),
            out_features=int(self.base.out_features),
            rank=self.rank,
            alpha=self.alpha,
        )
        branch = branch.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.adapters[key] = branch

    def set_active_adapter(self, name: str | None) -> None:
        if name is None:
            self.active_adapter = None
            return
        key = str(name)
        if key not in self.adapters:
            raise KeyError(f"Unknown adapter: {key}")
        self.active_adapter = key

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        key = str(name)
        if key not in self.adapters:
            return []
        branch = self.adapters[key]
        return [branch.a, branch.b]

    def adapter_state(self, name: str) -> dict[str, torch.Tensor]:
        key = str(name)
        if key not in self.adapters:
            return {}
        branch = self.adapters[key]
        return {"a": branch.a.detach().clone(), "b": branch.b.detach().clone()}

    def load_adapter_state(self, name: str, state: dict[str, torch.Tensor]) -> None:
        key = str(name)
        self.add_adapter(key)
        branch = self.adapters[key]
        a = state.get("a", None)
        b = state.get("b", None)
        if a is None or b is None:
            raise RuntimeError(f"Adapter state for {key} must contain 'a' and 'b'.")
        branch.a.data.copy_(a.to(device=branch.a.device, dtype=branch.a.dtype))
        branch.b.data.copy_(b.to(device=branch.b.device, dtype=branch.b.dtype))

    def fold_adapter_into_base_(self, name: str) -> None:
        key = str(name)
        if key not in self.adapters:
            return
        branch = self.adapters[key]
        delta = torch.matmul(branch.b.float(), branch.a.float()) * float(branch.scaling)
        self.base.weight.data.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        if hasattr(self.base, "_invalidate_cache"):
            self.base._invalidate_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active_adapter is None:
            return out
        if self.active_adapter not in self.adapters:
            return out
        branch = self.adapters[self.active_adapter]
        return out + branch(x).to(dtype=out.dtype)


@dataclass
class _PreferencePair:
    domain: str
    prompt_ids: list[int]
    pos_ids: list[int]
    neg_ids: list[int]
    pos_reward: float
    neg_reward: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": str(self.domain),
            "prompt_ids": [int(x) for x in self.prompt_ids],
            "pos_ids": [int(x) for x in self.pos_ids],
            "neg_ids": [int(x) for x in self.neg_ids],
            "pos_reward": float(self.pos_reward),
            "neg_reward": float(self.neg_reward),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "_PreferencePair":
        return _PreferencePair(
            domain=str(payload.get("domain", "text")),
            prompt_ids=[int(x) for x in payload.get("prompt_ids", [])],
            pos_ids=[int(x) for x in payload.get("pos_ids", [])],
            neg_ids=[int(x) for x in payload.get("neg_ids", [])],
            pos_reward=float(payload.get("pos_reward", 0.0)),
            neg_reward=float(payload.get("neg_reward", 0.0)),
        )


class HydraV21Engine:
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        output_dir: Path,
        config: HydraV21Config,
        verifier_math_enabled: bool,
        verifier_code_enabled: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.rng = random.Random(int(config.seed))
        self.verifier_math_enabled = bool(verifier_math_enabled)
        self.verifier_code_enabled = bool(verifier_code_enabled)

        configured_domains = [str(d).strip().lower() for d in config.domains if str(d).strip()]
        valid_domains = [d for d in configured_domains if d in {"math", "code", "text"}]
        if not valid_domains:
            valid_domains = ["math", "code", "text"]
        if not self.verifier_math_enabled:
            valid_domains = [d for d in valid_domains if d != "math"]
        if not self.verifier_code_enabled:
            valid_domains = [d for d in valid_domains if d != "code"]
        if not valid_domains:
            valid_domains = ["text"]
        self.domains = tuple(valid_domains)

        self.hydra_dir = self.output_dir / "hydra"
        self.hydra_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.hydra_dir / "hydra_state.json"

        self._target_keywords = (
            "attn_exp.q_content",
            "attn_exp.q_rope",
            "attn_exp.kv_down",
            "attn_exp.k_content_up",
            "attn_exp.k_rope_up",
            "attn_exp.v_up",
            "attn_exp.out_proj",
            "attn_ref.q_content",
            "attn_ref.q_rope",
            "attn_ref.kv_down",
            "attn_ref.k_content_up",
            "attn_ref.k_rope_up",
            "attn_ref.v_up",
            "attn_ref.out_proj",
        )
        self._wrappers = self._inject_lora()
        if not self._wrappers:
            raise RuntimeError("Hydra-V2.1 requires at least one LoRA-injected attention projection.")

        params: list[nn.Parameter] = []
        for domain in self.domains:
            params.extend(self._domain_lora_params(domain))
        unique_params: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            unique_params.append(p)
        self.optimizer = torch.optim.AdamW(unique_params, lr=float(self.config.lora_lr), weight_decay=0.0)

        self.replay = {d: deque(maxlen=max(32, int(config.replay_capacity))) for d in self.domains}
        self.anchor = {d: deque(maxlen=max(16, int(config.anchor_capacity))) for d in self.domains}
        self.phase_start_scores = {d: 0.0 for d in self.domains}
        self.last_gate_scores = {d: 0.0 for d in self.domains}
        self.best_state = {d: self._capture_domain_state(d) for d in self.domains}
        self.phase_idx = 0

        self.anchor_prompts = {
            "math": [
                "Solve and return one integer: 173 + 289.\nAnswer:",
                "Solve and return one integer: 944 - 377.\nAnswer:",
            ],
            "code": [
                "Read the code and return only one integer.\n```python\nx=7\ny=11\nprint((x*y+5)%(x+3))\n```\nAnswer:",
                "Read the code and return only one integer.\n```python\na=9\nb=4\nprint((a*b+2)%(b+5))\n```\nAnswer:",
            ],
            "text": [
                "Write a concise 3-sentence explanation of why overfitting hurts generalization.",
                "Give a short and precise summary of gradient clipping benefits.",
            ],
        }
        self._refresh_anchors()

    def _safe_decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(token_ids)
        except Exception:
            enc = getattr(self.tokenizer, "_enc", None)
            if enc is not None and hasattr(enc, "decode_single_token_bytes"):
                out: list[str] = []
                for tid in token_ids:
                    try:
                        out.append(enc.decode_single_token_bytes(int(tid)).decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                return "".join(out)
            filtered = [int(t) for t in token_ids if 0 <= int(t) < int(getattr(self.tokenizer, "vocab_size", 10**9))]
            try:
                return self.tokenizer.decode(filtered)
            except Exception:
                return ""

    def _resolve_parent(self, module_name: str) -> tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent: nn.Module = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def _inject_lora(self) -> dict[str, _LoRAInjectedBitLinear]:
        wrappers: dict[str, _LoRAInjectedBitLinear] = {}
        named_modules = list(self.model.named_modules())
        for name, mod in named_modules:
            if not isinstance(mod, BitLinear):
                continue
            if not any(k in name for k in self._target_keywords):
                continue
            parent, attr = self._resolve_parent(name)
            wrapped = _LoRAInjectedBitLinear(mod, rank=int(self.config.lora_rank), alpha=float(self.config.lora_alpha))
            for domain in self.domains:
                wrapped.add_adapter(domain)
            wrapped = wrapped.to(device=self.device)
            setattr(parent, attr, wrapped)
            wrappers[name] = wrapped
        return wrappers

    def _domain_lora_params(self, domain: str) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for w in self._wrappers.values():
            out.extend(w.adapter_parameters(domain))
        return out

    def _capture_domain_state(self, domain: str) -> dict[str, dict[str, torch.Tensor]]:
        state: dict[str, dict[str, torch.Tensor]] = {}
        for name, w in self._wrappers.items():
            state[name] = w.adapter_state(domain)
        return state

    def _restore_domain_state(self, domain: str, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, payload in state.items():
            w = self._wrappers.get(name, None)
            if w is None:
                continue
            w.load_adapter_state(domain, payload)

    @contextmanager
    def _active_adapter(self, domain: str | None):
        old = {k: w.active_adapter for k, w in self._wrappers.items()}
        try:
            for w in self._wrappers.values():
                w.set_active_adapter(domain)
            yield
        finally:
            for k, w in self._wrappers.items():
                w.set_active_adapter(old[k])

    def _build_prompt(self, domain: str) -> tuple[str, SigmaTask | None]:
        if domain == "math":
            t = build_math_task(self.rng)
            return t.prompt, t
        if domain == "code":
            t = build_code_task(self.rng)
            return t.prompt, t
        p = self.rng.choice(self.anchor_prompts["text"])
        return p, None

    def _text_score(self, text: str, completion_tokens: int) -> float:
        words = [w for w in text.strip().split() if w]
        n_words = len(words)
        uniq = len(set(words)) / max(n_words, 1)
        rep = 1.0 - uniq
        verbosity = max(0.0, (float(completion_tokens) - 96.0) / 96.0)
        score = (0.55 * uniq) + (0.45 * (1.0 - verbosity)) - (0.25 * rep)
        return float(min(max(score, 0.0), 1.0))

    def _reward(self, domain: str, task: SigmaTask | None, seq: torch.Tensor, prompt_len: int) -> tuple[float, bool]:
        text = self._safe_decode([int(x) for x in seq.tolist()])
        if domain in {"math", "code"} and task is not None:
            vr = verify_answer(task, text)
            return float(vr.score), bool(vr.passed)
        completion_tokens = max(0, int(seq.numel()) - int(prompt_len))
        score = self._text_score(text, completion_tokens=completion_tokens)
        cap = float(min(max(self.config.unverified_cap_text, 0.0), 1.0))
        score = min(score, cap + ((1.0 - cap) * score))
        return float(score), bool(score >= 0.5)

    def _completion_logprob(self, seq_ids: list[int], prompt_len: int, domain: str | None) -> torch.Tensor:
        seq = torch.tensor(seq_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        if seq.size(1) < 2:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        x = seq[:, :-1]
        y = seq[:, 1:]
        with self._active_adapter(domain):
            logits = self.model(x)
            logp = F.log_softmax(logits, dim=-1)
            picked = logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        start = max(0, int(prompt_len) - 1)
        picked = picked[:, start:]
        return picked.sum(dim=1).mean()

    def _pair_loss(self, pair: _PreferencePair) -> torch.Tensor:
        prompt_len = len(pair.prompt_ids)
        pi_pos = self._completion_logprob(pair.pos_ids, prompt_len, pair.domain)
        pi_neg = self._completion_logprob(pair.neg_ids, prompt_len, pair.domain)
        with torch.no_grad():
            ref_pos = self._completion_logprob(pair.pos_ids, prompt_len, None)
            ref_neg = self._completion_logprob(pair.neg_ids, prompt_len, None)
        delta = (pi_pos - ref_pos) - (pi_neg - ref_neg)
        dpo = -F.logsigmoid(float(self.config.dpo_beta) * delta)

        pos_len = max(1, len(pair.pos_ids) - prompt_len)
        neg_len = max(1, len(pair.neg_ids) - prompt_len)
        avg_len = 0.5 * float(pos_len + neg_len)
        verbosity = avg_len / 128.0
        reg = float(self.config.sampo_verbosity_weight) * verbosity
        return dpo + reg

    def _pair_loss_cached(
        self,
        pair: _PreferencePair,
        *,
        pi_cache: dict[tuple[str, int, tuple[int, ...]], torch.Tensor],
        ref_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor],
    ) -> torch.Tensor:
        prompt_len = len(pair.prompt_ids)
        pos_key = (str(pair.domain), int(prompt_len), tuple(int(x) for x in pair.pos_ids))
        neg_key = (str(pair.domain), int(prompt_len), tuple(int(x) for x in pair.neg_ids))
        if pos_key not in pi_cache:
            pi_cache[pos_key] = self._completion_logprob(pair.pos_ids, prompt_len, pair.domain)
        if neg_key not in pi_cache:
            pi_cache[neg_key] = self._completion_logprob(pair.neg_ids, prompt_len, pair.domain)
        pi_pos = pi_cache[pos_key]
        pi_neg = pi_cache[neg_key]

        ref_pos_key = (int(prompt_len), tuple(int(x) for x in pair.pos_ids))
        ref_neg_key = (int(prompt_len), tuple(int(x) for x in pair.neg_ids))
        if ref_pos_key not in ref_cache:
            with torch.no_grad():
                ref_cache[ref_pos_key] = self._completion_logprob(pair.pos_ids, prompt_len, None)
        if ref_neg_key not in ref_cache:
            with torch.no_grad():
                ref_cache[ref_neg_key] = self._completion_logprob(pair.neg_ids, prompt_len, None)
        ref_pos = ref_cache[ref_pos_key]
        ref_neg = ref_cache[ref_neg_key]

        delta = (pi_pos - ref_pos) - (pi_neg - ref_neg)
        dpo = -F.logsigmoid(float(self.config.dpo_beta) * delta)
        pos_len = max(1, len(pair.pos_ids) - prompt_len)
        neg_len = max(1, len(pair.neg_ids) - prompt_len)
        avg_len = 0.5 * float(pos_len + neg_len)
        verbosity = avg_len / 128.0
        reg = float(self.config.sampo_verbosity_weight) * verbosity
        return dpo + reg

    def _generate_pair(self, domain: str) -> tuple[_PreferencePair | None, dict[str, float]]:
        prompt, task = self._build_prompt(domain)
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return None, {"hydra_prompt_empty": 1.0}
        prompt_t = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        candidates: list[list[int]] = []
        rewards: list[float] = []
        passes: list[bool] = []
        n_cand = max(1, int(self.config.n_candidates))
        with self._active_adapter(domain):
            # Batched candidate generation preserves sampling semantics while reducing per-candidate overhead.
            cand_prompt = prompt_t.expand(n_cand, -1).contiguous()
            cand_batch = self.model.generate(cand_prompt, max_new_tokens=24, temperature=0.85, top_k=40).detach().cpu()
            for i in range(int(cand_batch.size(0))):
                seq = cand_batch[i]
                score, passed = self._reward(domain, task, seq, len(prompt_ids))
                candidates.append([int(x) for x in seq.tolist()])
                rewards.append(float(score))
                passes.append(bool(passed))
            greedy = self.model.generate(prompt_t, max_new_tokens=24, temperature=0.2, top_k=1)[0].detach().cpu()
            g_score, g_pass = self._reward(domain, task, greedy, len(prompt_ids))
            candidates.append([int(x) for x in greedy.tolist()])
            rewards.append(float(g_score))
            passes.append(bool(g_pass))

        if not candidates:
            return None, {"hydra_candidates_empty": 1.0}
        order = sorted(range(len(candidates)), key=lambda i: rewards[i], reverse=True)
        best_i = order[0]
        worst_i = order[-1]
        pair = _PreferencePair(
            domain=domain,
            prompt_ids=[int(x) for x in prompt_ids],
            pos_ids=candidates[best_i],
            neg_ids=candidates[worst_i],
            pos_reward=float(rewards[best_i]),
            neg_reward=float(rewards[worst_i]),
        )
        metrics = {
            "hydra_pair_reward_pos": float(pair.pos_reward),
            "hydra_pair_reward_neg": float(pair.neg_reward),
            "hydra_pair_margin": float(pair.pos_reward - pair.neg_reward),
            "hydra_candidate_pass_rate": float(sum(1 for x in passes if x) / max(len(passes), 1)),
        }
        return pair, metrics

    def _refresh_anchors(self) -> None:
        for domain in self.domains:
            q = self.anchor[domain]
            q.clear()
            for prompt in self.anchor_prompts.get(domain, [])[:2]:
                p_ids = [int(x) for x in self.tokenizer.encode(prompt)]
                if not p_ids:
                    continue
                eos = int(p_ids[-1])
                q.append(
                    _PreferencePair(
                        domain=domain,
                        prompt_ids=p_ids,
                        pos_ids=p_ids + [eos],
                        neg_ids=p_ids + [eos],
                        pos_reward=0.0,
                        neg_reward=0.0,
                    )
                )

    def _sample_train_pairs(self, domain: str, new_pair: _PreferencePair) -> list[_PreferencePair]:
        out: list[_PreferencePair] = [new_pair]
        replay_q = list(self.replay[domain])
        anchor_q = list(self.anchor[domain])
        n_replay = 2
        n_anchor = 1
        if replay_q:
            for _ in range(min(n_replay, len(replay_q))):
                out.append(replay_q[self.rng.randrange(0, len(replay_q))])
        if anchor_q:
            for _ in range(min(n_anchor, len(anchor_q))):
                out.append(anchor_q[self.rng.randrange(0, len(anchor_q))])
        return out

    def _domain_eval(self, domain: str, n_tasks: int = 4) -> float:
        if domain == "text":
            prompts = self.anchor_prompts["text"][: max(1, n_tasks)]
            scores: list[float] = []
            with self._active_adapter(domain):
                for p in prompts:
                    p_ids = self.tokenizer.encode(p)
                    p_t = torch.tensor(p_ids, device=self.device, dtype=torch.long).unsqueeze(0)
                    seq = self.model.generate(p_t, max_new_tokens=24, temperature=0.75, top_k=30)[0].detach().cpu()
                    s, _ = self._reward("text", None, seq, len(p_ids))
                    scores.append(float(s))
            return float(sum(scores) / max(len(scores), 1))

        passed = 0
        total = 0
        with self._active_adapter(domain):
            for _ in range(max(1, n_tasks)):
                task = build_math_task(self.rng) if domain == "math" else build_code_task(self.rng)
                p_ids = self.tokenizer.encode(task.prompt)
                p_t = torch.tensor(p_ids, device=self.device, dtype=torch.long).unsqueeze(0)
                seq = self.model.generate(p_t, max_new_tokens=24, temperature=0.75, top_k=30)[0].detach().cpu()
                _, ok = self._reward(domain, task, seq, len(p_ids))
                passed += int(ok)
                total += 1
        return float(passed / max(total, 1))

    def _merge_phase(self, step_idx: int) -> dict[str, float]:
        deltas = {}
        for d in self.domains:
            current = self._domain_eval(d, n_tasks=3)
            delta = float(current - self.phase_start_scores.get(d, 0.0))
            deltas[d] = delta
            self.phase_start_scores[d] = current
        pos = {k: max(0.0, float(v)) for k, v in deltas.items()}
        if sum(pos.values()) <= 1e-8:
            return {"hydra_merge_skipped": 1.0}

        clipped = {k: min(0.80, max(0.05, v)) for k, v in pos.items() if v > 0.0}
        s = sum(clipped.values())
        weights = {k: (v / s) for k, v in clipped.items()}
        states: dict[str, dict[str, torch.Tensor]] = {}
        for domain in weights.keys():
            states[domain] = {}
            for name, wrapper in self._wrappers.items():
                for p_name, tensor in wrapper.adapter_state(domain).items():
                    states[domain][f"{name}:{p_name}"] = tensor.detach().cpu().float()
        merged = merge_adapter_states(
            states=states,
            weights=weights,
            cfg=MergeConfig(method=self.config.merge_method, density=self.config.merge_density, seed=self.config.seed),
        )
        if not merged:
            return {"hydra_merge_skipped": 1.0}

        for name, wrapper in self._wrappers.items():
            payload = {
                "a": merged[f"{name}:a"].to(device=self.device, dtype=wrapper.adapters[self.domains[0]].a.dtype),
                "b": merged[f"{name}:b"].to(device=self.device, dtype=wrapper.adapters[self.domains[0]].b.dtype),
            }
            wrapper.load_adapter_state("adapter_hydra_merge", payload)
            if bool(self.config.merge_fold_into_backbone):
                wrapper.fold_adapter_into_base_("adapter_hydra_merge")
        artifact = {
            "step": int(step_idx),
            "weights": {k: float(v) for k, v in weights.items()},
            "deltas": {k: float(v) for k, v in deltas.items()},
            "method": str(self.config.merge_method),
            "density": float(self.config.merge_density),
        }
        with (self.hydra_dir / f"merge_step_{int(step_idx):08d}.json").open("w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=True, indent=2)
        return {
            "hydra_merge_skipped": 0.0,
            "hydra_merge_domains": float(len(weights)),
            "hydra_merge_positive_delta_sum": float(sum(pos.values())),
        }

    def step(self, step_idx: int) -> dict[str, float]:
        if not bool(self.config.enabled):
            return {}
        if (int(step_idx) % max(1, int(self.config.update_interval))) != 0:
            return {}
        domain = self.domains[(int(step_idx) // max(1, int(self.config.update_interval))) % len(self.domains)]
        pair, pair_metrics = self._generate_pair(domain)
        if pair is None:
            return {"hydra_active": 1.0, "hydra_step_skipped": 1.0, **pair_metrics}

        self.replay[domain].append(pair)
        self.anchor[domain].append(pair)
        train_pairs = self._sample_train_pairs(domain, pair)

        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        pi_cache: dict[tuple[str, int, tuple[int, ...]], torch.Tensor] = {}
        ref_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        for p in train_pairs:
            loss = loss + self._pair_loss_cached(p, pi_cache=pi_cache, ref_cache=ref_cache)
        loss = loss / max(len(train_pairs), 1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Hydra loss is non-finite at step {step_idx}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._domain_lora_params(domain), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "hydra_active": 1.0,
            "hydra_step_skipped": 0.0,
            "hydra_domain_math": 1.0 if domain == "math" else 0.0,
            "hydra_domain_code": 1.0 if domain == "code" else 0.0,
            "hydra_domain_text": 1.0 if domain == "text" else 0.0,
            "hydra_train_pairs": float(len(train_pairs)),
            "hydra_loss": float(loss.detach().item()),
            "hydra_replay_size": float(len(self.replay[domain])),
            **pair_metrics,
        }

        if (int(step_idx) % max(1, int(self.config.rollback_interval))) == 0:
            score = self._domain_eval(domain, n_tasks=4)
            last = float(self.last_gate_scores.get(domain, score))
            if score + float(self.config.rollback_threshold) < last:
                self._restore_domain_state(domain, self.best_state[domain])
                drop_n = min(8, len(self.replay[domain]))
                for _ in range(drop_n):
                    self.replay[domain].pop()
                metrics["hydra_rollback_triggered"] = 1.0
                metrics["hydra_rollback_drop"] = float(last - score)
            else:
                self.last_gate_scores[domain] = float(score)
                self.best_state[domain] = self._capture_domain_state(domain)
                metrics["hydra_rollback_triggered"] = 0.0
                metrics["hydra_rollback_drop"] = 0.0
            metrics["hydra_eval_score"] = float(score)

        if (int(step_idx) % max(1, int(self.config.steps_per_phase))) == 0:
            self.phase_idx += 1
            merge_metrics = self._merge_phase(step_idx)
            for k, v in merge_metrics.items():
                metrics[k] = float(v)

        return metrics

    def state_dict(self) -> dict[str, Any]:
        replay_payload = {d: [p.as_dict() for p in list(q)] for d, q in self.replay.items()}
        anchor_payload = {d: [p.as_dict() for p in list(q)] for d, q in self.anchor.items()}
        return {
            "phase_idx": int(self.phase_idx),
            "phase_start_scores": {k: float(v) for k, v in self.phase_start_scores.items()},
            "last_gate_scores": {k: float(v) for k, v in self.last_gate_scores.items()},
            "replay": replay_payload,
            "anchor": anchor_payload,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.phase_idx = int(payload.get("phase_idx", self.phase_idx))
        pss = payload.get("phase_start_scores")
        if isinstance(pss, dict):
            for d in self.domains:
                self.phase_start_scores[d] = float(pss.get(d, self.phase_start_scores[d]))
        lgs = payload.get("last_gate_scores")
        if isinstance(lgs, dict):
            for d in self.domains:
                self.last_gate_scores[d] = float(lgs.get(d, self.last_gate_scores[d]))
        replay_payload = payload.get("replay")
        if isinstance(replay_payload, dict):
            for d in self.domains:
                q = self.replay[d]
                q.clear()
                items = replay_payload.get(d, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            q.append(_PreferencePair.from_dict(item))
        anchor_payload = payload.get("anchor")
        if isinstance(anchor_payload, dict):
            for d in self.domains:
                q = self.anchor[d]
                q.clear()
                items = anchor_payload.get(d, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            q.append(_PreferencePair.from_dict(item))
        opt = payload.get("optimizer")
        if isinstance(opt, dict):
            self.optimizer.load_state_dict(opt)
