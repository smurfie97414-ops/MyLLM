from __future__ import annotations

import torch
import torch.nn as nn

from .bitlinear import RMSNorm


class ConditionalNgramMemory(nn.Module):
    """
    Lightweight hashed n-gram memory (Engram-inspired).

    - Hashes token n-grams to fixed memory slots.
    - Retrieves trainable memory vectors and blends them into hidden states.
    - Optional metric sampling tracks memory-slot diversity.
    """

    def __init__(
        self,
        d_model: int,
        memory_slots: int,
        ngram: int = 3,
        dropout: float = 0.0,
        metrics_interval: int = 10,
    ) -> None:
        super().__init__()
        if memory_slots <= 0:
            raise ValueError(f"memory_slots must be > 0, got {memory_slots}.")
        if ngram <= 0:
            raise ValueError(f"ngram must be > 0, got {ngram}.")
        self.d_model = d_model
        self.memory_slots = int(memory_slots)
        self.ngram = int(ngram)
        self.metrics_interval = max(1, int(metrics_interval))

        self.memory = nn.Embedding(self.memory_slots, d_model)
        self.memory_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("last_unique_slot_fraction", torch.tensor(0.0), persistent=False)
        self._metric_step = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.memory.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate)

    def _hash_ngrams(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be rank-2 [B,T], got shape {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        ids = input_ids.long()
        if self.ngram == 1:
            return torch.remainder(ids, self.memory_slots)

        padded = torch.cat(
            [
                torch.zeros((bsz, self.ngram - 1), device=device, dtype=ids.dtype),
                ids,
            ],
            dim=1,
        )
        slots = torch.zeros((bsz, seq_len), device=device, dtype=torch.long)
        # Large odd base for rolling hash.
        base = 1_315_423_911
        for k in range(self.ngram):
            token_ids = padded[:, k : k + seq_len]
            slots = torch.remainder(slots * base + token_ids, self.memory_slots)
        return slots

    @torch.no_grad()
    def _update_metrics(self, slot_ids: torch.Tensor) -> None:
        self._metric_step += 1
        if (self._metric_step % self.metrics_interval) != 0:
            return
        unique = torch.unique(slot_ids).numel()
        total_slots = max(1, self.memory_slots)
        self.last_unique_slot_fraction.fill_(float(unique) / float(total_slots))

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be rank-3 [B,T,D], got shape {tuple(hidden.shape)}")
        slot_ids = self._hash_ngrams(input_ids)
        mem = self.memory(slot_ids)
        mem = self.memory_norm(mem)
        mem = self.dropout(mem)
        gate = torch.sigmoid(self.gate).to(dtype=hidden.dtype)
        self._update_metrics(slot_ids)
        return hidden + gate * mem

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "memory_gate": float(torch.sigmoid(self.gate).item()),
            "memory_unique_slot_fraction": float(self.last_unique_slot_fraction.item()),
            "memory_slots": float(self.memory_slots),
        }
