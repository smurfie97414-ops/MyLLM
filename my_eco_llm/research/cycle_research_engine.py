from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ResearchHypothesis:
    hypothesis_id: str
    title: str
    mechanism: str
    expected_gain: str
    sources: list[str]


def latest_disruptive_hypotheses() -> list[ResearchHypothesis]:
    # Curated high-impact set used by Prometheus cycle controller.
    return [
        ResearchHypothesis(
            hypothesis_id="seal_rlvrr_credit",
            title="SEAL-style self-editing + RLVRR reward-chain for verifier-grounded adaptation",
            mechanism="Use self-edit and verifiable reward-chain signals to optimize update directives and replay quality.",
            expected_gain=">15% composite in early cycles via better sample efficiency.",
            sources=[
                "https://arxiv.org/abs/2506.10943",
                "https://arxiv.org/abs/2601.18533",
            ],
        ),
        ResearchHypothesis(
            hypothesis_id="c3o_second_order_credit",
            title="Counterfactual credit on top of second-order preconditioning",
            mechanism="Combine curvature-aware updates with counterfactual reward attribution to scale useful blocks.",
            expected_gain="Faster convergence at fixed step budget.",
            sources=[
                "https://arxiv.org/abs/2510.09378",
                "https://arxiv.org/abs/2601.02417",
            ],
        ),
        ResearchHypothesis(
            hypothesis_id="ib_router",
            title="Information bottleneck routing for MoE",
            mechanism="Reciprocal bottleneck objective to keep predictive information while reducing routing collapse.",
            expected_gain="Higher verifier pass with stable MoE usage.",
            sources=[
                "https://arxiv.org/abs/2505.16950",
            ],
        ),
    ]


def write_research_log(output_root: str | Path, cycle_idx: int) -> Path:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_root / f"RESEARCH_LOG_{stamp}_cycle{cycle_idx:03d}.md"
    hyps = latest_disruptive_hypotheses()
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Research Log - Cycle {cycle_idx}\n\n")
        f.write("UTC timestamp: " + datetime.utcnow().isoformat() + "\n\n")
        for h in hyps:
            f.write(f"## {h.hypothesis_id}\n")
            f.write(f"- Title: {h.title}\n")
            f.write(f"- Mechanism: {h.mechanism}\n")
            f.write(f"- Expected gain: {h.expected_gain}\n")
            f.write("- Sources:\n")
            for s in h.sources:
                f.write(f"  - {s}\n")
            f.write("\n")
    return path


def choose_top_hypothesis(cycle_idx: int) -> ResearchHypothesis:
    hyps = latest_disruptive_hypotheses()
    # Rotating pick prevents single-hypothesis lock-in across failure cycles.
    return hyps[(cycle_idx - 1) % len(hyps)]


def append_cycle_decision(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
