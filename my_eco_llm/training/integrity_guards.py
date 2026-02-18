from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class IntegrityConfig:
    enabled: bool = True
    proof_mode: bool = False
    hidden_eval_manifest: str = ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_manifest(
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
    tracked_files: list[Path],
    seeds: dict[str, int],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for p in tracked_files:
        if p.exists() and p.is_file():
            entries.append(
                {
                    "path": str(p.as_posix()),
                    "sha256": _sha256_file(p),
                    "size": int(p.stat().st_size),
                }
            )
    manifest = {
        "config_hash": _sha256_json(config_payload),
        "seeds": dict(seeds),
        "tracked_files": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "integrity_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    return manifest


def ensure_proof_inputs(cfg: IntegrityConfig) -> None:
    if not cfg.proof_mode:
        return
    if not cfg.hidden_eval_manifest:
        raise RuntimeError("Proof mode requires --integrity-hidden-eval-manifest.")
    p = Path(cfg.hidden_eval_manifest)
    if not p.exists():
        raise RuntimeError(f"Hidden eval manifest does not exist: {p}")
    if not p.is_file():
        raise RuntimeError(f"Hidden eval manifest is not a file: {p}")


def check_feature_effective_calls(
    *,
    proof_mode: bool,
    feature_calls: dict[str, int],
) -> None:
    if not proof_mode:
        return
    missing = [k for k, v in feature_calls.items() if int(v) <= 0]
    if missing:
        raise RuntimeError(
            "Proof run invalid: claimed features had zero effective calls: " + ", ".join(sorted(missing))
        )

