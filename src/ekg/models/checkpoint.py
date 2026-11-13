"""Ukladanie a načítanie checkpointov modelov.

Checkpoint je slovník s váhami, konfiguráciou modelu a verziami všetkého, od čoho
výsledok závisí (kontrakt, preprocessing, mapovanie labelov, rozdelenie dát).
Pri načítaní sa overí, že kontrakt sedí – model natrénovaný na inom poradí zvodov
alebo inej dĺžke signálu sa nenačíta potichu.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ekg import contract
from ekg.data.labels import MAPPING_VERSION
from ekg.data.splits import SPLIT_VERSION
from ekg.preprocessing import PREPROCESSING_VERSION

CHECKPOINT_FORMAT = 1


def contract_info() -> dict:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "leads": list(contract.LEADS),
        "labels": list(contract.LABELS),
        "sampling_rate": contract.SAMPLING_RATE,
        "n_samples": contract.N_SAMPLES,
    }


def save(path: Path, model_type: str, config: dict, model: torch.nn.Module, **extra) -> None:
    """Uloží checkpoint; extra = napr. epoch, metriky, seed (iba jednoduché typy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_type": model_type,
            "config": config,
            "state_dict": model.state_dict(),
            "contract": contract_info(),
            "versions": {
                "preprocessing": PREPROCESSING_VERSION,
                "label_mapping": MAPPING_VERSION,
                "splits": SPLIT_VERSION,
                "torch": str(torch.__version__),  # TorchVersion by weights_only odmietol
            },
            "extra": extra,
        },
        path,
    )


def load(path: Path, model_type: str) -> dict:
    """Načíta checkpoint (iba dáta, bez spúšťania kódu) a overí typ a kontrakt."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Neznámy formát checkpointu: {ckpt.get('format')!r}")
    if ckpt["model_type"] != model_type:
        raise ValueError(f"Checkpoint je {ckpt['model_type']!r}, očakáva sa {model_type!r}.")
    if ckpt["contract"] != contract_info():
        raise ValueError(
            "Checkpoint bol vytvorený s iným kontraktom (zvody, labely, dĺžka signálu): "
            f"{ckpt['contract']}"
        )
    return ckpt
