"""Spoločné časti tréningu: seed, zariadenie, early stopping, log a stav pre pokračovanie."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    from ekg.check_env import pick_device as _pick

    return _pick()[0]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass
class EarlyStopping:
    """Zastaví tréning, keď sa val loss `patience` epoch nezlepšila aspoň o `min_delta`."""

    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    best_epoch: int = -1
    bad_epochs: int = 0

    def step(self, value: float, epoch: int) -> bool:
        """Vráti True, ak je toto nová najlepšia hodnota."""
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state(self, state: dict) -> None:
        self.best, self.best_epoch, self.bad_epochs = (
            state["best"],
            state["best_epoch"],
            state["bad_epochs"],
        )


class CsvLog:
    """Jeden riadok na epochu v metrics.csv; pri pokračovaní sa dopisuje."""

    def __init__(self, path: Path, fields: list[str], append: bool):
        self.path, self.fields = Path(path), fields
        if not append or not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(row)

    def read(self) -> list[dict]:
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))


def epoch_generator(seed: int, epoch: int) -> torch.Generator:
    """Poradie miešania závisí iba od seedu a epochy.

    Po pokračovaní (--resume) je preto rovnaké ako pri tréningu bez prerušenia.
    """
    return torch.Generator().manual_seed(seed * 1000 + epoch)
