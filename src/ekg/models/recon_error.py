"""Rekonštrukčná chyba autoencodera – skóre anomálie (issue #12).

Definícia (verzia ERROR_VERSION):

    chyba zvodu l   e_l = 1/T · Σ_t (x[l,t] − x̂[l,t])²         MSE cez čas, T = 1000
    chyba záznamu   e   = 1/12 · Σ_l e_l                         priemer cez 12 zvodov

e je teda obyčajné MSE cez všetkých 12 000 hodnôt a zhoduje sa s loss funkciou
pri tréningu (#11). Jednotka: (z-score)², pretože signál je po normalizácii (#8).

Prečo MSE: druhá mocnina zvýrazní veľké lokálne odchýlky (napr. deformovaný QRS alebo
posunutý ST segment) viac ako MAE a je to tá istá veličina, ktorú model minimalizuje.

Uchovávajú sa aj chyby po zvodoch (B, 12) a chyba v čase (B, 1000):
- po zvodoch: analýza chýb (#14) a výstup error_per_lead v kontrakte,
- v čase: vizualizácia, kde v zázname je anomália (#20, #27),
- agregácia "max" (najhorší zvod) je alternatíva na porovnanie v #13 – anomália
  len v niekoľkých zvodoch (napr. inferiórny MI) sa priemerom cez 12 zvodov „rozriedi“.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ekg import contract

ERROR_VERSION = "1.0"
AGGREGATIONS = ("mean", "max")


def _check(x: torch.Tensor, recon: torch.Tensor) -> None:
    if x.shape != recon.shape:
        raise ValueError(f"Tvar vstupu {tuple(x.shape)} ≠ tvar rekonštrukcie {tuple(recon.shape)}.")
    if x.ndim != 3 or tuple(x.shape[1:]) != (contract.N_LEADS, contract.N_SAMPLES):
        raise ValueError(f"Očakáva sa (B, 12, 1000), dostal som {tuple(x.shape)}.")
    if not (torch.isfinite(x).all() and torch.isfinite(recon).all()):
        raise ValueError("Vstup alebo rekonštrukcia obsahuje NaN alebo nekonečno.")


def squared_error(x, recon) -> torch.Tensor:
    """(x − x̂)² tvaru (B, 12, 1000), po kontrole tvaru a konečných hodnôt."""
    x, recon = torch.as_tensor(x), torch.as_tensor(recon)
    _check(x, recon)
    return (x.float() - recon.float()) ** 2


def per_lead_error(x, recon) -> torch.Tensor:
    """MSE každého zvodu cez čas, tvar (B, 12)."""
    return squared_error(x, recon).mean(dim=2)


def error_over_time(x, recon) -> torch.Tensor:
    """Chyba v každom časovom bode, priemer cez zvody, tvar (B, 1000)."""
    return squared_error(x, recon).mean(dim=1)


def aggregate(lead_errors: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Jedno skóre na záznam z chýb po zvodoch (B, 12) -> (B,)."""
    if method == "mean":
        return lead_errors.mean(dim=1)
    if method == "max":
        return lead_errors.max(dim=1).values
    raise ValueError(f"Neznáma agregácia {method!r}, povolené: {AGGREGATIONS}")


def record_error(x, recon, method: str = "mean") -> torch.Tensor:
    """Skóre anomálie: jedno číslo na záznam, tvar (B,)."""
    return aggregate(per_lead_error(x, recon), method)


def lead_dict(lead_errors_row) -> dict[str, float]:
    """Chyby jedného záznamu po zvodoch ako {zvod: chyba} pre výstup podľa kontraktu."""
    values = np.asarray(lead_errors_row, dtype=float)
    if values.shape != (contract.N_LEADS,):
        raise ValueError(f"Očakáva sa 12 hodnôt, dostal som {values.shape}.")
    return {lead: float(v) for lead, v in zip(contract.LEADS, values, strict=True)}


@dataclass
class ErrorResult:
    """Chyby pre všetky záznamy loadera v poradí, v akom ich loader vrátil."""

    ecg_ids: np.ndarray  # (N,)
    per_lead: np.ndarray  # (N, 12)
    targets: np.ndarray  # (N, 5) – labely z loadera, na vyhodnotenie

    def record(self, method: str = "mean") -> np.ndarray:
        return aggregate(torch.from_numpy(self.per_lead), method).numpy()

    def to_frame(self, method: str = "mean"):
        import pandas as pd

        frame = pd.DataFrame(self.per_lead, columns=[f"error_{lead}" for lead in contract.LEADS])
        frame.insert(0, "error", self.record(method))
        frame[list(contract.LABELS)] = self.targets.astype(int)
        frame.index = pd.Index(self.ecg_ids, name="ecg_id")
        return frame


@torch.no_grad()
def compute_errors(
    model: torch.nn.Module, loader, device: str | torch.device = "cpu"
) -> ErrorResult:
    """Prejde loader autoencoderom v režime eval() a vráti chyby po zvodoch pre každý záznam."""
    was_training = model.training
    model.eval()  # BatchNorm používa naučené štatistiky, nie štatistiky dávky
    model.to(device)
    ids, errors, targets = [], [], []
    try:
        for batch in loader:
            x = batch["signal"].to(device)
            errors.append(per_lead_error(x, model(x)).cpu().numpy())
            ids.append(np.asarray(batch["ecg_id"]))
            targets.append(batch["target"].numpy())
    finally:
        model.train(was_training)
    return ErrorResult(np.concatenate(ids), np.concatenate(errors), np.concatenate(targets))
