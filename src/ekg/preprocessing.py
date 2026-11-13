"""Preprocessing EKG: filtrovanie a normalizácia po zvodoch bez úniku informácií (issue #8).

Kroky (verzia PREPROCESSING_VERSION):
1. Hornopriepustný Butterworth filter 0,5 Hz (2. rád, nulová fáza – sosfiltfilt)
   odstráni drift základnej línie (dýchanie, pohyb). Nulová fáza nedeformuje ST segment.
   Nič sa z dát neučí, takže nemôže spôsobiť únik.
2. Z-score po zvodoch: (x - mean[zvod]) / std[zvod]. mean a std sa počítajú
   IBA z train množiny (po filtrovaní) a ukladajú sa do manifests/preprocessing_v1.0.json.
   Validation, test aj inference používajú tie isté uložené hodnoty.

Prečo nie normalizácia každého záznamu zvlášť: tá by zmazala absolútnu amplitúdu,
ktorá nesie diagnostickú informáciu (napr. napäťové kritériá hypertrofie – HYP).

Spustenie:  uv run ekg-preprocess   # cache signálov (raz) + fit na train + uloženie parametrov
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

from ekg import contract
from ekg.paths import PROJECT_ROOT

PREPROCESSING_VERSION = "1.0"
PARAMS_PATH = PROJECT_ROOT / "manifests" / f"preprocessing_v{PREPROCESSING_VERSION}.json"
MIN_STD = 1e-6  # mV – menšia smerodajná odchýlka = prakticky plochý zvod


@dataclass
class Preprocessor:
    highpass_hz: float | None = 0.5
    filter_order: int = 2
    sampling_rate: int = contract.SAMPLING_RATE
    mean: list[float] | None = None  # po zvodoch, poradie contract.LEADS
    std: list[float] | None = None
    fitted_on: dict = field(default_factory=dict)  # popis dát, z ktorých sa počítali mean/std
    version: str = PREPROCESSING_VERSION

    # --- filtrovanie (bez učenia) -------------------------------------------

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Hornopriepustný filter pozdĺž času; x tvaru (12, 1000) alebo (B, 12, 1000)."""
        if self.highpass_hz is None:
            return np.asarray(x, dtype=np.float32)
        sos = sps.butter(
            self.filter_order,
            self.highpass_hz,
            btype="highpass",
            fs=self.sampling_rate,
            output="sos",
        )
        return sps.sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    # --- učenie parametrov (iba train) ---------------------------------------

    def fit(self, batches, fitted_on: dict | None = None) -> Preprocessor:
        """Vypočíta mean a std po zvodoch z dávok (B, 12, 1000) – volať IBA s train dátami."""
        total = 0
        s = np.zeros(contract.N_LEADS, dtype=np.float64)
        sq = np.zeros(contract.N_LEADS, dtype=np.float64)
        for batch in batches:
            batch = self._check_shape(batch, batched=True)
            filtered = self.filter(batch).astype(np.float64)
            s += filtered.sum(axis=(0, 2))
            sq += (filtered**2).sum(axis=(0, 2))
            total += filtered.shape[0] * filtered.shape[2]
        if total == 0:
            raise ValueError("fit() nedostal žiadne dáta.")
        mean = s / total
        std = np.sqrt(np.maximum(sq / total - mean**2, 0.0))
        flat = [contract.LEADS[i] for i in np.where(std < MIN_STD)[0]]
        if flat:
            raise ValueError(f"Nulová variancia v zvodoch {flat} – nedá sa normalizovať.")
        self.mean, self.std = mean.tolist(), std.tolist()
        self.fitted_on = dict(fitted_on or {}, n_values_per_lead=int(total))
        return self

    # --- použitie -------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Filter + normalizácia; vstup (12, 1000) alebo (B, 12, 1000), výstup float32."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor nie je natrénovaný – zavolaj fit() alebo load().")
        x = self._check_shape(x, batched=np.ndim(x) == 3)
        mean = np.asarray(self.mean, dtype=np.float32)[:, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None]
        out = (self.filter(x) - mean) / std
        if not np.isfinite(out).all():
            raise ValueError("Po preprocessingu vznikli NaN alebo nekonečné hodnoty.")
        return out.astype(np.float32, copy=False)

    # --- uloženie -------------------------------------------------------------

    def save(self, path: Path = PARAMS_PATH) -> None:
        if not self.is_fitted:
            raise RuntimeError("Nedá sa uložiť nenatrénovaný preprocessor.")
        data = asdict(self)
        data["leads"] = list(contract.LEADS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> Preprocessor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != PREPROCESSING_VERSION:
            raise ValueError(
                f"Parametre majú verziu {data.get('version')!r}, kód {PREPROCESSING_VERSION!r}."
            )
        if data.pop("leads") != list(contract.LEADS):
            raise ValueError("Poradie zvodov v parametroch nezodpovedá kontraktu.")
        return cls(**data)

    # --- pomocné --------------------------------------------------------------

    @staticmethod
    def _check_shape(x: np.ndarray, batched: bool) -> np.ndarray:
        x = np.asarray(x)
        expected = (contract.N_LEADS, contract.N_SAMPLES)
        if (batched and x.shape[1:] != expected) or (not batched and x.shape != expected):
            raise contract.InvalidInputError(
                f"Tvar {x.shape}, očakáva sa {'(B, ' if batched else '('}12, 1000)."
            )
        return x


def _batches(cache, ids, size: int = 512):
    for start in range(0, len(ids), size):
        yield cache.get(ids[start : start + size])


def main() -> int:
    from ekg.data import cache as cache_mod
    from ekg.data import splits

    manifest = splits.load_manifest()
    if not cache_mod.cache_matches(manifest):
        print("Vytváram cache signálov (raz, ~4 min)…")
        cache_mod.build_cache(manifest)
    cache = cache_mod.SignalCache()

    train_ids = splits.split_ids("train", manifest)  # iba train – val/test sa nepoužijú
    manifest_sha = hashlib.sha256(splits.MANIFEST_PATH.read_bytes()).hexdigest()
    prep = Preprocessor().fit(
        _batches(cache, train_ids),
        fitted_on={
            "split": "train",
            "n_records": len(train_ids),
            "manifest": splits.MANIFEST_PATH.name,
            "manifest_sha256": manifest_sha,
        },
    )
    prep.save()
    print(f"Uložené: {PARAMS_PATH.relative_to(PROJECT_ROOT)} (fit na {len(train_ids)} train)\n")
    print(f"{'zvod':<5} {'mean [mV]':>10} {'std [mV]':>9}")
    for lead, m, s in zip(contract.LEADS, prep.mean, prep.std, strict=True):
        print(f"{lead:<5} {m:>10.4f} {s:>9.4f}")

    print("\nKontrola na validation (po normalizácii by mal byť priemer ≈ 0, std ≈ 1):")
    val = prep.transform(cache.get(splits.split_ids("val", manifest)))
    print(f"  tvar {val.shape}, priemer {val.mean():.3f}, std {val.std():.3f}, konečné: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
