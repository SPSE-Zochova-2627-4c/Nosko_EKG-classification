"""PyTorch Dataset a DataLoader pre oba modely (issue #9).

    from ekg.data.dataset import make_loader
    train = make_loader("train", "classifier", batch_size=64, shuffle=True, seed=42)
    for batch in train:
        batch["signal"]   # torch.float32 (B, 12, 1000) – po preprocessingu v1.0
        batch["target"]   # torch.float32 (B, 5), poradie contract.LABELS
        batch["ecg_id"]   # torch.int64 (B,)

Podmnožiny (subset):
    "classifier"  – záznamy pre Model B (usable_classifier)
    "strict_norm" – prísny NORM pre Model A (autoencoder)
    "all"         – všetky záznamy množiny (napr. vyhodnotenie autoencodera na val)

Test množina je zamknutá rovnako ako v ekg.data.splits (allow_test=True).
Kontrolný priechod:  uv run ekg-check-data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ekg import contract
from ekg.data import splits
from ekg.data.cache import SignalCache
from ekg.preprocessing import Preprocessor

SUBSETS = {"classifier": "classifier", "strict_norm": "strict_norm", "all": None}
LABEL_COLUMNS = list(contract.LABELS)


class EKGDataset(Dataset):
    """Záznamy jednej množiny po preprocessingu, s cieľmi Modelu B a ecg_id.

    preload=True: všetky signály sa spracujú raz pri vytvorení a držia v pamäti
    (train Modelu B ≈ 0,8 GB) – tréning potom nečaká na disk ani filter.
    preload=False: každý záznam sa číta z cache a spracuje až pri __getitem__.
    """

    def __init__(
        self,
        ecg_ids,
        manifest: pd.DataFrame,
        cache: SignalCache,
        preprocessor: Preprocessor,
        preload: bool = True,
        chunk: int = 1024,
    ):
        self.ecg_ids = np.asarray(ecg_ids, dtype=np.int64)
        # copy=True: pandas 3 môže vrátiť pole iba na čítanie, PyTorch potrebuje zapisovateľné
        self.targets = manifest.loc[self.ecg_ids, LABEL_COLUMNS].to_numpy(np.float32, copy=True)
        self.patient_ids = manifest.loc[self.ecg_ids, "patient_id"].to_numpy(np.int64, copy=True)
        self.cache = cache
        self.preprocessor = preprocessor
        self.signals: np.ndarray | None = None
        if preload:
            self.signals = np.empty(
                (len(self.ecg_ids), contract.N_LEADS, contract.N_SAMPLES), dtype=np.float32
            )
            for start in range(0, len(self.ecg_ids), chunk):
                ids = self.ecg_ids[start : start + chunk]
                self.signals[start : start + len(ids)] = preprocessor.transform(cache.get(ids))

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, i: int) -> dict:
        if self.signals is not None:
            x = self.signals[i]
        else:
            x = self.preprocessor.transform(self.cache.get([self.ecg_ids[i]])[0])
        return {
            "signal": torch.from_numpy(np.ascontiguousarray(x)),
            "target": torch.from_numpy(self.targets[i]),
            "ecg_id": int(self.ecg_ids[i]),
        }


def _seed_worker(worker_id: int) -> None:
    """Každý worker DataLoadera dostane odvodený, ale reprodukovateľný seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def make_dataset(
    split: str,
    subset: str = "classifier",
    *,
    allow_test: bool = False,
    manifest: pd.DataFrame | None = None,
    cache: SignalCache | None = None,
    preprocessor: Preprocessor | None = None,
    preload: bool = True,
) -> EKGDataset:
    if subset not in SUBSETS:
        raise ValueError(f"Neznámy subset {subset!r}, povolené: {list(SUBSETS)}")
    manifest = splits.load_manifest() if manifest is None else manifest
    ids = splits.split_ids(split, manifest, subset=SUBSETS[subset], allow_test=allow_test)
    return EKGDataset(
        ids,
        manifest,
        cache or SignalCache(),
        preprocessor or Preprocessor.load(),
        preload=preload,
    )


def make_loader(
    split: str,
    subset: str = "classifier",
    *,
    batch_size: int = 64,
    shuffle: bool | None = None,
    seed: int = 42,
    num_workers: int = 0,
    allow_test: bool = False,
    dataset: EKGDataset | None = None,
    **dataset_kwargs,
) -> DataLoader:
    """DataLoader pre množinu. Miešanie iba pre train (ak nie je povedané inak).

    Rovnaký seed => rovnaké poradie dávok v každom behu (reprodukovateľnosť).
    Validation a test sa nikdy nemiešajú a posledná neúplná dávka sa nezahadzuje.
    """
    if shuffle is None:
        shuffle = split == "train"
    if shuffle and split != "train":
        raise ValueError("Validation a test sa nemiešajú – výsledky musia byť deterministické.")
    dataset = dataset or make_dataset(split, subset, allow_test=allow_test, **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
