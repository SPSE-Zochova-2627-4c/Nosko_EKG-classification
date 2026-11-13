"""Cache surových 100 Hz signálov v jednom .npy súbore (memmap).

Načítanie 21 799 záznamov cez WFDB trvá ~4 minúty; z cache trvá zlomok sekundy
a dá sa čítať po kúskoch bez načítania celého súboru do pamäte (np.load mmap_mode="r").

Cache obsahuje NESPRACOVANÉ signály (iba prevedené na tvar kontraktu (12, 1000), float32)
v poradí riadkov manifestu. Filtrovanie a normalizácia sa robia až v ekg.preprocessing,
takže zmena preprocessingu nevyžaduje novú cache.

    data/processed/signals_100hz.npy        (N, 12, 1000) float32, ~1 GB
    data/processed/signals_100hz_ids.npy    (N,) ecg_id v rovnakom poradí
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ekg import contract, paths
from ekg.data import ptbxl, splits

CACHE_DIR = paths.DATA_DIR / "processed"
SIGNALS_PATH = CACHE_DIR / "signals_100hz.npy"
IDS_PATH = CACHE_DIR / "signals_100hz_ids.npy"


class SignalCache:
    """Prístup k signálom z cache podľa ecg_id."""

    def __init__(self, signals_path: Path = SIGNALS_PATH, ids_path: Path = IDS_PATH):
        if not signals_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"Cache {signals_path} neexistuje. Vytvor ju: uv run ekg-preprocess"
            )
        self.signals = np.load(signals_path, mmap_mode="r")
        self.ids = np.load(ids_path)
        self._row = {int(ecg_id): i for i, ecg_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, ecg_ids) -> np.ndarray:
        try:
            return np.array([self._row[int(e)] for e in ecg_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"ecg_id {exc.args[0]} nie je v cache.") from None

    def get(self, ecg_ids) -> np.ndarray:
        """Signály (B, 12, 1000) float32 pre zoznam ecg_id (kópia v pamäti)."""
        rows = self.rows(ecg_ids)
        order = np.argsort(rows)  # memmap sa číta rýchlejšie v poradí súboru
        out = np.empty((len(rows), contract.N_LEADS, contract.N_SAMPLES), dtype=np.float32)
        out[order] = self.signals[rows[order]]
        return out


def build_cache(
    manifest=None,
    root: Path = ptbxl.PTBXL_DIR,
    signals_path: Path = SIGNALS_PATH,
    ids_path: Path = IDS_PATH,
) -> None:
    """Načíta všetky záznamy manifestu cez WFDB a uloží ich do cache."""
    manifest = splits.load_manifest() if manifest is None else manifest
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = signals_path.with_name(signals_path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float32,
        shape=(len(manifest), contract.N_LEADS, contract.N_SAMPLES),
    )
    for i, filename in enumerate(tqdm(manifest["filename_lr"], desc="cache signálov")):
        out[i] = ptbxl.load_signal(filename, root)
    out.flush()
    del out
    tmp.replace(signals_path)
    np.save(ids_path, manifest.index.to_numpy(dtype=np.int64))


def cache_matches(manifest, ids_path: Path = IDS_PATH, signals_path: Path = SIGNALS_PATH) -> bool:
    """True, ak cache existuje a obsahuje presne záznamy manifestu v rovnakom poradí."""
    if not ids_path.exists() or not signals_path.exists():
        return False
    return np.array_equal(np.load(ids_path), manifest.index.to_numpy(dtype=np.int64))
