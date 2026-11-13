"""Rozdelenie PTB-XL na train / validation / test bez prekrývania pacientov (issue #6).

Používa sa odporúčané rozdelenie autorov PTB-XL podľa stĺpca strat_fold:
folds 1–8 = train, 9 = val, 10 = test. Foldy sú rozdelené podľa pacientov
a stratifikované podľa labelov; foldy 9 a 10 majú anotácie overené človekom.

Výsledok je manifest manifests/splits_v<verzia>.csv (v gite), ktorý obsahuje
pre každý záznam split, labely a príznak prísneho NORM. Ostatný kód načíta
splity iba cez load_manifest() / split_ids().

Test množina je ZAMKNUTÁ: split_ids("test") vyhodí chybu, ak sa explicitne
nepovolí allow_test=True. Použiť sa má iba pri záverečnom vyhodnotení (#22).

Spustenie:  uv run ekg-splits          # vytvorí manifest a vypíše report
            uv run ekg-splits --check  # overí, že manifest v gite sedí s dátami
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ekg import contract
from ekg.data import labels
from ekg.paths import PROJECT_ROOT

SPLIT_VERSION = "1.0"
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)
SPLITS = ("train", "val", "test")

MANIFEST_DIR = PROJECT_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / f"splits_v{SPLIT_VERSION}.csv"
LABEL_COLUMNS = list(contract.LABELS)


class SplitLeakError(RuntimeError):
    """Ten istý pacient alebo záznam je vo viacerých množinách."""


class TestSetLockedError(PermissionError):
    """Pokus použiť test množinu bez explicitného povolenia."""


def assign_split(strat_fold: pd.Series) -> pd.Series:
    """strat_fold -> 'train' / 'val' / 'test'. Neznámy fold je chyba."""
    split = pd.Series(
        np.select(
            [
                strat_fold.isin(TRAIN_FOLDS),
                strat_fold.isin(VAL_FOLDS),
                strat_fold.isin(TEST_FOLDS),
            ],
            SPLITS,
            default="",
        ),
        index=strat_fold.index,
        name="split",
    )
    unknown = sorted(strat_fold[split == ""].unique())
    if unknown:
        raise ValueError(f"Neznáme hodnoty strat_fold: {unknown}")
    return split


def check_no_leak(manifest: pd.DataFrame) -> None:
    """Overí, že žiadny pacient ani záznam nie je vo viac ako jednej množine."""
    if manifest.index.duplicated().any():
        raise SplitLeakError("Duplicitné ecg_id v manifeste.")
    splits_per_patient = manifest.groupby("patient_id")["split"].nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    if not leaked.empty:
        raise SplitLeakError(
            f"{len(leaked)} pacientov je vo viacerých množinách, napr. {list(leaked.index[:5])}"
        )


def build_manifest(db: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Manifest: 1 riadok = 1 záznam, index ecg_id."""
    targets = labels.make_targets(db, scp)
    manifest = pd.DataFrame(
        {
            "patient_id": db["patient_id"].astype(int),
            "strat_fold": db["strat_fold"].astype(int),
            "split": assign_split(db["strat_fold"]),
            "filename_lr": db["filename_lr"],
        },
        index=db.index,
    )
    manifest[LABEL_COLUMNS] = targets[LABEL_COLUMNS]
    manifest["usable_classifier"] = targets["usable"]
    manifest["strict_norm"] = labels.strict_norm(db, scp)
    manifest["mapping_version"] = labels.MAPPING_VERSION
    manifest.index.name = "ecg_id"
    check_no_leak(manifest)
    return manifest


def save_manifest(manifest: pd.DataFrame, path: Path = MANIFEST_PATH) -> str:
    """Uloží manifest a vráti jeho SHA256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Načíta manifest a znovu overí, že nedochádza k úniku medzi množinami."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest {path} neexistuje. Vytvor ho: uv run ekg-splits")
    manifest = pd.read_csv(path, index_col="ecg_id")
    check_no_leak(manifest)
    return manifest


def split_ids(
    split: str,
    manifest: pd.DataFrame | None = None,
    *,
    subset: str | None = None,
    allow_test: bool = False,
) -> pd.Index:
    """ecg_id záznamov v množine.

    subset: None = všetky, "classifier" = iba záznamy pre Model B,
            "strict_norm" = iba prísny NORM pre Model A.
    """
    if split not in SPLITS:
        raise ValueError(f"Neznámy split {split!r}, povolené sú {SPLITS}.")
    if split == "test" and not allow_test:
        raise TestSetLockedError(
            "Test množina je zamknutá pre záverečné vyhodnotenie (#22). "
            "Ak ju naozaj potrebuješ, zavolaj split_ids('test', allow_test=True)."
        )
    manifest = load_manifest() if manifest is None else manifest
    mask = manifest["split"] == split
    if subset == "classifier":
        mask &= manifest["usable_classifier"].astype(bool)
    elif subset == "strict_norm":
        mask &= manifest["strict_norm"].astype(bool)
    elif subset is not None:
        raise ValueError(f"Neznámy subset {subset!r}.")
    return manifest.index[mask]


def report(manifest: pd.DataFrame) -> str:
    """Počty záznamov, pacientov a labelov v každej množine."""
    by = manifest.groupby("split")
    table = pd.DataFrame(
        {
            "záznamy": by.size(),
            "pacienti": by["patient_id"].nunique(),
            "Model B": by["usable_classifier"].sum(),
            "prísny NORM": by["strict_norm"].sum(),
        }
    ).reindex(SPLITS)
    usable = manifest[manifest["usable_classifier"].astype(bool)]
    positives = usable.groupby("split")[LABEL_COLUMNS].sum().reindex(SPLITS).T
    share = (positives / usable.groupby("split").size().reindex(SPLITS) * 100).round(1)
    share_table = positives.astype(str) + " (" + share.astype(str) + " %)"
    return "\n".join(
        [
            f"Rozdelenie v{SPLIT_VERSION} (strat_fold 1–8 / 9 / 10), mapovanie labelov "
            f"v{manifest['mapping_version'].iloc[0]}",
            "",
            table.to_string(),
            "",
            "Pozitívne labely v Modeli B (podiel v množine):",
            share_table.to_string(),
            "",
            "Prekrývanie pacientov medzi množinami: 0 (overené)",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    from ekg.data import ptbxl

    parser = argparse.ArgumentParser(description="Vytvorí alebo overí manifest splitov.")
    parser.add_argument("--check", action="store_true", help="iba overiť manifest v gite")
    args = parser.parse_args(argv)

    manifest = build_manifest(ptbxl.load_database(), ptbxl.load_scp_statements())
    if args.check:
        committed = load_manifest()
        if not committed.equals(pd.read_csv(_roundtrip(manifest), index_col="ecg_id")):
            print(f"CHYBA: {MANIFEST_PATH.name} nezodpovedá dátam a kódu.")
            return 1
        print(f"OK: {MANIFEST_PATH.name} zodpovedá dátam a kódu.")
    else:
        digest = save_manifest(manifest)
        print(f"Uložené: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}  (SHA256 {digest[:16]}…)\n")
    print(report(manifest))
    return 0


def _roundtrip(manifest: pd.DataFrame):
    """Manifest prevedený cez CSV, aby sa dal porovnať s načítaným súborom."""
    import io

    buffer = io.StringIO()
    manifest.to_csv(buffer, lineterminator="\n")
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    sys.exit(main())
