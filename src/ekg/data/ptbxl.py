"""Načítanie PTB-XL: anotácie (CSV) a 100 Hz signály cez WFDB (issue #3).

    from ekg.data import ptbxl
    db = ptbxl.load_database()            # DataFrame, index = ecg_id
    scp = ptbxl.load_scp_statements()     # DataFrame, index = SCP kód
    x = ptbxl.load_signal(db.loc[1, "filename_lr"])   # np.ndarray (12, 1000), float32

Overenie celého datasetu:  uv run ekg-verify-data
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from ekg import contract
from ekg.data.download import PTBXL_DIR


class DatasetNotFoundError(FileNotFoundError):
    """PTB-XL nie je stiahnutý alebo chýba súbor."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"Nenašiel som {path}. Stiahni dataset príkazom: uv run ekg-download"
        )
    return path


def load_database(root: Path = PTBXL_DIR) -> pd.DataFrame:
    """ptbxl_database.csv; stĺpec scp_codes je prevedený z textu na dict {kód: likelihood}."""
    db = pd.read_csv(_require(root / "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    return db


def load_scp_statements(root: Path = PTBXL_DIR) -> pd.DataFrame:
    """scp_statements.csv – popis 71 SCP kódov (diagnostic_class, rhythm, …)."""
    scp = pd.read_csv(_require(root / "scp_statements.csv"), index_col=0)
    scp.index.name = "scp_code"
    return scp


def load_signal(filename_lr: str, root: Path = PTBXL_DIR) -> np.ndarray:
    """Načíta jeden 100 Hz záznam a vráti ho v tvare kontraktu (12, 1000), float32.

    filename_lr je hodnota stĺpca z ptbxl_database.csv, napr. 'records100/00000/00001_lr'.
    """
    path = root / filename_lr
    _require(path.with_suffix(".hea"))
    record = wfdb.rdrecord(str(path))
    return contract.from_wfdb(record.p_signal, record.sig_name, record.fs)


def verify_dataset(root: Path = PTBXL_DIR) -> list[str]:
    """Načíta všetky záznamy a vráti zoznam problémov. Nič nevynecháva potichu."""
    db = load_database(root)
    scp = load_scp_statements(root)
    problems: list[str] = []

    unknown = {code for codes in db["scp_codes"] for code in codes} - set(scp.index)
    if unknown:
        problems.append(f"SCP kódy chýbajúce v scp_statements.csv: {sorted(unknown)}")

    for ecg_id, filename in tqdm(db["filename_lr"].items(), total=len(db), desc="načítavam"):
        try:
            x = load_signal(filename, root)
            if x.shape != (contract.N_LEADS, contract.N_SAMPLES):  # pre istotu, kontrakt to stráži
                problems.append(f"ecg_id {ecg_id}: tvar {x.shape}")
        except Exception as exc:  # zámerne široké – každý problém chceme vidieť v reporte
            problems.append(f"ecg_id {ecg_id} ({filename}): {type(exc).__name__}: {exc}")
    return problems


def main() -> int:
    db = load_database()
    print(f"Záznamov: {len(db)}, pacientov: {db['patient_id'].nunique()}")
    print(f"SCP kódov v scp_statements.csv: {len(load_scp_statements())}")
    problems = verify_dataset()
    if problems:
        print(f"\nOverenie ZLYHALO ({len(problems)} problémov), prvých 20:")
        for problem in problems[:20]:
            print("  -", problem)
        return 1
    print(f"\nOK: všetkých {len(db)} záznamov má tvar (12, 1000) pri 100 Hz a konečné hodnoty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
