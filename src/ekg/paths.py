"""Cesty k adresárom projektu.

Dáta, checkpointy a výstupy sú mimo gitu. Ich umiestnenie sa dá zmeniť
premennými prostredia (napr. keď je PTB-XL na inom disku na trénovacom PC):

    EKG_DATA_DIR, EKG_CHECKPOINT_DIR, EKG_OUTPUT_DIR
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: str) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / default


DATA_DIR = _dir("EKG_DATA_DIR", "data")
CHECKPOINT_DIR = _dir("EKG_CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR = _dir("EKG_OUTPUT_DIR", "outputs")
CONFIG_DIR = PROJECT_ROOT / "configs"
