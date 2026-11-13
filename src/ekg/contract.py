"""Kontrakt vstupov a výstupov systému (issue #2).

Jediné miesto, kde je definované poradie zvodov, poradie labelov, tvar vstupu
a štruktúra výstupu. Ostatný kód (loader, modely, inference, export) importuje
konštanty odtiaľto, aby sa poradie nikde nerozišlo.

Podrobný popis s príkladmi je v docs/kontrakt.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

# --- Vstup -------------------------------------------------------------------

SCHEMA_VERSION = "1.1"
# Verzie, ktoré vie from_dict() načítať. 1.1 pridala voliteľné metadata.age_over_89 (#21);
# staršie výstupy sa pri načítaní povýšia na aktuálnu verziu.
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")

# Poradie zvodov presne podľa PTB-XL (atribút sig_name vo WFDB záznamoch).
LEADS: tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)  # fmt: skip
N_LEADS = len(LEADS)  # 12
SAMPLING_RATE = 100  # Hz
DURATION_S = 10  # sekúnd
N_SAMPLES = SAMPLING_RATE * DURATION_S  # 1000
SIGNAL_UNIT = "mV"

# --- Výstup ------------------------------------------------------------------

# Poradie výstupov Modelu B (index i v tenzore (B, 5) = LABELS[i]).
LABELS: tuple[str, ...] = ("MI", "STTC", "CD", "HYP", "AFIB")
N_LABELS = len(LABELS)  # 5

DISCLAIMER = (
    "Výsledok modelu strojového učenia na výskumné a vzdelávacie účely, nie lekárska diagnóza."
)

QualityStatus = Literal["ok", "warning", "unknown"]
Sex = Literal["male", "female"]


class InvalidInputError(ValueError):
    """Vstupný signál nespĺňa kontrakt; inference sa nespustí."""


# --- Validácia a prevod vstupu ----------------------------------------------


def validate_signal(signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Overí jeden záznam v tvare modelu (12, 1000) a vráti ho ako float32.

    Vyhodí InvalidInputError s popisom problému, ak:
    - vzorkovacia frekvencia nie je 100 Hz (prevzorkovanie v1 nepodporuje),
    - tvar nie je (12, 1000) – vrátane prípadu, keď je záznam otočený (1000, 12),
    - signál nie je číselný alebo obsahuje NaN / nekonečno.

    Kvalitu signálu (plochý zvod, šum, …) nerieši – to je kontrola kvality (#7),
    ktorá záznam neodmieta, ale vráti varovanie vo výstupe.
    """
    if sampling_rate != SAMPLING_RATE:
        raise InvalidInputError(
            f"Vzorkovacia frekvencia {sampling_rate} Hz nie je podporovaná, očakáva sa "
            f"{SAMPLING_RATE} Hz. Signál treba najprv prevzorkovať."
        )

    arr = np.asarray(signal)
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise InvalidInputError(f"Signál musí byť reálne číslo, nie {arr.dtype}.")

    expected = (N_LEADS, N_SAMPLES)
    if arr.shape != expected:
        hint = ""
        if arr.shape == expected[::-1]:
            hint = " Záznam je vo formáte WFDB (1000, 12) – použi from_wfdb()."
        raise InvalidInputError(f"Tvar signálu {arr.shape}, očakáva sa {expected}.{hint}")

    if not np.isfinite(arr).all():
        bad_leads = [LEADS[i] for i in np.where(~np.isfinite(arr).all(axis=1))[0]]
        raise InvalidInputError(f"Signál obsahuje NaN alebo nekonečno v zvodoch: {bad_leads}.")

    return arr.astype(np.float32, copy=False)


def from_wfdb(
    p_signal: np.ndarray,
    sig_names: list[str] | tuple[str, ...] | None = None,
    sampling_rate: int = SAMPLING_RATE,
) -> np.ndarray:
    """Prevedie WFDB záznam (1000, 12) na vstup modelu (12, 1000).

    Ak sú zadané názvy zvodov (record.sig_name), zvody sa preusporiadajú
    do poradia LEADS a chýbajúci zvod vyhodí InvalidInputError.
    """
    arr = np.asarray(p_signal)
    if arr.ndim != 2 or arr.shape[1] != N_LEADS:
        raise InvalidInputError(
            f"WFDB signál musí mať tvar (vzorky, {N_LEADS}), dostal som {arr.shape}."
        )
    if sig_names is not None:
        names = [s.upper() for s in sig_names]
        missing = [lead for lead in LEADS if lead not in names]
        if missing:
            raise InvalidInputError(f"Chýbajú zvody {missing}; záznam má {list(sig_names)}.")
        arr = arr[:, [names.index(lead) for lead in LEADS]]
    return validate_signal(arr.T, sampling_rate)


def to_batch(signals: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Spojí overené záznamy do dávky (B, 12, 1000) typu float32."""
    return np.stack([validate_signal(s) for s in signals])


# --- Štruktúra výstupu -------------------------------------------------------


@dataclass
class AnomalyResult:
    """Výstup Modelu A (autoencoder)."""

    is_anomaly: bool
    reconstruction_error: float  # jedno skóre na záznam (definícia v #12)
    threshold: float  # kalibrovaný na validation (#13)
    error_per_lead: dict[str, float]  # chyba po zvodoch, kľúče = LEADS


@dataclass
class LabelResult:
    """Výstup Modelu B pre jeden label."""

    probability: float  # sigmoid skóre v [0, 1]
    threshold: float  # prah tohto labelu z validation (#17)
    positive: bool  # probability >= threshold


@dataclass
class Consistency:
    """Súlad medzi Modelom A a Modelom B (presné pravidlo dopĺňa #19)."""

    models_agree: bool
    note: str | None = None


@dataclass
class Quality:
    """Kvalita záznamu (#7). 'unknown', kým kontrola kvality nie je hotová."""

    status: QualityStatus = "unknown"
    issues: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    """Dostupné metadáta záznamu. Chýbajúce hodnoty sú None, nikdy sa nevymýšľajú."""

    age: int | None = None
    age_over_89: bool = False  # PTB-XL anonymizuje vek > 89 (uvádza 300) → age = None, toto True
    sex: Sex | None = None
    source: str | None = None  # napr. "PTB-XL" alebo názov súboru


@dataclass
class PredictionResult:
    """Kompletný výstup systému pre jeden EKG záznam."""

    record_id: str
    model_versions: dict[str, str]
    anomaly: AnomalyResult
    labels: dict[str, LabelResult]  # kľúče presne LABELS, v tomto poradí
    consistency: Consistency
    quality: Quality = field(default_factory=Quality)
    metadata: Metadata = field(default_factory=Metadata)
    schema_version: str = SCHEMA_VERSION
    disclaimer: str = DISCLAIMER

    def __post_init__(self) -> None:
        if tuple(self.labels) != LABELS:
            raise ValueError(f"Labely musia byť {LABELS} v tomto poradí, sú {tuple(self.labels)}.")
        if tuple(self.anomaly.error_per_lead) != LEADS:
            raise ValueError(f"error_per_lead musí mať kľúče {LEADS} v tomto poradí.")
        for name, label in self.labels.items():
            if not 0.0 <= label.probability <= 1.0 or math.isnan(label.probability):
                raise ValueError(f"Pravdepodobnosť {name} = {label.probability} nie je v [0, 1].")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> PredictionResult:
        """Načíta výstup späť zo slovníka (napr. z JSON exportu) a znovu ho overí."""
        if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Nepodporovaná verzia schémy {data.get('schema_version')!r}, "
                f"podporované sú {SUPPORTED_SCHEMA_VERSIONS}."
            )
        return cls(
            record_id=data["record_id"],
            model_versions=dict(data["model_versions"]),
            anomaly=AnomalyResult(**data["anomaly"]),
            labels={name: LabelResult(**value) for name, value in data["labels"].items()},
            consistency=Consistency(**data["consistency"]),
            quality=Quality(**data["quality"]),
            metadata=Metadata(**data["metadata"]),
            schema_version=SCHEMA_VERSION,  # staršia verzia sa povýši
            disclaimer=data["disclaimer"],
        )


def sex_from_ptbxl(value: float | int | None) -> Sex | None:
    """PTB-XL kóduje pohlavie ako 0 = muž, 1 = žena."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return {0: "male", 1: "female"}.get(int(value))
