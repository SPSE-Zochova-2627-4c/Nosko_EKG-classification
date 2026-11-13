"""Kontrola kvality EKG záznamu (issue #7).

Dve úrovne kontroly:

1. TECHNICKÁ PLATNOSŤ – rieši kontrakt (#2, validate_signal): zlý tvar, iná frekvencia,
   NaN/nekonečno → InvalidInputError, záznam sa NESPRACUJE.

2. KVALITA SIGNÁLU – tento modul. Záznam sa nikdy neodmietne, iba sa označí:
   status "ok" (žiadny problém) alebo "warning" (zoznam problémov v `issues`).
   Model sa spustí aj pri warning, ale používateľ vidí, že výsledku treba dôverovať menej.

Pravidlá na signáli (fungujú pre akýkoľvek vstup, aj mimo PTB-XL; signál v mV pred preprocessingom):
   plochý zvod          rozkmit (max − min) < 0,05 mV       → odpojená elektróda / chýbajúci zvod
   extrémna amplitúda   |x| > 10 mV                         → artefakt (99,99. perc. PTB-XL 9,6 mV)
   saturácia            ≥ 10 po sebe idúcich vzoriek (100 ms) presne na maxime |x| > 1 mV
                                                             → orezaný signál (prebudený zosilňovač)

Pravidlá z anotácií PTB-XL (iba pre záznamy z datasetu):
   static_noise, burst_noise, electrodes_problems, baseline_drift → text s dotknutými zvodmi.

Spustenie (celý dataset):  uv run ekg-quality
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ekg import contract
from ekg.contract import Quality

QUALITY_VERSION = "1.0"
FLAT_PTP_MV = 0.05
EXTREME_MV = 10.0
SATURATION_SAMPLES = 10  # 100 ms; v1 návrh s 5 vzorkami dával falošné poplachy
SATURATION_MIN_MV = 1.0

ANNOTATIONS = {
    "static_noise": "statický šum",
    "burst_noise": "nárazový šum",
    "electrodes_problems": "problém s elektródami",
    "baseline_drift": "drift základnej línie",
}


def _longest_run_at_extreme(lead: np.ndarray) -> int:
    """Najdlhší úsek po sebe idúcich vzoriek, ktoré sú presne na maxime |x| zvodu."""
    peak = np.abs(lead).max()
    at_peak = np.abs(lead) == peak
    best = run = 0
    for flag in at_peak:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def signal_issues(signal: np.ndarray) -> list[str]:
    """Problémy zistené priamo zo signálu (12, 1000) v mV."""
    x = contract.validate_signal(signal)
    issues = []
    ptp = x.max(axis=1) - x.min(axis=1)
    flat = [lead for lead, v in zip(contract.LEADS, ptp, strict=True) if v < FLAT_PTP_MV]
    if flat:
        issues.append(f"plochý zvod: {', '.join(flat)}")
    absmax = np.abs(x).max(axis=1)
    extreme = [
        f"{lead} ({v:.1f} mV)"
        for lead, v in zip(contract.LEADS, absmax, strict=True)
        if v > EXTREME_MV
    ]
    if extreme:
        issues.append(f"extrémna amplitúda: {', '.join(extreme)}")
    saturated = [
        lead
        for lead, row, v in zip(contract.LEADS, x, absmax, strict=True)
        if v > SATURATION_MIN_MV and _longest_run_at_extreme(row) >= SATURATION_SAMPLES
    ]
    if saturated:
        issues.append(f"saturácia signálu: {', '.join(saturated)}")
    return issues


def _clean_annotation(text) -> str | None:
    """' , I-V1,  ' → 'I-V1'; 'alles' (nem. všetko) → 'všetky zvody'; prázdne → None."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    cleaned = str(text).strip().strip(",").strip().upper()
    if not cleaned:
        return None
    return cleaned.replace("ALLES", "všetky zvody").replace("???", "?")


def annotation_issues(row: pd.Series | dict) -> list[str]:
    """Problémy z anotácií kvality PTB-XL (riadok ptbxl_database.csv)."""
    issues = []
    for column, name in ANNOTATIONS.items():
        detail = _clean_annotation(row.get(column))
        if detail is not None:
            issues.append(f"{name} ({detail})")
    return issues


def assess(signal: np.ndarray, annotations: pd.Series | dict | None = None) -> Quality:
    """Výsledok kontroly kvality pre výstup podľa kontraktu (pole `quality`)."""
    issues = signal_issues(signal)
    if annotations is not None:
        issues += annotation_issues(annotations)
    return Quality(status="warning" if issues else "ok", issues=issues)


def assess_dataset(signals: np.ndarray, ids: np.ndarray, db: pd.DataFrame) -> pd.DataFrame:
    """Kontrola kvality pre všetky záznamy; jeden riadok na ecg_id s príznakmi a dôvodmi."""
    rows = []
    for x, ecg_id in zip(signals, ids, strict=True):
        sig = signal_issues(np.asarray(x))
        ann = annotation_issues(db.loc[int(ecg_id)])
        issues = sig + ann
        rows.append(
            {
                "ecg_id": int(ecg_id),
                "status": "warning" if issues else "ok",
                "flat_lead": any(i.startswith("plochý") for i in sig),
                "extreme_amplitude": any(i.startswith("extrémna") for i in sig),
                "saturation": any(i.startswith("saturácia") for i in sig),
                **{col: any(i.startswith(name) for i in ann) for col, name in ANNOTATIONS.items()},
                "issues": "; ".join(issues),
            }
        )
    return pd.DataFrame(rows).set_index("ecg_id")


def report(quality: pd.DataFrame, split: pd.Series) -> str:
    flags = ["flat_lead", "extreme_amplitude", "saturation", *ANNOTATIONS]
    q = quality.join(split.rename("split"))
    counts = q.groupby("split")[flags].sum().T.reindex(columns=["train", "val", "test"])
    counts["spolu"] = counts.sum(axis=1)
    status = pd.crosstab(q["status"], q["split"]).reindex(columns=["train", "val", "test"])
    status["spolu"] = status.sum(axis=1)
    signal_flag = q[["flat_lead", "extreme_amplitude", "saturation"]].any(axis=1)
    annotated = q[list(ANNOTATIONS)].any(axis=1)
    agree = int((signal_flag & annotated).sum())
    return "\n".join(
        [
            f"Kontrola kvality v{QUALITY_VERSION}: {len(q)} záznamov\n",
            "Stav:",
            status.to_string(),
            "\nProblémy podľa typu (počet záznamov):",
            counts.to_string(),
            f"\nZáznamy označené pravidlami na signáli: {int(signal_flag.sum())}, "
            f"z toho s anotáciou kvality v PTB-XL: {agree}",
            "Vyradené záznamy: 0 (kvalita záznam iba označí, neodmieta ho)",
        ]
    )


def main() -> int:
    from ekg.data import ptbxl, splits
    from ekg.data.cache import SignalCache
    from ekg.paths import PROJECT_ROOT

    cache, db = SignalCache(), ptbxl.load_database()
    manifest = splits.load_manifest()
    quality = assess_dataset(np.asarray(cache.signals), cache.ids, db)
    out = PROJECT_ROOT / "manifests" / f"quality_v{QUALITY_VERSION}.csv"
    quality.to_csv(out, lineterminator="\n")
    print(report(quality, manifest["split"]))
    print(f"\nUložené: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
