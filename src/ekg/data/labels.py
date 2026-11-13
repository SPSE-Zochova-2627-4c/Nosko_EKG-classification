"""Mapovanie SCP kódov PTB-XL na labely MI, STTC, CD, HYP, AFIB a výber NORM (issue #4).

Pravidlá (verzia MAPPING_VERSION, zdôvodnenie v docs/mapovanie_labelov.md):

Labely pre Model B
- MI, STTC, CD, HYP: záznam má aspoň jeden diagnostický SCP kód s danou
  `diagnostic_class`, bez ohľadu na likelihood (ako benchmark Strodthoff et al. 2020).
- AFIB: záznam má rytmický kód AFIB, bez ohľadu na likelihood (rytmické kódy
  likelihood takmer nepoužívajú – 1466 z 1514 AFIB má 0). Flutter (AFLT) sa nezaratúva.
- Do Modelu B ide záznam, ktorý má aspoň jeden diagnostický kód (vrátane NORM)
  alebo AFIB. Záznamy bez nich nevieme označiť, preto sa vynechajú.

Prísny NORM pre Model A (autoencoder) – všetky podmienky naraz, v tomto poradí:
  1. NORM s likelihood 100
  2. žiadny iný diagnostický kód
  3. z rytmických kódov iba SR (alebo žiadny)
  4. bez static_noise, burst_noise, electrodes_problems
  5. bez baseline_drift, extra_beats, pacemaker
  6. bez form kódov (PVC, PAC, STD_, INVT, LVOLT, …)
validated_by_human sa nevyžaduje.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ekg import contract

MAPPING_VERSION = "1.0"

AFIB_CODE = "AFIB"
NORM_CODE = "NORM"
SINUS_RHYTHM = "SR"
NOISE_COLUMNS = ("static_noise", "burst_noise", "electrodes_problems")
OTHER_QUALITY_COLUMNS = ("baseline_drift", "extra_beats", "pacemaker")

# Dôvody vylúčenia z prísneho NORM v poradí, v akom sa kontrolujú.
NORM_RULES = (
    "nie je NORM s likelihood 100",
    "iný diagnostický kód",
    "rytmus iný ako SR",
    "šum alebo problém s elektródami",
    "baseline drift, extra beats alebo kardiostimulátor",
    "form kód",
)


def label_map(scp: pd.DataFrame) -> dict[str, str]:
    """SCP kód -> label Modelu B. Kódy mimo mapy label nedávajú."""
    mapping = {
        code: cls
        for code, cls in scp.loc[scp["diagnostic"] == 1, "diagnostic_class"].items()
        if cls in contract.LABELS
    }
    mapping[AFIB_CODE] = "AFIB"
    return mapping


def mapping_table(scp: pd.DataFrame) -> pd.DataFrame:
    """Tabuľka všetkých 71 SCP kódov: typ, trieda a výsledný label (alebo prázdne)."""
    mapping = label_map(scp)
    kind = np.select(
        [scp["diagnostic"] == 1, scp["rhythm"] == 1, scp["form"] == 1],
        ["diagnostic", "rhythm", "form"],
        default="?",
    )
    table = pd.DataFrame(
        {
            "description": scp["description"],
            "kind": kind,
            "diagnostic_class": scp["diagnostic_class"],
            "label": [mapping.get(code, "") for code in scp.index],
        },
        index=scp.index,
    )
    return table.sort_values(["label", "kind"], key=lambda s: s.replace("", "~"))


def make_targets(db: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Ciele Modelu B: stĺpce LABELS (0/1, int8) + `usable` (patrí do Modelu B)."""
    mapping = label_map(scp)
    diagnostic = set(scp.index[scp["diagnostic"] == 1])
    targets = pd.DataFrame(0, index=db.index, columns=list(contract.LABELS), dtype=np.int8)
    usable = pd.Series(False, index=db.index)
    for ecg_id, codes in db["scp_codes"].items():
        for code in codes:
            if code in mapping:
                targets.at[ecg_id, mapping[code]] = 1
        usable[ecg_id] = any(c in diagnostic for c in codes) or AFIB_CODE in codes
    targets["usable"] = usable
    return targets


def _filled(db: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """True, ak je aspoň jeden z textových stĺpcov kvality vyplnený."""
    mask = pd.Series(False, index=db.index)
    for col in columns:
        mask |= db[col].notna() & (db[col].astype(str).str.strip() != "")
    return mask


def norm_exclusion_reason(db: pd.DataFrame, scp: pd.DataFrame) -> pd.Series:
    """Pre každý záznam prvý dôvod, prečo nie je prísny NORM; None = prísny NORM."""
    diagnostic = set(scp.index[scp["diagnostic"] == 1])
    rhythm = set(scp.index[scp["rhythm"] == 1])
    form = set(scp.index[scp["form"] == 1])
    codes = db["scp_codes"]
    failed = [
        codes.apply(lambda d: d.get(NORM_CODE) != 100),
        codes.apply(lambda d: any(c in diagnostic and c != NORM_CODE for c in d)),
        codes.apply(lambda d: any(c in rhythm and c != SINUS_RHYTHM for c in d)),
        _filled(db, NOISE_COLUMNS),
        _filled(db, OTHER_QUALITY_COLUMNS),
        codes.apply(lambda d: any(c in form for c in d)),
    ]
    reason = pd.Series(None, index=db.index, dtype=object)
    for rule, mask in zip(NORM_RULES, failed, strict=True):
        reason[mask & reason.isna()] = rule
    # pandas môže prázdne miesta uložiť ako NaN – navonok vždy None
    return reason.astype(object).where(reason.notna(), None)


def strict_norm(db: pd.DataFrame, scp: pd.DataFrame) -> pd.Series:
    """True pre záznamy vhodné na tréning a kalibráciu autoencodera."""
    return norm_exclusion_reason(db, scp).isna()


def report(db: pd.DataFrame, scp: pd.DataFrame) -> str:
    """Textový prehľad počtov – rovnaké čísla ako v docs/mapovanie_labelov.md."""
    split = np.select([db["strat_fold"] <= 8, db["strat_fold"] == 9], ["train", "val"], "test")
    targets = make_targets(db, scp)
    usable = targets.pop("usable")
    lines = [f"Mapovanie labelov v{MAPPING_VERSION}", ""]
    counts = targets[usable].groupby(split[usable]).sum().T
    counts["spolu"] = counts.sum(axis=1)
    lines += [
        f"Model B: {int(usable.sum())} záznamov, vynechaných bez labelu: {int((~usable).sum())}",
        counts[["train", "val", "test", "spolu"]].to_string(),
        "",
    ]
    reason = norm_exclusion_reason(db, scp).fillna("prísny NORM")
    norm_rows = pd.crosstab(reason.to_numpy(), split, rownames=[""], colnames=[""])
    norm_rows = norm_rows.reindex(
        index=[*NORM_RULES, "prísny NORM"], columns=["train", "val", "test"]
    )
    norm_rows = norm_rows.fillna(0).astype(int)
    lines += ["NORM pre autoencoder (prvý dôvod vylúčenia):", norm_rows.to_string()]
    return "\n".join(lines)


def main() -> int:
    from ekg.data import ptbxl

    print(report(ptbxl.load_database(), ptbxl.load_scp_statements()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
