"""Pomocné funkcie web aplikácie (streamlit_app.py, issue #20) – bez závislosti na Streamlite.

Logika je tu, aby sa dala testovať bežnými testami; aplikácia iba zobrazuje.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ekg import contract
from ekg.contract import InvalidInputError, PredictionResult

# Ukážkové záznamy z VALIDATION (nie z test množiny) – každý prípad pravidla súladu (#19)
EXAMPLES = {
    10: "NORM – oba modely v norme",
    17: "AFIB – anomália, Model B označí AFIB (a falošne MI)",
    127: "STTC – Model A v norme, Model B STTC (nesúlad)",
    352: "NORM so šumom – falošný poplach autoencodera (nesúlad)",
}


def signal_from_upload(files: list[tuple[str, bytes]]) -> tuple[str, np.ndarray]:
    """Nahrané súbory → (identifikátor, signál (12, 1000) v mV).

    Podporované:
      - WFDB záznam: dvojica <meno>.hea + <meno>.dat (100 Hz, 12 zvodov, 10 s),
      - NumPy .npy s poľom (12, 1000) v mV v poradí zvodov kontraktu.
    Neplatný vstup → InvalidInputError s vysvetlením (kontrakt #2).
    """
    if not files:
        raise InvalidInputError("Nenahral si žiadny súbor.")
    by_ext: dict[str, list[tuple[str, bytes]]] = {}
    for name, data in files:
        by_ext.setdefault(Path(name).suffix.lower(), []).append((name, data))
    unknown = set(by_ext) - {".hea", ".dat", ".npy"}
    if unknown:
        raise InvalidInputError(
            f"Nepodporovaný typ súboru {sorted(unknown)} – očakáva sa .hea + .dat alebo .npy."
        )

    if ".npy" in by_ext:
        if len(files) != 1:
            raise InvalidInputError("Nahraj iba jeden súbor .npy.")
        name, data = by_ext[".npy"][0]
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except ValueError as exc:
            raise InvalidInputError(f"Súbor {name} nie je platné NumPy pole: {exc}") from exc
        return Path(name).stem, contract.validate_signal(arr)

    heads, dats = by_ext.get(".hea", []), by_ext.get(".dat", [])
    if len(heads) != 1 or len(dats) != 1:
        raise InvalidInputError("WFDB záznam potrebuje práve jeden súbor .hea a jeden .dat.")
    stem = Path(heads[0][0]).stem
    if Path(dats[0][0]).stem != stem:
        raise InvalidInputError(f"Súbory .hea a .dat musia mať rovnaké meno ({stem}).")

    import wfdb

    with tempfile.TemporaryDirectory() as tmp:
        for name, data in (heads[0], dats[0]):
            (Path(tmp) / Path(name).name).write_bytes(data)
        try:
            record = wfdb.rdrecord(str(Path(tmp) / stem))
        except Exception as exc:  # wfdb hlási chybný súbor rôznymi výnimkami
            raise InvalidInputError(f"WFDB záznam sa nedá načítať: {exc}") from exc
    if record.p_signal is None:
        raise InvalidInputError("WFDB záznam neobsahuje fyzikálny signál (p_signal).")
    return stem, contract.from_wfdb(record.p_signal, record.sig_name, int(record.fs))


def results_table(result: PredictionResult) -> pd.DataFrame:
    """Tabuľka Modelu B: pravdepodobnosť, prah, rozhodnutie pre každý label."""
    return pd.DataFrame(
        [
            {
                "Porucha": name,
                "Pravdepodobnosť": lab.probability,
                "Prah": lab.threshold,
                "Výsledok": "pozitívny" if lab.positive else "negatívny",
            }
            for name, lab in result.labels.items()
        ]
    )


def lead_errors_table(result: PredictionResult) -> pd.DataFrame:
    """Rekonštrukčná chyba po zvodoch (Model A)."""
    from ekg.visualization import lead_label

    return pd.DataFrame(
        {
            "Zvod": [lead_label(lead) for lead in result.anomaly.error_per_lead],
            "Chyba": list(result.anomaly.error_per_lead.values()),
        }
    )
