"""Zobrazenie EKG, rekonštrukcie a výsledkov oboch modelov (issue #20).

Jeden obrázok obsahuje:
    - 12 pomenovaných zvodov, časová os v sekundách: signál po filtrovaní (mV) a rekonštrukciu
      autoencodera (mV), pri každom zvode jeho rekonštrukčnú chybu,
    - anomaly score s prahom a príznakom anomálie (Model A),
    - pravdepodobnosti 5 porúch s prahmi a rozhodnutím (Model B),
    - súlad modelov, kvalitu signálu a upozornenie, že nejde o lekársku diagnózu.
Hodnoty sa berú priamo z PredictionResult pipeline (#19) – obrázok nič neprepočítava.

Spustenie:  uv run ekg-show --ecg-id 10 [-o prehlad.png]
            uv run ekg-show cesta/k/zaznamu [-o prehlad.png]
Web aplikácia s rovnakým obrázkom: uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ekg import contract
from ekg.contract import PredictionResult

# Usporiadanie zvodov ako na bežnom 12-zvodovom zázname: končatinové vľavo, hrudné vpravo
LAYOUT = (("I", "II", "III", "AVR", "AVL", "AVF"), ("V1", "V2", "V3", "V4", "V5", "V6"))
LEAD_NAMES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}  # klinický zápis
COLOR_SIGNAL = "#111111"
COLOR_RECON = "#e8710a"
COLOR_POSITIVE = "#c5221f"
COLOR_NEGATIVE = "#5f6b7a"
COLOR_THRESHOLD = "#202124"


def lead_label(lead: str) -> str:
    return LEAD_NAMES.get(lead, lead)


def time_axis() -> np.ndarray:
    """Čas v sekundách pre 1000 vzoriek pri 100 Hz (0,00 … 9,99 s)."""
    return np.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE


def verdict(result: PredictionResult) -> str:
    """Krátke zhrnutie výsledku (nie diagnóza)."""
    found = [name for name, r in result.labels.items() if r.positive]
    a = "anomália" if result.anomaly.is_anomaly else "bez anomálie"
    b = ", ".join(found) if found else "žiadna z 5 porúch"
    return f"Model A: {a} · Model B: {b}"


def plot_report(
    signal_mv: np.ndarray,
    recon_mv: np.ndarray,
    result: PredictionResult,
    show_reconstruction: bool = True,
):
    """Prehľad jedného záznamu ako matplotlib Figure.

    signal_mv, recon_mv: (12, 1000) v mV v poradí contract.LEADS (Pipeline.reconstruct).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal_mv, recon_mv = np.asarray(signal_mv), np.asarray(recon_mv)
    expected = (contract.N_LEADS, contract.N_SAMPLES)
    if signal_mv.shape != expected or recon_mv.shape != expected:
        raise ValueError(f"Signál aj rekonštrukcia musia mať tvar {expected}.")

    t = time_axis()
    fig = plt.figure(figsize=(16, 12.5))
    grid = fig.add_gridspec(8, 2, height_ratios=[1] * 6 + [0.15, 1.6], hspace=0.55, wspace=0.12)
    errors = result.anomaly.error_per_lead
    # spoločná mierka pre všetky zvody – amplitúdy sa dajú porovnať medzi sebou
    lim = float(np.percentile(np.abs(signal_mv), 99.5)) * 1.25 or 1.0

    for col, leads in enumerate(LAYOUT):
        for row, lead in enumerate(leads):
            ax = fig.add_subplot(grid[row, col])
            i = contract.LEADS.index(lead)
            ax.plot(t, signal_mv[i], color=COLOR_SIGNAL, lw=0.9, label="signál (po filtrovaní)")
            if show_reconstruction:
                ax.plot(t, recon_mv[i], color=COLOR_RECON, lw=0.9, alpha=0.7, label="rekonštrukcia")
            ax.set_xlim(0, contract.DURATION_S)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(range(contract.DURATION_S + 1))
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            ax.set_ylabel("mV", fontsize=7)
            ax.set_title(
                f"{lead_label(lead)}   chyba {errors[lead]:.4f}", fontsize=9, loc="left", pad=2
            )
            if row == len(leads) - 1:
                ax.set_xlabel("čas (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right", ncols=2)

    _plot_anomaly(fig.add_subplot(grid[7, 0]), result)
    _plot_labels(fig.add_subplot(grid[7, 1]), result)

    title = f"Záznam {result.record_id} – {verdict(result)}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    notes = []
    c = result.consistency
    notes.append("Súlad modelov: áno" if c.models_agree else f"Súlad modelov: NIE – {c.note}")
    if result.quality.issues:
        notes.append(f"Kvalita signálu: {'; '.join(result.quality.issues)}")
    notes.append(f"⚠ {result.disclaimer}")
    fig.text(0.01, 0.005, "\n".join(notes), fontsize=9, va="bottom", wrap=True)
    fig.subplots_adjust(top=0.955, bottom=0.10, left=0.05, right=0.98)
    return fig


def _plot_anomaly(ax, result: PredictionResult) -> None:
    a = result.anomaly
    color = COLOR_POSITIVE if a.is_anomaly else COLOR_NEGATIVE
    top = max(a.anomaly_score, a.threshold) * 1.6
    ax.barh([0], [a.anomaly_score], color=color, height=0.5)
    ax.axvline(a.threshold, color=COLOR_THRESHOLD, ls="--", lw=1.2)
    ax.text(a.threshold, 0.42, f" prah {a.threshold:.4f}", fontsize=8, va="bottom")
    ax.text(
        max(a.anomaly_score, a.threshold) * 1.04,
        0,
        f"{a.anomaly_score:.4f} – {'ANOMÁLIA' if a.is_anomaly else 'v norme'}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
    )
    ax.set_xlim(0, top * 1.35)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("anomaly score = rekonštrukčná chyba (priemer cez zvody)", fontsize=8)
    ax.set_title("Model A – autoencoder", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.25)


def _plot_labels(ax, result: PredictionResult) -> None:
    names = list(result.labels)
    y = np.arange(len(names))[::-1]
    for yi, name in zip(y, names, strict=True):
        lab = result.labels[name]
        color = COLOR_POSITIVE if lab.positive else COLOR_NEGATIVE
        ax.barh(yi, lab.probability, color=color, height=0.6)
        ax.plot([lab.threshold] * 2, [yi - 0.38, yi + 0.38], color=COLOR_THRESHOLD, lw=1.6)
        mark = "ÁNO" if lab.positive else "nie"
        ax.text(1.03, yi, f"{lab.probability:.3f}  (prah {lab.threshold:.2f})  {mark}",
                va="center", fontsize=8, color=color)  # fmt: skip
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(0, 1.42)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.axvline(1, color="#cccccc", lw=0.8)
    ax.set_xlabel("pravdepodobnosť (čierna čiarka = prah labelu)", fontsize=8)
    ax.set_title("Model B – klasifikátor (5 nezávislých porúch)", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.25)


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from ekg import inference

    parser = argparse.ArgumentParser(description="Obrázok EKG, rekonštrukcie a výsledkov.")
    parser.add_argument("record", nargs="?", help="WFDB záznam (cesta bez prípony)")
    parser.add_argument("--ecg-id", type=int, help="záznam PTB-XL (nie z test množiny)")
    parser.add_argument("-o", "--output", type=Path, help="cieľový PNG (predvolene outputs/)")
    parser.add_argument("--bez-rekonstrukcie", action="store_true", help="iba signál")
    args = parser.parse_args(argv)
    if (args.record is None) == (args.ecg_id is None):
        parser.error("zadaj práve jeden WFDB záznam alebo --ecg-id")

    try:
        pipe = inference.Pipeline.load()
        if args.ecg_id is None and inference.ptbxl_id(args.record) is not None:
            args.ecg_id = inference.ptbxl_id(args.record)
        if args.ecg_id is not None:
            [(rid, signal, meta, ann)] = list(inference.ptbxl_records([args.ecg_id]))
        else:
            rid, ann = Path(args.record).name, None
            signal = inference.read_wfdb(args.record)
            meta = contract.Metadata(source=rid)
        result = pipe.predict(signal, rid, meta, ann)
        filtered, recon = pipe.reconstruct(signal)
    except (inference.PipelineError, contract.InvalidInputError, OSError, KeyError) as exc:
        print(f"Chyba: {exc}", file=sys.stderr)
        return 1

    from ekg import paths

    out = args.output or paths.OUTPUT_DIR / "zobrazenie" / f"zaznam_{rid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_report(filtered, recon, result, show_reconstruction=not args.bez_rekonstrukcie)
    fig.savefig(out, dpi=110)
    print(inference.summary(result))
    print(f"Obrázok: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
