"""Diagnostika natrénovaného autoencodera na VALIDATION (issue #11).

Spustenie:  uv run ekg-ae-report [--run ae_v1]

Odpovedá na otázku: rozlišuje autoencoder normálne a patologické EKG?
- rekonštrukčná chyba pre skupiny: prísny NORM, ostatné bez labelu, každý z 5 labelov,
- AUROC: prísny NORM (0) vs. záznam s aspoň jedným labelom (1) – priemer aj max cez zvody,
- histogram chýb (outputs/<run>/chyby_val.png) a tabuľka (outputs/<run>/chyby_val.csv).

Nie je to kalibrácia prahu (#13) ani finálne vyhodnotenie (#14) – iba kontrola, že tréning
dáva zmysel. Test množina sa nepoužíva.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ekg import contract, paths
from ekg.data import splits
from ekg.data.dataset import make_loader
from ekg.models.autoencoder import ConvAutoencoder
from ekg.models.recon_error import compute_errors

LABELS = list(contract.LABELS)


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = {"prísny NORM": frame["strict_norm"]}
    no_label = (frame[LABELS].sum(axis=1) == 0) & ~frame["strict_norm"]
    groups["bez labelu, nie prísny NORM"] = no_label
    groups.update({label: frame[label] == 1 for label in LABELS})
    groups["aspoň 1 label"] = frame[LABELS].sum(axis=1) > 0
    rows = {
        name: {
            "záznamy": int(mask.sum()),
            "medián": frame.loc[mask, "error"].median(),
            "priemer": frame.loc[mask, "error"].mean(),
            "90. percentil": frame.loc[mask, "error"].quantile(0.9),
        }
        for name, mask in groups.items()
    }
    return pd.DataFrame(rows).T


def auroc(frame: pd.DataFrame, column: str) -> float:
    """prísny NORM = 0, aspoň jeden label = 1; ostatné záznamy sa vynechajú."""
    abnormal = frame[LABELS].sum(axis=1) > 0
    mask = frame["strict_norm"] | abnormal
    return roc_auc_score(abnormal[mask].astype(int), frame.loc[mask, column])


def plot(frame: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abnormal = frame[LABELS].sum(axis=1) > 0
    bins = np.logspace(np.log10(frame["error"].min()), np.log10(frame["error"].max()), 60)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(
        frame.loc[frame["strict_norm"], "error"],
        bins=bins,
        alpha=0.6,
        density=True,
        label="prísny NORM",
    )
    ax.hist(
        frame.loc[abnormal, "error"],
        bins=bins,
        alpha=0.6,
        density=True,
        label="aspoň 1 label (MI/STTC/CD/HYP/AFIB)",
    )
    ax.set(
        xscale="log",
        xlabel="rekonštrukčná chyba (MSE, log mierka)",
        ylabel="hustota",
        title="Rozdelenie rekonštrukčnej chyby na validation",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ae_v1")
    args = parser.parse_args(argv)

    model, ckpt = ConvAutoencoder.load(paths.CHECKPOINT_DIR / args.run / "best.pt")
    result = compute_errors(model, make_loader("val", "all", batch_size=256))
    frame = result.to_frame("mean")
    frame["error_max"] = result.record("max")
    frame["strict_norm"] = splits.load_manifest().loc[frame.index, "strict_norm"].astype(bool)

    out = paths.OUTPUT_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "chyby_val.csv")
    plot(frame, out / "chyby_val.png")

    print(
        f"Model: {args.run}, epocha {ckpt['extra']['epoch']}, "
        f"val MSE (NORM) {ckpt['extra']['val_loss']:.5f}\n"
    )
    print("Rekonštrukčná chyba na validation podľa skupín:")
    print(group_table(frame).round(4).to_string())
    print(
        f"\nAUROC prísny NORM vs. aspoň 1 label:  priemer cez zvody {auroc(frame, 'error'):.3f}"
        f"   |   max zvod {auroc(frame, 'error_max'):.3f}"
    )
    print(f"(0.5 = náhoda, 1.0 = dokonalé oddelenie)\n\nGraf: {out / 'chyby_val.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
