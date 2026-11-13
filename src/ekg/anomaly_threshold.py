"""Kalibrácia prahu detekcie anomálií Modelu A na VALIDATION (issue #13).

Referenčný cieľ (podľa anotácií overených človekom – foldy 9 a 10 sú celé validované):
    0 = prísny NORM (#4),  1 = aspoň jeden z labelov MI, STTC, CD, HYP, AFIB.
Ostatné záznamy (napr. NORM so šumom, záznam bez diagnostického kódu) sa pri kalibrácii
vynechajú, lebo pre ne nevieme jednoznačne povedať, či je „správna odpoveď“ anomália.

Záznam je anomália, ak rekonštrukčná chyba >= prah.

Spustenie:  uv run ekg-calibrate-ae [--run ae_v1] [--rule p95]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ekg import contract, paths

THRESHOLD_VERSION = "1.0"
LABELS = list(contract.LABELS)

RULES = {
    "mean": "priemer chyby NORM (pôvodný návrh zo slajdu 11 – na porovnanie)",
    "mean+2std": "priemer + 2 × smerodajná odchýlka chyby NORM",
    "mean+3std": "priemer + 3 × smerodajná odchýlka chyby NORM",
    "p95": "95. percentil chyby NORM (≈ 5 % NORM označených ako anomália)",
    "p99": "99. percentil chyby NORM (≈ 1 % NORM označených ako anomália)",
    "youden": "maximum Youdenovho J = senzitivita + špecificita − 1",
    "f1": "maximum F1 pre triedu anomália",
}


def reference_target(frame: pd.DataFrame) -> pd.Series:
    """1 = patológia, 0 = prísny NORM, NaN = vynechať."""
    abnormal = frame[LABELS].sum(axis=1) > 0
    target = pd.Series(np.nan, index=frame.index)
    target[frame["strict_norm"].astype(bool)] = 0
    target[abnormal] = 1
    return target


def confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp and prec + sens) else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "senzitivita": sens,
        "špecificita": spec,
        "precision": prec,
        "F1": f1,
        "falošné poplachy NORM": 1 - spec,
    }


def candidate_thresholds(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Prahy podľa všetkých pravidiel; NORM štatistiky iba zo záznamov y == 0."""
    norm = scores[y == 0]
    if len(norm) == 0 or np.sum(y == 1) == 0:
        raise ValueError("Na kalibráciu treba NORM aj patologické záznamy.")
    mean, std = norm.mean(), norm.std(ddof=1)
    out = {
        "mean": mean,
        "mean+2std": mean + 2 * std,
        "mean+3std": mean + 3 * std,
        "p95": np.quantile(norm, 0.95),
        "p99": np.quantile(norm, 0.99),
    }
    fpr, tpr, roc_th = roc_curve(y, scores)
    finite = np.isfinite(roc_th)
    out["youden"] = roc_th[finite][np.argmax((tpr - fpr)[finite])]
    best_f1, best_th = -1.0, None
    for th in np.unique(scores):
        f1 = confusion(y, scores, th)["F1"]
        if f1 > best_f1:
            best_f1, best_th = f1, th
    out["f1"] = best_th
    return {k: float(v) for k, v in out.items()}


def comparison_table(
    y: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]
) -> pd.DataFrame:
    rows = {name: {"prah": th, **confusion(y, scores, th)} for name, th in thresholds.items()}
    return pd.DataFrame(rows).T


def plot_curves(y, scores, thresholds, table, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fpr, tpr, _ = roc_curve(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot(fpr, tpr, label=f"ROC (AUROC {roc_auc_score(y, scores):.3f})")
    a.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    b.plot(rec, prec, label=f"PR (AUPRC {average_precision_score(y, scores):.3f})")
    b.axhline(y.mean(), color="gray", ls=":", lw=1, label=f"náhoda ({y.mean():.2f})")
    for name in thresholds:
        r = table.loc[name]
        a.scatter(r["falošné poplachy NORM"], r["senzitivita"], zorder=3)
        a.annotate(
            name,
            (r["falošné poplachy NORM"], r["senzitivita"]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
        )
        b.scatter(r["senzitivita"], r["precision"], zorder=3)
        b.annotate(
            name,
            (r["senzitivita"], r["precision"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    a.set(
        xlabel="falošne pozitívne (1 − špecificita)",
        ylabel="senzitivita",
        title="ROC – prísny NORM vs. patológia",
    )
    b.set(xlabel="senzitivita (recall)", ylabel="precision", title="Precision–recall")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is a else "upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_val_errors(run: str, aggregation: str = "mean") -> tuple[pd.DataFrame, dict]:
    """Chyby autoencodera na validation (všetky záznamy) + prísny NORM z manifestu."""
    from ekg.data import splits
    from ekg.data.dataset import make_loader
    from ekg.models.autoencoder import ConvAutoencoder
    from ekg.models.recon_error import compute_errors

    model, ckpt = ConvAutoencoder.load(paths.CHECKPOINT_DIR / run / "best.pt")
    result = compute_errors(model, make_loader("val", "all", batch_size=256))
    frame = result.to_frame(aggregation)
    frame["strict_norm"] = splits.load_manifest().loc[frame.index, "strict_norm"].astype(bool)
    return frame, ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalibrácia prahu anomálie na validation.")
    parser.add_argument("--run", default="ae_v1")
    parser.add_argument("--rule", choices=list(RULES), help="zvolené pravidlo – uloží prah")
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    args = parser.parse_args(argv)

    frame, ckpt = load_val_errors(args.run, args.aggregation)
    target = reference_target(frame)
    used = target.notna()
    y, scores = target[used].to_numpy(int), frame.loc[used, "error"].to_numpy()
    thresholds = candidate_thresholds(y, scores)
    table = comparison_table(y, scores, thresholds)

    out = paths.OUTPUT_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(y, scores, thresholds, table, out / "prah_roc_pr.png")
    table.to_csv(out / "prah_porovnanie.csv")

    print(
        f"Validation: {int((y == 0).sum())} prísny NORM, {int((y == 1).sum())} s patológiou, "
        f"{int((~used).sum())} vynechaných. Agregácia: {args.aggregation}."
    )
    print(f"AUROC {roc_auc_score(y, scores):.3f}, AUPRC {average_precision_score(y, scores):.3f}\n")
    show = table[["prah", "senzitivita", "špecificita", "precision", "F1", "FP", "FN"]].copy()
    print(show.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nGraf: {out / 'prah_roc_pr.png'}")

    if args.rule:
        ckpt_path = paths.CHECKPOINT_DIR / args.run / "best.pt"
        data = {
            "version": THRESHOLD_VERSION,
            "run": args.run,
            "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
            "checkpoint_epoch": ckpt["extra"]["epoch"],
            "aggregation": args.aggregation,
            "rule": args.rule,
            "rule_description": RULES[args.rule],
            "threshold": thresholds[args.rule],
            "calibrated_on": {
                "split": "val",
                "n_strict_norm": int((y == 0).sum()),
                "n_abnormal": int((y == 1).sum()),
                "n_excluded": int((~used).sum()),
            },
            "validation_metrics": {
                k: (int(v) if k in ("TP", "FP", "TN", "FN") else float(v))
                for k, v in table.loc[args.rule].drop("prah").items()
            },
            "auroc": float(roc_auc_score(y, scores)),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        (paths.CHECKPOINT_DIR / args.run / "threshold.json").write_text(text, encoding="utf-8")
        docs = paths.PROJECT_ROOT / "docs" / "vysledky" / args.run
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "prah.json").write_text(text, encoding="utf-8")
        print(
            f"\nUložený prah {data['threshold']:.5f} ({args.rule}) → "
            f"checkpoints/{args.run}/threshold.json"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
