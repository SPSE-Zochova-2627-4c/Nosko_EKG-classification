"""Rozhodovacie prahy Modelu B pre každý label, nastavené na VALIDATION (issue #17).

Kritérium bolo určené vopred (pred pohľadom na výsledky): pre každý label sa zvolí prah
s **maximálnym F1** na validation. Label je pozitívny, ak pravdepodobnosť >= prah.

Pri nedostatku dát (menej ako MIN_POS pozitívnych alebo MIN_NEG negatívnych príkladov
na validation) sa F1 neoptimalizuje a použije sa náhradný prah FALLBACK_THRESHOLD = 0,5.
Model je trénovaný s pos_weight = neg / pos (#16), takže 0,5 zodpovedá rozhodnutiu,
akoby boli pozitívne a negatívne príklady rovnako časté.

Stabilita prahu sa overí bootstrapom (opakovaný výber validation záznamov s opakovaním).
Test množina sa nepoužíva (je zamknutá pre #22).

Spustenie:  uv run ekg-calibrate-clf [--run clf_v1] [--save]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ekg import contract, paths

THRESHOLDS_VERSION = "1.0"
LABELS = list(contract.LABELS)
CRITERION = "f1"
CRITERION_DESCRIPTION = "maximum F1 na validation, zvlášť pre každý label"
MIN_POS = 30
MIN_NEG = 30
FALLBACK_THRESHOLD = 0.5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0


# --- výpočet ------------------------------------------------------------------


def label_stats(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Matica zámen a odvodené metriky jedného labelu (pozitívny, ak p >= prah)."""
    pred = p >= threshold
    y = y.astype(bool)
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    tn, fn = int(np.sum(~pred & ~y)), int(np.sum(~pred & y))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "senzitivita": sens,
        "špecificita": spec,
        "precision": prec,
        "F1": f1,
    }


def f1_curve(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F1 pre každý rozlišiteľný prah.

    Vráti (prahy zostupne, F1, prahy v strede medzi susednými hodnotami p). Prah v strede
    dáva rovnaké rozhodnutia na validation, ale je odolnejší voči drobným rozdielom
    v pravdepodobnostiach (iný hardvér, iná presnosť výpočtu).
    """
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]  # posledný index každej hodnoty
    tp, fp, th = tp[last], fp[last], ps[last]
    f1 = 2 * tp / (tp + fp + y.sum())
    nxt = np.r_[ps[last[:-1] + 1], th[-1]]  # najbližšia nižšia hodnota
    mid = np.where(last < len(ps) - 1, (th + nxt) / 2, th)
    return th, f1, mid


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Prah s maximálnym F1; pri zhode vyšší prah (menej falošných poplachov)."""
    _, f1, mid = f1_curve(y, p)
    return float(mid[int(np.argmax(f1))])


def bootstrap(
    y: np.ndarray, p: np.ndarray, threshold: float, n: int = N_BOOTSTRAP, seed: int = 0
) -> dict:
    """95 % intervaly: prah zvolený na bootstrap vzorke a F1 pri pevnom prahu."""
    rng = np.random.default_rng(seed)
    ths, f1s = np.empty(n), np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        ths[i] = best_f1_threshold(yb, pb) if 0 < yb.sum() < len(yb) else np.nan
        f1s[i] = label_stats(yb, pb, threshold)["F1"]
    lo_th, hi_th = np.nanquantile(ths, [0.025, 0.975])
    lo_f1, hi_f1 = np.quantile(f1s, [0.025, 0.975])
    return {
        "n": n,
        "threshold_ci95": [float(lo_th), float(hi_th)],
        "f1_ci95": [float(lo_f1), float(hi_f1)],
    }


def calibrate(
    targets: np.ndarray,
    probs: np.ndarray,
    *,
    min_pos: int = MIN_POS,
    min_neg: int = MIN_NEG,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict]:
    """Prah pre každý label v poradí contract.LABELS."""
    if targets.shape != probs.shape or targets.shape[1] != len(LABELS):
        raise ValueError(f"Očakáva sa tvar (N, {len(LABELS)}), je {targets.shape}, {probs.shape}.")
    out = {}
    for j, name in enumerate(LABELS):
        y, p = targets[:, j].astype(np.int64), probs[:, j]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        enough = n_pos >= min_pos and n_neg >= min_neg
        th = best_f1_threshold(y, p) if enough else FALLBACK_THRESHOLD
        entry = {
            "threshold": th,
            "method": CRITERION if enough else "fallback",
            "n_pos": n_pos,
            "n_neg": n_neg,
            "validation_metrics": label_stats(y, p, th),
        }
        if enough and n_bootstrap:
            entry["bootstrap"] = bootstrap(y, p, th, n_bootstrap, seed + j)
        else:
            entry["fallback_reason"] = (
                f"málo príkladov na validation ({n_pos} pozitívnych, {n_neg} negatívnych; "
                f"minimum {min_pos} / {min_neg}) → prah {FALLBACK_THRESHOLD}"
            )
        out[name] = entry
    return out


def cross_validated_f1(
    targets: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    *,
    min_pos: int = MIN_POS,
    min_neg: int = MIN_NEG,
) -> dict[str, dict]:
    """Odhad F1 na nevidených dátach: prah sa zvolí na 4/5 validation, F1 sa meria na 1/5.

    Rozdelenie je po pacientoch (groups), aby záznamy jedného pacienta neboli naraz
    vo výbere prahu aj v hodnotení. Rozhodnutia zo všetkých 5 častí sa spoja a F1 sa
    spočíta raz. Rozdiel oproti F1 na celej validation ukazuje, o koľko je výber prahu
    na tých istých dátach optimistický.
    """
    from sklearn.model_selection import GroupKFold

    decisions = np.zeros_like(targets, dtype=bool)
    for fit, held in GroupKFold(n_splits=n_splits).split(probs, groups=groups):
        fold = calibrate(targets[fit], probs[fit], min_pos=min_pos, min_neg=min_neg, n_bootstrap=0)
        th = np.array([fold[name]["threshold"] for name in LABELS])
        decisions[held] = probs[held] >= th
    out = {}
    for j, name in enumerate(LABELS):
        y, d = targets[:, j].astype(bool), decisions[:, j]
        tp, fp, fn = np.sum(d & y), np.sum(d & ~y), np.sum(~d & y)
        out[name] = {"f1_cv": float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0}
    return out


def comparison_table(targets: np.ndarray, probs: np.ndarray, thresholds: dict) -> pd.DataFrame:
    """Pevný prah 0,5 vs. zvolený prah pre každý label."""
    rows = []
    for j, name in enumerate(LABELS):
        for method, th in (("0,5", 0.5), ("zvolený", thresholds[name]["threshold"])):
            stats = label_stats(targets[:, j], probs[:, j], th)
            rows.append({"label": name, "prah": method, "hodnota prahu": th, **stats})
    return pd.DataFrame(rows).set_index(["label", "prah"])


def macro_f1(targets: np.ndarray, probs: np.ndarray, thresholds: list[float]) -> float:
    return float(
        np.mean(
            [label_stats(targets[:, j], probs[:, j], th)["F1"] for j, th in enumerate(thresholds)]
        )
    )


# --- uloženie a načítanie -----------------------------------------------------


def apply(probs: np.ndarray, data: dict) -> np.ndarray:
    """Rozhodnutia (N, 5) podľa uložených prahov; poradie stĺpcov = contract.LABELS."""
    th = np.array([data["labels"][name]["threshold"] for name in LABELS])
    return probs >= th


def load_thresholds(path: Path) -> dict:
    """Načíta prahy a overí verziu a poradie labelov podľa kontraktu."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != THRESHOLDS_VERSION:
        raise ValueError(f"Nepodporovaná verzia prahov {data.get('version')!r}.")
    if data.get("label_order") != LABELS or list(data["labels"]) != LABELS:
        raise ValueError(f"Prahy majú iné poradie labelov ako kontrakt {LABELS}.")
    return data


def threshold_document(run: str, ckpt_path: Path, ckpt: dict, labels: dict, extra: dict) -> dict:
    return {
        "version": THRESHOLDS_VERSION,
        "run": run,
        "checkpoint_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
        "checkpoint_epoch": ckpt["extra"].get("epoch"),
        "model_versions": ckpt["versions"],
        "label_order": LABELS,
        "rule": "p >= prah → label pozitívny",
        "criterion": CRITERION,
        "criterion_description": CRITERION_DESCRIPTION,
        "min_pos": MIN_POS,
        "min_neg": MIN_NEG,
        "fallback_threshold": FALLBACK_THRESHOLD,
        "calibrated_on": {"split": "val", "subset": "classifier", **extra},
        "labels": labels,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --- dáta a graf --------------------------------------------------------------


def load_val_predictions(
    run: str, amp: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pravdepodobnosti, ciele a pacienti na validation (podmnožina classifier)."""
    from ekg.data.dataset import make_loader
    from ekg.models.classifier import ResNet1dClassifier
    from ekg.training.common import pick_device
    from ekg.training.train_classifier import predict

    model, ckpt = ResNet1dClassifier.load(paths.CHECKPOINT_DIR / run / "best.pt")
    device = pick_device()
    model.to(device)
    loader = make_loader("val", "classifier", batch_size=256)  # validation sa nemieša
    probs, targets = predict(model, loader, device, amp)
    return probs, targets, loader.dataset.patient_ids, ckpt


def plot_f1_curves(targets: np.ndarray, probs: np.ndarray, thresholds: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 3.8), sharey=True)
    for j, (ax, name) in enumerate(zip(axes, LABELS, strict=True)):
        y, p = targets[:, j], probs[:, j]
        th, f1, _ = f1_curve(y, p)
        t = thresholds[name]
        ax.plot(th, f1, lw=1.2)
        ax.axvline(0.5, color="gray", ls=":", lw=1, label="0,5")
        ax.axvline(t["threshold"], color="C3", lw=1, label=f"zvolený {t['threshold']:.2f}")
        if "bootstrap" in t:
            lo, hi = t["bootstrap"]["threshold_ci95"]
            ax.axvspan(lo, hi, color="C3", alpha=0.12, label="95 % bootstrap")
        ax.set(title=f"{name} ({t['n_pos']} poz.)", xlabel="prah", xlim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower center")
    axes[0].set_ylabel("F1 (validation)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prahy klasifikátora pre každý label (val).")
    parser.add_argument("--run", default="clf_v1")
    parser.add_argument("--save", action="store_true", help="uloží prahy k checkpointu")
    args = parser.parse_args(argv)

    probs, targets, groups, ckpt = load_val_predictions(args.run, amp=False)
    probs_bf16, _, _, _ = load_val_predictions(args.run, amp=True)
    thresholds = calibrate(targets, probs)
    cv = cross_validated_f1(targets, probs, groups)
    for name in LABELS:
        thresholds[name]["f1_cv5"] = cv[name]["f1_cv"]
    table = comparison_table(targets, probs, thresholds)
    chosen = [thresholds[n]["threshold"] for n in LABELS]
    changed = int(np.sum((probs_bf16 >= np.array(chosen)) != (probs >= np.array(chosen))))
    extra = {
        "n_records": len(targets),
        "precision": "float32",
        "max_abs_diff_bf16": float(np.abs(probs_bf16 - probs).max()),
        "decisions_changed_bf16": changed,
        "macro_f1_threshold_0_5": macro_f1(targets, probs, [0.5] * len(LABELS)),
        "macro_f1_chosen": macro_f1(targets, probs, chosen),
        "macro_f1_cv5": float(np.mean([cv[n]["f1_cv"] for n in LABELS])),
        "n_patients": int(len(np.unique(groups))),
    }

    out = paths.OUTPUT_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "prahy_porovnanie.csv")
    plot_f1_curves(targets, probs, thresholds, out / "prahy_f1.png")

    print(f"Validation: {len(targets)} záznamov, kritérium: {CRITERION_DESCRIPTION}\n")
    for name, t in thresholds.items():
        ci = t.get("bootstrap", {}).get("threshold_ci95")
        ci_text = f"  95 % CI prahu [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        print(
            f"{name:5s} poz. {t['n_pos']:4d} neg. {t['n_neg']:4d}  prah {t['threshold']:.4f} "
            f"({t['method']}){ci_text}  F1 {t['validation_metrics']['F1']:.3f}, "
            f"krížová validácia {t['f1_cv5']:.3f}"
        )
    show = table[["hodnota prahu", "senzitivita", "špecificita", "precision", "F1", "FP", "FN"]]
    print("\n" + show.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nmacro F1: prah 0,5 → {extra['macro_f1_threshold_0_5']:.3f}, "
        f"zvolené prahy → {extra['macro_f1_chosen']:.3f} "
        f"(krížová validácia po pacientoch: {extra['macro_f1_cv5']:.3f})"
    )
    print(
        f"bf16 vs. float32: max. rozdiel pravdepodobnosti {extra['max_abs_diff_bf16']:.4f}, "
        f"zmenené rozhodnutia {changed} z {probs.size}"
    )
    print(f"Graf: {out / 'prahy_f1.png'}")

    if args.save:
        ckpt_path = paths.CHECKPOINT_DIR / args.run / "best.pt"
        data = threshold_document(args.run, ckpt_path, ckpt, thresholds, extra)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        (paths.CHECKPOINT_DIR / args.run / "thresholds.json").write_text(text, encoding="utf-8")
        docs = paths.PROJECT_ROOT / "docs" / "vysledky" / args.run
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "prahy.json").write_text(text, encoding="utf-8")
        print(
            f"\nUložené: checkpoints/{args.run}/thresholds.json, "
            f"docs/vysledky/{args.run}/prahy.json"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
