"""Vyhodnotenie klasifikátora (Model B) a analýza chýb na VALIDATION (issue #18).

Spustenie:  uv run ekg-clf-report [--run clf_v1] [--docs]

Používa uložený checkpoint a prahy z #17 – overí, že prahy patria k checkpointu (SHA256).
Pravdepodobnosti sa počítajú vo float32 (rovnako ako pri nastavení prahov).

Výstupy v outputs/<run>/hodnotenie/ (s --docs aj v docs/vysledky/<run>/hodnotenie/):
    report.json                metriky po labeloch, macro/micro súhrny, nedefinované metriky
    metriky_po_labeloch.csv    support, matica zámen, precision, recall, špecificita, F1, AUROC…
    matice_zamen.png           matica zámen každého labelu
    falosne_poplachy.csv       aké skutočné labely majú falošne pozitívne záznamy
    prehliadnute.csv           s akými inými labelmi sa vyskytujú prehliadnuté záznamy
    podla_poctu_labelov.csv    chyby podľa počtu skutočných labelov záznamu
    podla_skupin.csv           chyby podľa pohlavia, veku a šumu v anotácii
    podtriedy.csv              recall podľa diagnostickej podtriedy PTB-XL
    podla_istoty_anotacie.csv  recall podľa likelihood SCP kódu (istota kardiológa)
    najvacsie_chyby.csv        najistejšie chyby modelu pre každý label
Test množina sa nepoužíva.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ekg import contract, paths

LABELS = list(contract.LABELS)
NO_LABEL = "bez labelu (NORM)"
AGE_BINS = (0, 40, 60, 80, 1000)
AGE_NAMES = ("< 40", "40–59", "60–79", "≥ 80")
PTBXL_AGE_OVER_89 = 300  # PTB-XL anonymizuje vek > 89 ako 300


# --- metriky (čisté funkcie, testované) --------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def confusion_counts(y: np.ndarray, decision: np.ndarray) -> dict[str, int]:
    y, d = np.asarray(y).astype(bool), np.asarray(decision).astype(bool)
    return {
        "TP": int(np.sum(d & y)),
        "FP": int(np.sum(d & ~y)),
        "TN": int(np.sum(~d & ~y)),
        "FN": int(np.sum(~d & y)),
    }


def metrics_from_counts(c: dict[str, int]) -> tuple[dict, dict]:
    """Metriky z matice zámen. Nedefinovaná metrika = None + dôvod (menovateľ 0)."""
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    values = {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "F1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
    reasons = {
        "precision": "model neoznačil žiadny záznam (TP + FP = 0)",
        "recall": "žiadny pozitívny záznam (TP + FN = 0)",
        "specificity": "žiadny negatívny záznam (TN + FP = 0)",
        "F1": "žiadny pozitívny záznam ani pozitívna predikcia (2TP + FP + FN = 0)",
    }
    undefined = {k: reasons[k] for k, v in values.items() if v is None}
    return values, undefined


def label_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Všetky metriky jedného labelu; pozitívny, ak p >= prah."""
    y = np.asarray(y).astype(int)
    counts = confusion_counts(y, p >= threshold)
    values, undefined = metrics_from_counts(counts)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos and n_neg:
        values["AUROC"] = float(roc_auc_score(y, p))
    else:
        values["AUROC"] = None
        undefined["AUROC"] = "iba jedna trieda (chýbajú pozitívne alebo negatívne záznamy)"
    if n_pos:
        values["AUPRC"] = float(average_precision_score(y, p))
    else:
        values["AUPRC"] = None
        undefined["AUPRC"] = "žiadny pozitívny záznam"
    return {
        "threshold": float(threshold),
        "support": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / len(y),
        **counts,
        **values,
        "undefined": undefined,
    }


def summaries(targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    """Macro, micro a vážené súhrny + metriky po záznamoch (multi-label)."""
    per_label = [
        label_metrics(targets[:, j], probs[:, j], thresholds[j]) for j in range(targets.shape[1])
    ]
    macro = {}
    for key in ("precision", "recall", "specificity", "F1", "AUROC", "AUPRC"):
        vals = [m[key] for m in per_label if m[key] is not None]
        macro[key] = float(np.mean(vals)) if vals else None
        macro[f"{key}_n_labels"] = len(vals)  # z koľkých labelov je priemer
    decisions = probs >= thresholds
    pooled = confusion_counts(targets.ravel(), decisions.ravel())
    micro, _ = metrics_from_counts(pooled)
    y, p = targets.ravel().astype(int), probs.ravel()
    micro["AUROC"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None
    micro["AUPRC"] = float(average_precision_score(y, p)) if y.sum() else None
    support = np.array([m["support"] for m in per_label])
    f1 = np.array([m["F1"] if m["F1"] is not None else 0.0 for m in per_label])
    errors = decisions != targets.astype(bool)
    return {
        "macro": macro,
        "micro": {**pooled, **micro},
        "weighted_F1": float(np.sum(f1 * support) / support.sum()) if support.sum() else None,
        "per_record": {
            "exact_match": float(np.mean(~errors.any(axis=1))),
            "hamming_loss": float(errors.mean()),
            "records_with_error": int(errors.any(axis=1).sum()),
        },
    }


# --- analýza chýb ------------------------------------------------------------


def outcome_frame(targets: np.ndarray, decisions: np.ndarray, index) -> pd.DataFrame:
    """Pre každý záznam a label 'TP'/'FP'/'TN'/'FN'."""
    t, d = targets.astype(bool), decisions.astype(bool)
    out = np.where(t & d, "TP", np.where(~t & d, "FP", np.where(t & ~d, "FN", "TN")))
    return pd.DataFrame(out, index=index, columns=LABELS)


def false_positive_composition(targets: np.ndarray, decisions: np.ndarray) -> pd.DataFrame:
    """Pre FP každého labelu: koľko záznamov nemá žiadny label a aký podiel má iné labely."""
    t = pd.DataFrame(targets.astype(bool), columns=LABELS)
    rows = {}
    for j, name in enumerate(LABELS):
        fp = decisions[:, j].astype(bool) & ~t[name].to_numpy()
        sub = t[fp]
        row = {"FP": int(fp.sum()), NO_LABEL: int((~sub.any(axis=1)).sum())}
        for other in LABELS:
            if other != name:
                row[f"má {other}"] = int(sub[other].sum())
        rows[name] = row
    columns = ["FP", NO_LABEL] + [f"má {other}" for other in LABELS]
    frame = pd.DataFrame(rows).T
    return frame.reindex(columns=[c for c in columns if c in frame]).astype(
        "Int64"
    )  # vlastný label = prázdne


def false_negative_composition(targets: np.ndarray, decisions: np.ndarray) -> pd.DataFrame:
    """Pre FN každého labelu: koľko má aj iný label a koľko model označil iným labelom."""
    t, d = targets.astype(bool), decisions.astype(bool)
    rows = {}
    for j, name in enumerate(LABELS):
        fn = t[:, j] & ~d[:, j]
        others = np.delete(np.arange(len(LABELS)), j)
        row = {
            "FN": int(fn.sum()),
            "iba tento label": int((fn & ~t[:, others].any(axis=1)).sum()),
            "model nedal žiadny label": int((fn & ~d.any(axis=1)).sum()),
        }
        for k in others:
            row[f"predikovaný {LABELS[k]}"] = int((fn & d[:, k]).sum())
        rows[name] = row
    columns = ["FN", "iba tento label", "model nedal žiadny label"]
    columns += [f"predikovaný {other}" for other in LABELS]
    frame = pd.DataFrame(rows).T
    return frame.reindex(columns=[c for c in columns if c in frame]).astype(
        "Int64"
    )  # vlastný label = prázdne


def _group_stats(targets: np.ndarray, decisions: np.ndarray) -> dict:
    c = confusion_counts(targets.ravel(), decisions.ravel())
    values, _ = metrics_from_counts(c)
    errors = decisions != targets.astype(bool)
    return {
        "záznamov": len(targets),
        "micro F1": values["F1"],
        "micro recall": values["recall"],
        "micro špecificita": values["specificity"],
        "bez chyby": float(np.mean(~errors.any(axis=1))) if len(targets) else None,
    }


def by_number_of_labels(targets: np.ndarray, decisions: np.ndarray) -> pd.DataFrame:
    n = targets.sum(axis=1).astype(int)
    rows = {}
    for k in np.unique(n):
        mask = n == k
        stats = _group_stats(targets[mask], decisions[mask])
        stats["priemerne predikovaných labelov"] = float(decisions[mask].sum(axis=1).mean())
        rows[int(k)] = stats
    frame = pd.DataFrame(rows).T
    frame.index.name = "počet skutočných labelov"
    return frame


def age_group(age: pd.Series) -> pd.Series:
    """Vekové skupiny; PTB-XL vek 300 (= viac ako 89) patrí do ≥ 80."""
    groups = pd.cut(age, bins=list(AGE_BINS), labels=list(AGE_NAMES), right=False)
    return groups.astype(object).where(age.notna(), "neznámy")


def by_groups(targets: np.ndarray, decisions: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Chyby podľa pohlavia, veku a šumu v anotácii. meta má stĺpce sex, age, noise."""
    rows = []
    sex = meta["sex"].map({0: "muž", 1: "žena"}).fillna("neznáme")
    for name, series, order in (
        ("pohlavie", sex, ("muž", "žena", "neznáme")),
        ("vek", age_group(meta["age"]), (*AGE_NAMES, "neznámy")),
        ("šum v anotácii", meta["noise"].map({True: "áno", False: "nie"}), ("nie", "áno")),
    ):
        for value in [v for v in order if v in set(series)]:
            mask = (series == value).to_numpy()
            rows.append(
                {"skupina": name, "hodnota": value, **_group_stats(targets[mask], decisions[mask])}
            )
    return pd.DataFrame(rows)


def subclass_recall(
    targets: np.ndarray, decisions: np.ndarray, scp_codes: pd.Series, scp: pd.DataFrame
) -> pd.DataFrame:
    """Recall pre každú diagnostickú podtriedu PTB-XL (záznam môže mať viac podtried)."""
    diag = scp[scp["diagnostic"] == 1]
    rows = []
    for j, name in enumerate(LABELS):
        if name == "AFIB":  # AFIB je jeden rytmický kód, podtriedy nemá
            continue
        sub_of = diag.loc[diag["diagnostic_class"] == name, "diagnostic_subclass"].to_dict()
        found: dict[str, list[int]] = {}
        for i, codes in enumerate(scp_codes):
            for sub in {sub_of[c] for c in codes if c in sub_of}:
                found.setdefault(sub, []).append(i)
        for sub, idx in sorted(found.items()):
            idx = np.array(idx)
            hits = decisions[idx, j].astype(bool)
            rows.append(
                {
                    "label": name,
                    "podtrieda": sub,
                    "záznamov": len(idx),
                    "zachytených": int(hits.sum()),
                    "recall": float(hits.mean()),
                }
            )
    return pd.DataFrame(rows)


def recall_by_likelihood(
    targets: np.ndarray, decisions: np.ndarray, scp_codes: pd.Series, scp: pd.DataFrame
) -> pd.DataFrame:
    """Recall podľa istoty anotácie: najvyššia likelihood SCP kódu, ktorý dal label (0–100).

    Mapovanie labelov (#4) berie kódy bez ohľadu na likelihood, preto label môže pochádzať
    aj z neistého nálezu (napr. 15 = „infarkt nie je vylúčený“).
    """
    from ekg.data.labels import label_map

    mapping = label_map(scp)
    rows = []
    for j, name in enumerate(LABELS):
        pos = np.flatnonzero(targets[:, j] == 1)
        lik = np.array(
            [
                max((v for c, v in scp_codes.iloc[i].items() if mapping.get(c) == name), default=-1)
                for i in pos
            ]
        )
        for value in np.unique(lik):
            hits = decisions[pos[lik == value], j].astype(bool)
            rows.append(
                {
                    "label": name,
                    "likelihood": int(value),
                    "záznamov": len(hits),
                    "zachytených": int(hits.sum()),
                    "recall": float(hits.mean()),
                }
            )
    return pd.DataFrame(rows)


def largest_errors(
    targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, ecg_ids, k: int = 10
) -> pd.DataFrame:
    """Najistejšie chyby: FP s najvyššou a FN s najnižšou pravdepodobnosťou."""
    rows = []
    for j, name in enumerate(LABELS):
        y, p = targets[:, j].astype(bool), probs[:, j]
        for kind, mask, order in (
            ("FP", ~y & (p >= thresholds[j]), -1),
            ("FN", y & (p < thresholds[j]), 1),
        ):
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(order * p[idx], kind="stable")][:k]
            for i in idx:
                true = [LABELS[m] for m in np.flatnonzero(targets[i])] or ["NORM"]
                pred = [LABELS[m] for m in np.flatnonzero(probs[i] >= thresholds)]
                rows.append(
                    {
                        "label": name,
                        "chyba": kind,
                        "ecg_id": int(ecg_ids[i]),
                        "pravdepodobnosť": float(p[i]),
                        "prah": float(thresholds[j]),
                        "skutočné labely": " ".join(true),
                        "predikované labely": " ".join(pred) or "-",
                    }
                )
    return pd.DataFrame(rows)


# --- graf ---------------------------------------------------------------------


def plot_confusion(per_label: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABELS), figsize=(17, 3.6))
    for ax, name in zip(axes, LABELS, strict=True):
        m = per_label[name]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}\n{share[r, c]:.0%}", ha="center", va="center",
                    color="white" if share[r, c] > 0.5 else "black", fontsize=10)  # fmt: skip
        ax.set_xticks([0, 1], ["nie", "áno"])
        ax.set_yticks([0, 1], ["nie", "áno"])
        ax.set(title=f"{name} (prah {m['threshold']:.2f})", xlabel="predikcia")
    axes[0].set_ylabel("skutočnosť")
    fig.suptitle("Matice zámen na validation (percentá z riadku)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- dáta a CLI ---------------------------------------------------------------


def load_val(run: str):
    """Pravdepodobnosti (float32), ciele a ecg_id na validation + checkpoint."""
    from ekg.data.dataset import make_loader
    from ekg.models.classifier import ResNet1dClassifier
    from ekg.training.common import pick_device
    from ekg.training.train_classifier import predict

    model, ckpt = ResNet1dClassifier.load(paths.CHECKPOINT_DIR / run / "best.pt")
    device = pick_device()
    model.to(device)
    loader = make_loader("val", "classifier", batch_size=256)
    probs, targets = predict(model, loader, device, amp=False)
    return probs, targets, loader.dataset.ecg_ids, ckpt


def load_metadata(ecg_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ekg.data import ptbxl
    from ekg.data.labels import NOISE_COLUMNS, _filled

    db = ptbxl.load_database().loc[ecg_ids]
    meta = pd.DataFrame(
        {
            "sex": db["sex"],
            "age": db["age"].where(db["age"] != PTBXL_AGE_OVER_89, 90),
            "noise": _filled(db, NOISE_COLUMNS),
            "scp_codes": db["scp_codes"],
        },
        index=db.index,
    )
    return meta, ptbxl.load_scp_statements()


def _clean(value):
    """NaN → None, numpy → Python (pre JSON)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    from ekg import label_thresholds as lt

    parser = argparse.ArgumentParser(description="Vyhodnotenie klasifikátora na validation.")
    parser.add_argument("--run", default="clf_v1")
    parser.add_argument("--docs", action="store_true", help="skopíruje výstupy do docs/vysledky")
    args = parser.parse_args(argv)

    ckpt_path = paths.CHECKPOINT_DIR / args.run / "best.pt"
    th_data = lt.load_thresholds(paths.CHECKPOINT_DIR / args.run / "thresholds.json")
    digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if th_data["checkpoint_sha256"] != digest:
        raise ValueError(
            "Prahy nepatria k tomuto checkpointu (iný SHA256). Spusti ekg-calibrate-clf."
        )
    thresholds = np.array([th_data["labels"][n]["threshold"] for n in LABELS])

    probs, targets, ecg_ids, ckpt = load_val(args.run)
    decisions = probs >= thresholds
    meta, scp = load_metadata(ecg_ids)

    per_label = {
        name: label_metrics(targets[:, j], probs[:, j], thresholds[j])
        for j, name in enumerate(LABELS)
    }
    summ = summaries(targets, probs, thresholds)
    report = _clean(
        {
            "run": args.run,
            "checkpoint_sha256": digest,
            "checkpoint_epoch": ckpt["extra"].get("epoch"),
            "thresholds_version": th_data["version"],
            "split": "val",
            "subset": "classifier",
            "n_records": len(targets),
            "n_records_without_label": int((targets.sum(axis=1) == 0).sum()),
            "precision_of_computation": "float32",
            "label_order": LABELS,
            "labels": per_label,
            **summ,
        }
    )

    out = paths.OUTPUT_DIR / args.run / "hodnotenie"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame(per_label).T.drop(columns="undefined")
    table.index.name = "label"
    table.to_csv(out / "metriky_po_labeloch.csv", float_format="%.4f")
    plot_confusion(per_label, out / "matice_zamen.png")
    false_positive_composition(targets, decisions).to_csv(out / "falosne_poplachy.csv")
    false_negative_composition(targets, decisions).to_csv(out / "prehliadnute.csv")
    by_number_of_labels(targets, decisions).to_csv(
        out / "podla_poctu_labelov.csv", float_format="%.4f"
    )
    groups = by_groups(targets, decisions, meta)
    groups.to_csv(out / "podla_skupin.csv", index=False, float_format="%.4f")
    subclass_recall(targets, decisions, meta["scp_codes"], scp).to_csv(
        out / "podtriedy.csv", index=False, float_format="%.4f"
    )
    recall_by_likelihood(targets, decisions, meta["scp_codes"], scp).to_csv(
        out / "podla_istoty_anotacie.csv", index=False, float_format="%.4f"
    )
    largest_errors(targets, probs, thresholds, ecg_ids).to_csv(
        out / "najvacsie_chyby.csv", index=False, float_format="%.4f"
    )

    print(f"Validation: {len(targets)} záznamov ({report['n_records_without_label']} bez labelu)\n")
    cols = ["threshold", "support", "n_neg", "TP", "FP", "TN", "FN", "precision", "recall",
            "specificity", "F1", "AUROC", "AUPRC"]  # fmt: skip
    print(table[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    for name, m in per_label.items():
        for metric, reason in m["undefined"].items():
            print(f"  {name}: {metric} nedefinované – {reason}")
    mac, mic = summ["macro"], summ["micro"]
    print(
        f"\nmacro: precision {mac['precision']:.3f}, recall {mac['recall']:.3f}, "
        f"F1 {mac['F1']:.3f}, AUROC {mac['AUROC']:.3f}, AUPRC {mac['AUPRC']:.3f}"
    )
    print(
        f"micro: precision {mic['precision']:.3f}, recall {mic['recall']:.3f}, "
        f"F1 {mic['F1']:.3f}, AUROC {mic['AUROC']:.3f}, AUPRC {mic['AUPRC']:.3f}"
    )
    pr = summ["per_record"]
    print(
        f"záznamy: bez chyby {pr['exact_match']:.3f}, Hamming loss {pr['hamming_loss']:.3f}, "
        f"s aspoň jednou chybou {pr['records_with_error']}"
    )
    print(f"\nVýstupy: {out}")

    if args.docs:
        docs = paths.PROJECT_ROOT / "docs" / "vysledky" / args.run / "hodnotenie"
        docs.mkdir(parents=True, exist_ok=True)
        for f in sorted(out.iterdir()):
            if f.suffix in (".json", ".csv"):
                shutil.copy(f, docs / f.name)
        images = paths.PROJECT_ROOT / "docs" / "obrazky"
        images.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "matice_zamen.png", images / f"{args.run}_matice_zamen.png")
        print(f"Skopírované do {docs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
