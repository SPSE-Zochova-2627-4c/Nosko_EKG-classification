"""Záverečné vyhodnotenie celého systému na oddelenej TEST množine (issue #22).

Postup (každý krok je samostatný príkaz, aby sa dalo doložiť poradie):

    uv run ekg-final-eval --freeze   1. zmrazenie: SHA256 modelov, prahov, preprocessingu,
                                        manifestov, mapovania labelov a celého kódu src/ekg
                                        → docs/vysledky/test/zmrazenie.json (commit PRED testom)
    uv run ekg-final-eval            2. vyhodnotenie: overí, že sa od zmrazenia nič nezmenilo,
                                        a raz vyhodnotí zmrazenú pipeline (#19) na test množine

Vyhodnotenie výsledky neprepíše (už existujúci report = chyba). Po pohľade na test výsledky sa
nič neladí; prípadné ďalšie pokusy patria do iného adresára a označujú sa ako exploratívne.

Test množina sa načíta iba tu: splits.split_ids("test", allow_test=True).
Výstupy: docs/vysledky/test/ (report, metriky, predikcie CSV, analýza chýb), JSON predikcií
v outputs/test/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ekg import contract, paths

FREEZE_VERSION = "1.0"
AE_RUN, CLF_RUN = "ae_v1", "clf_v1"
LABELS = list(contract.LABELS)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 22


def results_dir() -> Path:
    return paths.PROJECT_ROOT / "docs" / "vysledky" / "test"


# --- zmrazenie ----------------------------------------------------------------------


def frozen_files() -> dict[str, Path]:
    """Súbory, od ktorých výsledok závisí: modely, prahy, dáta, mapovanie a celý kód."""
    root = paths.PROJECT_ROOT
    files = {
        f"checkpoints/{AE_RUN}/best.pt": paths.CHECKPOINT_DIR / AE_RUN / "best.pt",
        f"checkpoints/{AE_RUN}/threshold.json": paths.CHECKPOINT_DIR / AE_RUN / "threshold.json",
        f"checkpoints/{CLF_RUN}/best.pt": paths.CHECKPOINT_DIR / CLF_RUN / "best.pt",
        f"checkpoints/{CLF_RUN}/thresholds.json": paths.CHECKPOINT_DIR
        / CLF_RUN
        / "thresholds.json",
    }
    for rel in (
        "manifests/preprocessing_v1.0.json",
        "manifests/splits_v1.0.csv",
        "docs/mapovanie_labelov_v1.0.csv",
        "uv.lock",
    ):
        files[rel] = root / rel
    for path in sorted((root / "src" / "ekg").rglob("*.py")):
        files[path.relative_to(root).as_posix()] = path
    return files


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, str]:
    import torch

    from ekg.data.labels import MAPPING_VERSION
    from ekg.data.splits import SPLIT_VERSION
    from ekg.label_thresholds import THRESHOLDS_VERSION
    from ekg.preprocessing import PREPROCESSING_VERSION
    from ekg.quality import QUALITY_VERSION

    return {
        "schema": contract.SCHEMA_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "label_mapping": MAPPING_VERSION,
        "splits": SPLIT_VERSION,
        "label_thresholds": THRESHOLDS_VERSION,
        "quality": QUALITY_VERSION,
        "torch": str(torch.__version__),
    }


def make_freeze() -> dict:
    files = frozen_files()
    missing = [rel for rel, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Chýbajú súbory na zmrazenie: {missing}")
    return {
        "freeze_version": FREEZE_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit_at_freeze": _git_commit(),
        "runs": {"autoencoder": AE_RUN, "classifier": CLF_RUN},
        "versions": versions(),
        "sha256": {rel: _sha256(p) for rel, p in files.items()},
    }


def check_freeze(freeze: dict) -> list[str]:
    """Zoznam rozdielov oproti zmrazeniu (prázdny = všetko sedí)."""
    problems = []
    current = frozen_files()
    for rel, digest in freeze["sha256"].items():
        path = current.get(rel)
        if path is None or not path.exists():
            problems.append(f"chýba {rel}")
        elif _sha256(path) != digest:
            problems.append(f"zmenený {rel}")
    for rel in sorted(set(current) - set(freeze["sha256"])):
        problems.append(f"nový súbor {rel}")
    for key, value in freeze["versions"].items():
        if versions().get(key) != value:
            problems.append(f"iná verzia {key}: {versions().get(key)} ≠ {value}")
    return problems


# --- dáta ---------------------------------------------------------------------------


def patient_overlap(manifest: pd.DataFrame) -> dict[str, int]:
    """Počet pacientov test množiny, ktorí sú aj v train alebo val (má byť 0)."""
    from ekg.data.splits import check_no_leak

    check_no_leak(manifest)
    test = set(manifest.loc[manifest["split"] == "test", "patient_id"])
    return {
        split: len(test & set(manifest.loc[manifest["split"] == split, "patient_id"]))
        for split in ("train", "val")
    }


# --- bootstrap po pacientoch --------------------------------------------------------


def cluster_bootstrap(groups: np.ndarray, stat, n: int = N_BOOTSTRAP, seed: int = 0) -> list:
    """95 % interval štatistiky stat(indexy) pri prevzorkovaní PACIENTOV s opakovaním.

    Záznamy jedného pacienta nie sú nezávislé, preto sa prevzorkúvajú celí pacienti.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    values = []
    for _ in range(n):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[k] for k in chosen])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            values.append(v)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return [float(lo), float(hi)]


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else None


def _f1(y, d):
    tp, fp, fn = np.sum(d & y), np.sum(d & ~y), np.sum(~d & y)
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None


# --- vyhodnotenie -------------------------------------------------------------------


def evaluate_autoencoder(frame: pd.DataFrame, threshold: float, groups: np.ndarray) -> dict:
    """Model A: prísny NORM vs. patológia (#13), rovnaká definícia ako na validation."""
    from ekg.evaluation.ae_analysis import add_predictions, summary_metrics

    frame = add_predictions(frame, threshold)
    metrics = summary_metrics(frame, threshold)
    used = frame["target"].notna().to_numpy()
    y = frame["target"].to_numpy()[used].astype(int)
    s = frame["error"].to_numpy()[used]
    g = groups[used]
    d = s >= threshold
    metrics["bootstrap_ci95"] = {
        "AUROC": cluster_bootstrap(g, lambda i: _auroc(y[i], s[i]), seed=BOOTSTRAP_SEED),
        "senzitivita": cluster_bootstrap(
            g, lambda i: d[i][y[i] == 1].mean() if (y[i] == 1).any() else None, seed=1
        ),
        "špecificita": cluster_bootstrap(
            g, lambda i: (~d[i][y[i] == 0]).mean() if (y[i] == 0).any() else None, seed=2
        ),
    }
    return {"metrics": metrics, "frame": frame}


def evaluate_classifier(
    targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, groups: np.ndarray
) -> dict:
    """Model B: metriky po labeloch a súhrny (#18) + bootstrap po pacientoch."""
    from ekg.evaluation.clf_report import label_metrics, summaries

    per_label = {
        n: label_metrics(targets[:, j], probs[:, j], thresholds[j]) for j, n in enumerate(LABELS)
    }
    summ = summaries(targets, probs, thresholds)
    t, dec = targets.astype(bool), probs >= thresholds
    ci = {
        "macro_AUROC": cluster_bootstrap(
            groups,
            lambda i: np.mean([_auroc(t[i, j], probs[i, j]) or np.nan for j in range(5)]),
            seed=BOOTSTRAP_SEED,
        ),
        "macro_F1": cluster_bootstrap(
            groups,
            lambda i: np.mean([_f1(t[i, j], dec[i, j]) or 0.0 for j in range(5)]),
            seed=3,
        ),
    }
    for j, name in enumerate(LABELS):
        ci[f"F1_{name}"] = cluster_bootstrap(
            groups, lambda i, j=j: _f1(t[i, j], dec[i, j]), n=N_BOOTSTRAP, seed=10 + j
        )
        ci[f"AUROC_{name}"] = cluster_bootstrap(
            groups, lambda i, j=j: _auroc(t[i, j], probs[i, j]), n=N_BOOTSTRAP, seed=20 + j
        )
    return {"labels": per_label, **summ, "bootstrap_ci95": ci}


def consistency_counts(results, targets: np.ndarray) -> dict:
    """Ako často modely nesúhlasia (#19) a koľko z toho má skutočnú poruchu."""
    has = targets.sum(axis=1) > 0
    isan = np.array([r.anomaly.is_anomaly for r in results])
    anypos = np.array([any(lab.positive for lab in r.labels.values()) for r in results])
    out = {}
    for name, mask in (
        ("súlad", isan == anypos),
        ("A v norme, B porucha", ~isan & anypos),
        ("A anomália, B nič", isan & ~anypos),
    ):
        out[name] = {
            "záznamov": int(mask.sum()),
            "so skutočnou poruchou": int((mask & has).sum()),
            "NORM": int((mask & ~has).sum()),
        }
    return out


def comparison_with_validation(ae: dict, clf: dict) -> pd.DataFrame:
    """Validation (#14, #18) vs. test – rovnaké metriky, rovnaké prahy."""
    docs = paths.PROJECT_ROOT / "docs" / "vysledky"
    ae_val = json.loads((docs / AE_RUN / "analyza" / "metriky.json").read_text("utf-8"))
    clf_val = json.loads((docs / CLF_RUN / "hodnotenie" / "report.json").read_text("utf-8"))
    rows = []
    for key in ("AUROC", "AUPRC", "senzitivita", "špecificita", "precision", "F1"):
        rows.append(("Model A", key, ae_val[key], ae["metrics"][key]))
    for key in ("AUROC", "AUPRC", "F1", "precision", "recall"):
        rows.append(("Model B macro", key, clf_val["macro"][key], clf["macro"][key]))
    rows.append(("Model B micro", "F1", clf_val["micro"]["F1"], clf["micro"]["F1"]))
    for name in LABELS:
        for key in ("AUROC", "F1", "recall", "precision"):
            rows.append(
                (f"Model B {name}", key, clf_val["labels"][name][key], clf["labels"][name][key])
            )
    frame = pd.DataFrame(rows, columns=["model", "metrika", "validation", "test"])
    frame["rozdiel"] = frame["test"] - frame["validation"]
    return frame


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(force: bool = False) -> int:
    from ekg import export
    from ekg.data import labels, ptbxl, splits
    from ekg.data.cache import SignalCache
    from ekg.evaluation import clf_report as cr
    from ekg.evaluation.ae_analysis import detection_by_label, excluded_groups
    from ekg.inference import Pipeline

    out_docs = results_dir()
    freeze_path = out_docs / "zmrazenie.json"
    if not freeze_path.exists():
        print("Chýba zmrazenie – najprv: uv run ekg-final-eval --freeze (a commit).")
        return 1
    freeze = json.loads(freeze_path.read_text("utf-8"))
    problems = check_freeze(freeze)
    if problems:
        print("Od zmrazenia sa niečo zmenilo – vyhodnotenie sa nespustí:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    if (out_docs / "report.json").exists() and not force:
        print("Test už bol vyhodnotený (docs/vysledky/test/report.json). Výsledky sa neprepisujú.")
        return 1

    manifest = splits.load_manifest()
    overlap = patient_overlap(manifest)
    if any(overlap.values()):
        print(f"Únik pacientov medzi množinami: {overlap}")
        return 1
    test_all = splits.split_ids("test", manifest, allow_test=True)
    test_clf = splits.split_ids("test", manifest, subset="classifier", allow_test=True)
    print(f"Test: {len(test_all)} záznamov, z toho {len(test_clf)} pre Model B; únik pacientov 0")

    # --- zmrazená pipeline na všetkých test záznamoch (CPU, po jednom zázname) ---
    pipe = Pipeline.load(AE_RUN, CLF_RUN)
    db = ptbxl.load_database()
    signals = SignalCache().get(test_all)
    results = pipe.predict_batch(
        signals,
        [str(i) for i in test_all],
        [export.metadata_from_ptbxl(db.loc[i]) for i in test_all],
        [db.loc[i] for i in test_all],
    )
    by_id = {int(r.record_id): r for r in results}

    # --- Model A ---
    frame = pd.DataFrame(
        {
            "error": [by_id[i].anomaly.anomaly_score for i in test_all],
            **{
                f"error_{lead}": [by_id[i].anomaly.error_per_lead[lead] for i in test_all]
                for lead in contract.LEADS
            },
        },
        index=pd.Index(test_all, name="ecg_id"),
    )
    frame[LABELS] = manifest.loc[test_all, LABELS].astype(int).to_numpy()
    frame["strict_norm"] = manifest.loc[test_all, "strict_norm"].astype(bool).to_numpy()
    ae = evaluate_autoencoder(
        frame, pipe.anomaly_threshold, manifest.loc[test_all, "patient_id"].to_numpy()
    )
    reasons = labels.norm_exclusion_reason(db, ptbxl.load_scp_statements())

    # --- Model B ---
    targets = manifest.loc[test_clf, LABELS].to_numpy(float)
    probs = np.array([[by_id[i].labels[n].probability for n in LABELS] for i in test_clf])
    groups = manifest.loc[test_clf, "patient_id"].to_numpy()
    clf = evaluate_classifier(targets, probs, pipe.label_thresholds, groups)
    decisions = probs >= pipe.label_thresholds
    meta, scp = cr.load_metadata(test_clf)
    consistency = consistency_counts([by_id[i] for i in test_clf], targets)

    # --- uloženie ---
    out_docs.mkdir(parents=True, exist_ok=True)
    out_json = paths.OUTPUT_DIR / "test"
    out_json.mkdir(parents=True, exist_ok=True)
    export.to_csv(results, out_docs / "predikcie_test.csv")
    export.to_json(results, out_json / "predikcie_test.json")
    comparison = comparison_with_validation(ae, clf)
    comparison.to_csv(out_docs / "porovnanie_val_test.csv", index=False, float_format="%.4f")
    detection_by_label(ae["frame"]).to_csv(out_docs / "ae_detekcia_podla_labelu.csv")
    excluded_groups(ae["frame"], reasons).to_csv(out_docs / "ae_vynechane_skupiny.csv")
    table = pd.DataFrame(clf["labels"]).T.drop(columns="undefined")
    table.index.name = "label"
    table.to_csv(out_docs / "clf_metriky_po_labeloch.csv", float_format="%.4f")
    cr.false_positive_composition(targets, decisions).to_csv(out_docs / "clf_falosne_poplachy.csv")
    cr.false_negative_composition(targets, decisions).to_csv(out_docs / "clf_prehliadnute.csv")
    cr.by_groups(targets, decisions, meta).to_csv(
        out_docs / "clf_podla_skupin.csv", index=False, float_format="%.4f"
    )
    cr.subclass_recall(targets, decisions, meta["scp_codes"], scp).to_csv(
        out_docs / "clf_podtriedy.csv", index=False, float_format="%.4f"
    )
    cr.recall_by_likelihood(targets, decisions, meta["scp_codes"], scp).to_csv(
        out_docs / "clf_podla_istoty_anotacie.csv", index=False, float_format="%.4f"
    )
    cr.plot_confusion(clf["labels"], out_docs / "clf_matice_zamen.png")

    report = _clean(
        {
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "freeze_sha256": _sha256(freeze_path),
            "freeze_check": "OK – žiadna zmena od zmrazenia",
            "split": "test",
            "n_records": len(test_all),
            "n_records_classifier": len(test_clf),
            "n_patients": int(manifest.loc[test_all, "patient_id"].nunique()),
            "patient_overlap": overlap,
            "model_versions": pipe.model_versions,
            "autoencoder": ae["metrics"],
            "classifier": {k: v for k, v in clf.items()},
            "consistency": consistency,
            "quality_warnings": int(sum(r.quality.status == "warning" for r in results)),
        }
    )
    (out_docs / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    m = ae["metrics"]
    print(
        f"\nModel A (prísny NORM {m['n_strict_norm']} vs. patológia {m['n_abnormal']}): "
        f"AUROC {m['AUROC']:.3f}, senzitivita {m['senzitivita']:.3f}, "
        f"špecificita {m['špecificita']:.3f}, F1 {m['F1']:.3f}"
    )
    print(
        f"Model B: macro AUROC {clf['macro']['AUROC']:.3f}, macro F1 {clf['macro']['F1']:.3f}, "
        f"micro F1 {clf['micro']['F1']:.3f}"
    )
    print("\n" + comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nVýstupy: {out_docs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Záverečné vyhodnotenie na test množine (#22).")
    parser.add_argument("--freeze", action="store_true", help="zmrazí modely, prahy, dáta a kód")
    parser.add_argument("--check", action="store_true", help="iba overí zmrazenie")
    parser.add_argument("--force", action="store_true", help="prepíše existujúce výsledky")
    args = parser.parse_args(argv)

    if args.freeze:
        path = results_dir() / "zmrazenie.json"
        if path.exists() and not args.force:
            print("Zmrazenie už existuje – neprepisujem (--force).")
            return 1
        data = make_freeze()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Zmrazených {len(data['sha256'])} súborov → {path}")
        print("Commitni zmrazenie PRED spustením vyhodnotenia.")
        return 0
    if args.check:
        freeze = json.loads((results_dir() / "zmrazenie.json").read_text("utf-8"))
        problems = check_freeze(freeze)
        print("Zmrazenie sedí." if not problems else "\n".join(problems))
        return 0 if not problems else 1
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
