"""Testy vyhodnotenia klasifikátora (issue #18)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from ekg import contract, paths
from ekg.evaluation import clf_report as cr

SAVED = paths.PROJECT_ROOT / "docs" / "vysledky" / "clf_v1" / "hodnotenie" / "report.json"
THRESHOLDS = paths.PROJECT_ROOT / "docs" / "vysledky" / "clf_v1" / "prahy.json"


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    targets = (rng.random((n, 5)) < np.array([0.25, 0.25, 0.2, 0.12, 0.07])).astype(np.float32)
    probs = 1 / (1 + np.exp(-(rng.normal(0, 1, (n, 5)) + 2.0 * targets - 0.5)))
    return targets, probs, np.array([0.5, 0.6, 0.5, 0.4, 0.7])


def test_label_metrics_match_sklearn():
    targets, probs, th = _synthetic()
    for j in range(5):
        y, d = targets[:, j], probs[:, j] >= th[j]
        m = cr.label_metrics(y, probs[:, j], th[j])
        assert m["precision"] == pytest.approx(precision_score(y, d))
        assert m["recall"] == pytest.approx(recall_score(y, d))
        assert m["F1"] == pytest.approx(f1_score(y, d))
        assert m["specificity"] == pytest.approx(m["TN"] / (m["TN"] + m["FP"]))
        assert m["support"] == m["TP"] + m["FN"] == int(y.sum())
        assert m["n_neg"] == m["TN"] + m["FP"]
        assert m["undefined"] == {}


def test_threshold_is_inclusive():
    m = cr.label_metrics(np.array([1, 0]), np.array([0.5, 0.49]), 0.5)
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (1, 0, 1, 0)


def test_undefined_metrics_have_reason():
    # žiadny pozitívny záznam a model nič neoznačí
    m = cr.label_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    for key in ("precision", "recall", "F1", "AUROC", "AUPRC"):
        assert m[key] is None and m["undefined"][key]
    assert m["specificity"] == 1.0
    # pozitívne existujú, ale model nič neoznačí → precision nedefinovaná, F1 = 0
    m = cr.label_metrics(np.array([1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.4]), 0.9)
    assert m["precision"] is None and "TP + FP = 0" in m["undefined"]["precision"]
    assert m["recall"] == 0.0 and m["F1"] == 0.0
    # iba pozitívne záznamy → špecificita a AUROC nedefinované, AUPRC áno
    m = cr.label_metrics(np.ones(3), np.array([0.2, 0.6, 0.9]), 0.5)
    assert m["specificity"] is None and m["AUROC"] is None and m["AUPRC"] is not None


def test_summaries_macro_micro():
    targets, probs, th = _synthetic()
    s = cr.summaries(targets, probs, th)
    d = probs >= th
    assert s["macro"]["F1"] == pytest.approx(f1_score(targets, d, average="macro"))
    assert s["micro"]["F1"] == pytest.approx(f1_score(targets, d, average="micro"))
    assert s["micro"]["precision"] == pytest.approx(precision_score(targets, d, average="micro"))
    assert s["weighted_F1"] == pytest.approx(f1_score(targets, d, average="weighted"))
    assert s["per_record"]["exact_match"] == pytest.approx(np.mean((d == targets).all(axis=1)))
    assert s["per_record"]["hamming_loss"] == pytest.approx(np.mean(d != targets))
    assert s["macro"]["F1_n_labels"] == 5


def test_macro_skips_undefined_and_counts_labels():
    targets, probs, th = _synthetic()
    targets[:, 4] = 0  # AFIB bez pozitívnych
    s = cr.summaries(targets, probs, th)
    assert s["macro"]["AUROC_n_labels"] == 4 and s["macro"]["recall_n_labels"] == 4
    assert s["macro"]["specificity_n_labels"] == 5


def test_error_compositions():
    labels = contract.LABELS
    targets = np.array(
        [
            [0, 0, 0, 0, 0],  # NORM, model dá MI → FP MI bez labelu
            [0, 1, 0, 0, 0],  # STTC, model dá MI → FP MI pri STTC, FN STTC
            [1, 0, 0, 0, 0],  # MI, model nič → FN MI bez predikcie
            [1, 0, 1, 0, 0],  # MI+CD, model dá CD → FN MI s predikovaným CD
        ]
    )
    decisions = np.array(
        [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=bool
    )
    fp = cr.false_positive_composition(targets, decisions)
    assert fp.loc["MI", "FP"] == 2 and fp.loc["MI", cr.NO_LABEL] == 1
    assert fp.loc["MI", "má STTC"] == 1 and pd.isna(fp.loc["MI", "má MI"])
    fn = cr.false_negative_composition(targets, decisions)
    assert fn.loc["MI", "FN"] == 2 and fn.loc["MI", "iba tento label"] == 1
    assert fn.loc["MI", "model nedal žiadny label"] == 1
    assert fn.loc["MI", "predikovaný CD"] == 1
    assert fn.loc["STTC", "predikovaný MI"] == 1
    assert list(fp.index) == list(labels)


def test_by_number_of_labels_and_groups():
    targets, probs, th = _synthetic(n=300)
    d = probs >= th
    by_n = cr.by_number_of_labels(targets, d)
    assert by_n["záznamov"].sum() == 300
    meta = pd.DataFrame(
        {
            "sex": np.r_[np.zeros(150), np.ones(150)],
            "age": np.r_[np.full(100, 30.0), np.full(100, 70.0), np.full(99, 90.0), np.nan],
            "noise": np.arange(300) % 3 == 0,
        }
    )
    groups = cr.by_groups(targets, d, meta)
    for name in ("pohlavie", "vek", "šum v anotácii"):
        assert groups.loc[groups["skupina"] == name, "záznamov"].sum() == 300
    ages = groups[groups["skupina"] == "vek"].set_index("hodnota")["záznamov"]
    assert ages.to_dict() == {"< 40": 100, "60–79": 100, "≥ 80": 99, "neznámy": 1}


def test_subclass_recall():
    scp = pd.DataFrame(
        {
            "diagnostic": [1, 1, 1],
            "diagnostic_class": ["MI", "MI", "HYP"],
            "diagnostic_subclass": ["IMI", "AMI", "LVH"],
        },
        index=["IMI", "ASMI", "LVH"],
    )
    targets = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0]])
    decisions = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=bool)
    codes = pd.Series([{"IMI": 100}, {"IMI": 50, "ASMI": 50, "LVH": 100}, {"ASMI": 80}])
    table = cr.subclass_recall(targets, decisions, codes, scp).set_index(["label", "podtrieda"])
    assert table.loc[("MI", "IMI"), "záznamov"] == 2
    assert table.loc[("MI", "IMI"), "recall"] == pytest.approx(0.5)
    assert table.loc[("MI", "AMI"), "zachytených"] == 1
    assert table.loc[("HYP", "LVH"), "recall"] == 1.0


def test_recall_by_likelihood():
    scp = pd.DataFrame(
        {
            "diagnostic": [1, 1, 0],
            "diagnostic_class": ["MI", "MI", None],
            "diagnostic_subclass": ["IMI", "AMI", None],
        },
        index=["IMI", "ASMI", "SR"],
    )
    targets = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    decisions = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0] * 5], dtype=bool)
    codes = pd.Series([{"IMI": 15}, {"IMI": 15, "ASMI": 100}, {"ASMI": 100, "SR": 0}, {"SR": 0}])
    table = cr.recall_by_likelihood(targets, decisions, codes, scp).set_index(
        ["label", "likelihood"]
    )
    assert table.loc[("MI", 15), "záznamov"] == 1 and table.loc[("MI", 15), "recall"] == 0.0
    assert table.loc[("MI", 100), "záznamov"] == 2  # rozhoduje najvyššia likelihood
    assert table.loc[("MI", 100), "recall"] == 1.0


def test_largest_errors_sorted():
    targets, probs, th = _synthetic()
    errors = cr.largest_errors(targets, probs, th, np.arange(len(targets)), k=5)
    for (label, kind), group in errors.groupby(["label", "chyba"]):
        p = group["pravdepodobnosť"].to_numpy()
        assert len(group) <= 5
        assert np.all(np.diff(p) <= 0) if kind == "FP" else np.all(np.diff(p) >= 0)
        j = contract.LABELS.index(label)
        assert (p >= th[j]).all() if kind == "FP" else (p < th[j]).all()


def test_saved_report_complete_and_consistent_with_thresholds():
    report = json.loads(SAVED.read_text(encoding="utf-8"))
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    assert report["split"] == "val" and report["label_order"] == list(contract.LABELS)
    assert report["checkpoint_sha256"] == thresholds["checkpoint_sha256"]
    for name in contract.LABELS:
        m, t = report["labels"][name], thresholds["labels"][name]
        assert m["threshold"] == t["threshold"]
        assert m["support"] + m["n_neg"] == report["n_records"]
        for key in ("TP", "FP", "TN", "FN"):
            assert m[key] == t["validation_metrics"][key]  # rovnaké pravdepodobnosti ako v #17
        for key in ("precision", "recall", "specificity", "F1", "AUROC", "AUPRC"):
            assert m[key] is not None or key in m["undefined"]


def _fake_run(tmp_path, monkeypatch, model_bytes=b"model"):
    from ekg import label_thresholds as lt

    targets, probs, th = _synthetic(n=400)
    run = tmp_path / "ckpt" / "run"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(model_bytes)
    data = {
        "version": lt.THRESHOLDS_VERSION,
        "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
        "label_order": list(contract.LABELS),
        "labels": {n: {"threshold": float(t)} for n, t in zip(contract.LABELS, th, strict=True)},
    }
    (run / "thresholds.json").write_text(json.dumps(data), encoding="utf-8")
    meta = pd.DataFrame(
        {
            "sex": np.arange(400) % 2,
            "age": np.full(400, 50.0),
            "noise": np.zeros(400, bool),
            "scp_codes": [{} for _ in range(400)],
        }
    )
    scp = pd.DataFrame(columns=["diagnostic", "diagnostic_class", "diagnostic_subclass"])
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cr, "load_val", lambda run: (probs, targets, np.arange(400), {"extra": {"epoch": 3}})
    )
    monkeypatch.setattr(cr, "load_metadata", lambda ids: (meta, scp))


def test_cli_writes_all_outputs(tmp_path, monkeypatch):
    _fake_run(tmp_path, monkeypatch)
    assert cr.main(["--run", "run", "--docs"]) == 0
    out = tmp_path / "out" / "run" / "hodnotenie"
    for name in ("report.json", "metriky_po_labeloch.csv", "matice_zamen.png",
                 "falosne_poplachy.csv", "prehliadnute.csv", "podla_poctu_labelov.csv",
                 "podla_skupin.csv", "podtriedy.csv", "podla_istoty_anotacie.csv",
                 "najvacsie_chyby.csv"):  # fmt: skip
        assert (out / name).exists(), name
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert list(report["labels"]) == list(contract.LABELS) and report["checkpoint_epoch"] == 3
    assert (tmp_path / "docs" / "vysledky" / "run" / "hodnotenie" / "report.json").exists()
    assert (tmp_path / "docs" / "obrazky" / "run_matice_zamen.png").exists()


def test_cli_rejects_thresholds_of_other_checkpoint(tmp_path, monkeypatch):
    _fake_run(tmp_path, monkeypatch, model_bytes=b"iny model")
    with pytest.raises(ValueError, match="SHA256"):
        cr.main(["--run", "run"])
