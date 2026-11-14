"""Testy záverečného vyhodnotenia na test množine (issue #22)."""

import numpy as np
import pandas as pd
import pytest

from ekg import contract, paths
from ekg.data.splits import SplitLeakError
from ekg.evaluation import clf_report as cr
from ekg.evaluation import final_evaluation as fe

RESULTS = paths.PROJECT_ROOT / "docs" / "vysledky" / "test"


# --- zmrazenie ----------------------------------------------------------------------


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Malý projekt: checkpointy, prahy, manifesty a kód – na overenie zmrazenia."""
    for rel in (
        "checkpoints/ae_v1/best.pt",
        "checkpoints/ae_v1/threshold.json",
        "checkpoints/clf_v1/best.pt",
        "checkpoints/clf_v1/thresholds.json",
        "manifests/preprocessing_v1.0.json",
        "manifests/splits_v1.0.csv",
        "docs/mapovanie_labelov_v1.0.csv",
        "uv.lock",
        "src/ekg/a.py",
        "src/ekg/sub/b.py",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel, encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "outputs")
    return tmp_path


def test_freeze_covers_models_data_and_code(fake_project):
    freeze = fe.make_freeze()
    assert set(freeze["sha256"]) >= {
        "checkpoints/ae_v1/best.pt",
        "checkpoints/clf_v1/thresholds.json",
        "manifests/splits_v1.0.csv",
        "docs/mapovanie_labelov_v1.0.csv",
        "src/ekg/a.py",
        "src/ekg/sub/b.py",
    }
    assert freeze["versions"]["schema"] == contract.SCHEMA_VERSION
    assert fe.check_freeze(freeze) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda root: (root / "checkpoints/clf_v1/thresholds.json").write_text("x"), "zmenený"),
        (lambda root: (root / "src/ekg/a.py").write_text("x = 1"), "zmenený src/ekg/a.py"),
        (lambda root: (root / "src/ekg/novy.py").write_text(""), "nový súbor"),
        (lambda root: (root / "manifests/splits_v1.0.csv").unlink(), "chýba"),
    ],
)
def test_any_change_after_freeze_is_detected(fake_project, change, expected):
    freeze = fe.make_freeze()
    change(fake_project)
    problems = fe.check_freeze(freeze)
    assert problems and any(expected in p for p in problems)


def test_run_refuses_without_freeze_or_after_change(fake_project, capsys):
    assert fe.run() == 1
    assert "Chýba zmrazenie" in capsys.readouterr().out
    assert fe.main(["--freeze"]) == 0
    assert fe.main(["--freeze"]) == 1  # zmrazenie sa neprepíše
    (fake_project / "checkpoints/ae_v1/threshold.json").write_text("iny prah")
    assert fe.run() == 1
    assert "zmenený checkpoints/ae_v1/threshold.json" in capsys.readouterr().out


def test_run_does_not_overwrite_results(fake_project, capsys):
    assert fe.main(["--freeze"]) == 0
    (fe.results_dir() / "report.json").write_text("{}")
    assert fe.run() == 1
    assert "neprepisujú" in capsys.readouterr().out


# --- pacienti a bootstrap -------------------------------------------------------------


def test_patient_overlap():
    manifest = pd.DataFrame(
        {"patient_id": [1, 2, 3, 4], "split": ["train", "val", "test", "test"]},
        index=pd.Index([10, 11, 12, 13], name="ecg_id"),
    )
    assert fe.patient_overlap(manifest) == {"train": 0, "val": 0}
    leaked = manifest.copy()
    leaked.loc[13, "patient_id"] = 1  # pacient 1 v train aj test
    with pytest.raises(SplitLeakError):
        fe.patient_overlap(leaked)


def test_cluster_bootstrap_resamples_whole_patients():
    groups = np.repeat(np.arange(50), 2)
    values = np.repeat(np.linspace(0, 1, 50), 2)
    seen = []

    def stat(idx):
        # oba záznamy pacienta sa vyberú vždy spolu → počet výskytov každej hodnoty je párny
        counts = np.bincount(groups[idx], minlength=50)
        seen.append(counts)
        return values[idx].mean()

    lo, hi = fe.cluster_bootstrap(groups, stat, n=200, seed=0)
    assert lo < values.mean() < hi
    assert all((c % 2 == 0).all() for c in seen)
    assert fe.cluster_bootstrap(groups, stat, n=50, seed=1) == fe.cluster_bootstrap(
        groups, stat, n=50, seed=1
    )


# --- metriky --------------------------------------------------------------------------


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    targets = (rng.random((n, 5)) < 0.25).astype(float)
    probs = 1 / (1 + np.exp(-(rng.normal(0, 1, (n, 5)) + 2.5 * targets - 1)))
    return targets, probs, np.full(5, 0.5), np.arange(n) // 2


def test_classifier_metrics_equal_validation_report():
    targets, probs, th, groups = _synthetic()
    out = fe.evaluate_classifier(targets, probs, th, groups)
    for j, name in enumerate(contract.LABELS):
        assert out["labels"][name] == cr.label_metrics(targets[:, j], probs[:, j], th[j])
    assert out["macro"] == cr.summaries(targets, probs, th)["macro"]
    lo, hi = out["bootstrap_ci95"]["macro_AUROC"]
    assert lo <= out["macro"]["AUROC"] <= hi


def test_autoencoder_metrics():
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "error": np.r_[rng.normal(0.02, 0.005, 100), rng.normal(0.05, 0.02, 100)],
            "strict_norm": np.r_[np.ones(100, bool), np.zeros(100, bool)],
            **{label: np.zeros(n) for label in contract.LABELS},
        }
    )
    frame.loc[100:, "MI"] = 1.0
    out = fe.evaluate_autoencoder(frame, 0.03, np.arange(n))
    m = out["metrics"]
    assert m["n_strict_norm"] == 100 and m["n_abnormal"] == 100
    assert m["TP"] + m["FN"] == 100 and m["TN"] + m["FP"] == 100
    lo, hi = m["bootstrap_ci95"]["AUROC"]
    assert lo <= m["AUROC"] <= hi


def test_consistency_counts():
    def res(anom, pos):
        labels = {
            n: contract.LabelResult(0.9 if n == pos else 0.1, 0.5, n == pos)
            for n in contract.LABELS
        }
        return contract.PredictionResult(
            record_id="x",
            model_versions={},
            anomaly=contract.AnomalyResult(
                0.1, anom, 0.1, 0.05, dict.fromkeys(contract.LEADS, 0.1)
            ),
            labels=labels,
            consistency=contract.Consistency(True),
        )  # fmt: skip

    results = [res(True, "MI"), res(False, None), res(False, "CD"), res(True, None)]
    targets = np.array([[1, 0, 0, 0, 0], [0] * 5, [0, 0, 1, 0, 0], [0] * 5])
    c = fe.consistency_counts(results, targets)
    assert c["súlad"] == {"záznamov": 2, "so skutočnou poruchou": 1, "NORM": 1}
    assert c["A v norme, B porucha"]["so skutočnou poruchou"] == 1
    assert c["A anomália, B nič"]["NORM"] == 1


def _test_set_openers() -> dict[str, str]:
    """Moduly v src/ekg, ktoré v kóde (nie v texte) otvárajú test množinu: allow_test=<nie iba
    odovzdaný parameter>. Vráti {modul: zdrojový text}."""
    import ast

    src = paths.PROJECT_ROOT / "src" / "ekg"
    found = {}
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    passthrough = isinstance(kw.value, ast.Name) and kw.value.id == "allow_test"
                    if kw.arg == "allow_test" and not passthrough:
                        found[path.relative_to(src).as_posix()] = text
    return found


def test_test_set_is_evaluated_only_in_final_evaluation():
    """Test množinu otvára iba kontrola dát (#9, bez modelov) a záverečné vyhodnotenie."""
    openers = _test_set_openers()
    assert sorted(openers) == ["data/check_pipeline.py", "evaluation/final_evaluation.py"]
    check = openers["data/check_pipeline.py"]
    assert "ekg.models" not in check and "ekg.inference" not in check  # žiadny model
