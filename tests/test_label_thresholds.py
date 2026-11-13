"""Testy prahov klasifikátora pre jednotlivé labely (issue #17)."""

import hashlib
import json

import numpy as np
import pytest

from ekg import contract, paths
from ekg import label_thresholds as lt

SAVED = paths.PROJECT_ROOT / "docs" / "vysledky" / "clf_v1" / "prahy.json"


def _synthetic(n=600, seed=0, prevalence=(0.25, 0.25, 0.2, 0.12, 0.07)):
    """Umelé dáta: pozitívne príklady majú v priemere vyššiu pravdepodobnosť."""
    rng = np.random.default_rng(seed)
    targets = (rng.random((n, 5)) < np.array(prevalence)).astype(np.float32)
    probs = 1 / (1 + np.exp(-(rng.normal(0, 1, (n, 5)) + 2.0 * targets - 0.5)))
    return targets, probs.astype(np.float32)


def test_label_stats_counts_and_inclusive_threshold():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.5, 0.5, 0.1, 0.2])
    s = lt.label_stats(y, p, 0.5)  # p == prah → pozitívny
    assert (s["TP"], s["FP"], s["TN"], s["FN"]) == (2, 1, 1, 1)
    assert s["F1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert s["senzitivita"] == pytest.approx(2 / 3) and s["špecificita"] == pytest.approx(0.5)


def test_f1_curve_matches_brute_force_and_midpoints_keep_decisions():
    targets, probs = _synthetic(n=300)
    y, p = targets[:, 0], np.round(probs[:, 0], 2)  # zaokrúhlenie → veľa zhodných hodnôt
    th, f1, mid = lt.f1_curve(y, p)
    assert np.all(np.diff(th) < 0)  # zostupne, bez duplikátov
    assert len(th) == len(np.unique(p))
    for t, f, m in zip(th, f1, mid, strict=True):
        assert f == pytest.approx(lt.label_stats(y, p, t)["F1"])
        assert np.array_equal(p >= m, p >= t)  # stredný prah nemení rozhodnutia


def test_best_threshold_separable_data():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    th = lt.best_f1_threshold(y, p)
    assert th == pytest.approx(0.5)  # stred medzi 0,3 a 0,7
    assert lt.label_stats(y, p, th)["F1"] == 1.0


def test_tie_prefers_higher_threshold():
    # prah 0,8 aj 0,2 dáva F1 = 2/3 (maximum) → vyberie sa vyšší (menej falošných poplachov)
    y = np.array([1, 0, 0, 1])
    p = np.array([0.8, 0.6, 0.4, 0.2])
    f_hi = lt.label_stats(y, p, 0.8)["F1"]
    f_lo = lt.label_stats(y, p, 0.2)["F1"]
    assert f_hi == pytest.approx(f_lo)
    assert lt.best_f1_threshold(y, p) == pytest.approx(0.7)


def test_calibrate_all_labels_in_contract_order():
    targets, probs = _synthetic()
    out = lt.calibrate(targets, probs, n_bootstrap=50)
    assert list(out) == list(contract.LABELS)
    for j, t in enumerate(out.values()):
        assert t["method"] == "f1"
        assert t["n_pos"] == int(targets[:, j].sum())
        assert t["n_pos"] + t["n_neg"] == len(targets)
        m = t["validation_metrics"]
        assert m["TP"] + m["FN"] == t["n_pos"] and m["TN"] + m["FP"] == t["n_neg"]
        _, f1, _ = lt.f1_curve(targets[:, j], probs[:, j])
        assert m["F1"] == pytest.approx(f1.max())  # naozaj maximum F1
        lo, hi = t["bootstrap"]["threshold_ci95"]
        assert lo <= hi


def test_fallback_when_too_few_examples():
    targets, probs = _synthetic(n=600)
    targets[:, 4] = 0
    targets[:5, 4] = 1  # AFIB: iba 5 pozitívnych
    out = lt.calibrate(targets, probs, n_bootstrap=0)
    afib = out["AFIB"]
    assert afib["method"] == "fallback" and afib["threshold"] == lt.FALLBACK_THRESHOLD
    assert "5 pozitívnych" in afib["fallback_reason"] and "bootstrap" not in afib
    assert out["MI"]["method"] == "f1"


def test_calibrate_rejects_wrong_shape():
    targets, probs = _synthetic()
    with pytest.raises(ValueError, match="tvar"):
        lt.calibrate(targets[:, :4], probs[:, :4])


def test_bootstrap_reproducible():
    targets, probs = _synthetic()
    a = lt.bootstrap(targets[:, 0], probs[:, 0], 0.5, n=50, seed=3)
    b = lt.bootstrap(targets[:, 0], probs[:, 0], 0.5, n=50, seed=3)
    assert a == b


def test_cross_validation_keeps_patients_together_and_is_honest():
    targets, probs = _synthetic(n=1000)
    groups = np.arange(1000) // 2  # 2 záznamy na pacienta
    cv = lt.cross_validated_f1(targets, probs, groups)
    full = lt.calibrate(targets, probs, n_bootstrap=0)
    for name in contract.LABELS:
        # prah zvolený na iných dátach nemôže dať vyššie F1 ako maximum na celých dátach
        assert 0 < cv[name]["f1_cv"] <= full[name]["validation_metrics"]["F1"] + 1e-12


def test_apply_and_load_thresholds(tmp_path):
    data = {
        "version": lt.THRESHOLDS_VERSION,
        "label_order": list(contract.LABELS),
        "labels": {name: {"threshold": 0.5 + 0.1 * j} for j, name in enumerate(contract.LABELS)},
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = lt.load_thresholds(path)
    probs = np.full((1, 5), 0.65)
    assert lt.apply(probs, loaded).tolist() == [[True, True, False, False, False]]

    wrong = dict(data, label_order=list(reversed(contract.LABELS)))
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="poradie"):
        lt.load_thresholds(path)
    path.write_text(json.dumps(dict(data, version="0.9")), encoding="utf-8")
    with pytest.raises(ValueError, match="verzia"):
        lt.load_thresholds(path)


def test_cli_saves_thresholds_with_model_version(tmp_path, monkeypatch):
    targets, probs = _synthetic(n=800)
    groups = np.arange(800) // 3
    ckpt = {"extra": {"epoch": 7}, "versions": {"preprocessing": "1.0", "splits": "1.0"}}
    (tmp_path / "ckpt" / "run").mkdir(parents=True)
    (tmp_path / "ckpt" / "run" / "best.pt").write_bytes(b"model")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        lt, "load_val_predictions", lambda run, amp=False: (probs, targets, groups, ckpt)
    )
    assert lt.main(["--run", "run", "--save"]) == 0

    data = lt.load_thresholds(tmp_path / "ckpt" / "run" / "thresholds.json")
    assert data == json.loads((tmp_path / "docs/vysledky/run/prahy.json").read_text("utf-8"))
    assert data["checkpoint_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert data["checkpoint_epoch"] == 7 and data["model_versions"] == ckpt["versions"]
    assert data["calibrated_on"]["split"] == "val"
    assert all("f1_cv5" in t for t in data["labels"].values())
    assert (tmp_path / "out" / "run" / "prahy_f1.png").exists()
    assert (tmp_path / "out" / "run" / "prahy_porovnanie.csv").exists()


def test_saved_thresholds_documented():
    data = lt.load_thresholds(SAVED)
    assert data["criterion"] == "f1" and data["calibrated_on"]["split"] == "val"
    n = data["calibrated_on"]["n_records"]
    for t in data["labels"].values():
        assert t["method"] == "f1" or "fallback_reason" in t
        assert t["n_pos"] + t["n_neg"] == n
        m = t["validation_metrics"]
        assert m["TP"] + m["FN"] == t["n_pos"] and m["TN"] + m["FP"] == t["n_neg"]
        assert 0 < t["threshold"] < 1


@pytest.mark.skipif(
    not (paths.CHECKPOINT_DIR / "clf_v1" / "best.pt").exists(),
    reason="checkpoint clf_v1 nie je k dispozícii",
)
def test_saved_thresholds_belong_to_checkpoint():
    data = json.loads(SAVED.read_text(encoding="utf-8"))
    digest = hashlib.sha256((paths.CHECKPOINT_DIR / "clf_v1" / "best.pt").read_bytes()).hexdigest()
    assert data["checkpoint_sha256"] == digest
