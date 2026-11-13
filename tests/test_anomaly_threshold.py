"""Testy kalibrácie prahu anomálie (issue #13)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ekg import anomaly_threshold as at
from ekg import contract, paths

LABELS = list(contract.LABELS)


def test_reference_target():
    frame = pd.DataFrame(
        {
            "strict_norm": [True, False, False, False],
            **{label: [0, 0, 0, 0] for label in LABELS},
        },
        index=[1, 2, 3, 4],
    )
    frame.loc[2, "MI"] = 1
    frame.loc[3, ["CD", "AFIB"]] = 1
    target = at.reference_target(frame)
    assert target[1] == 0 and target[2] == 1 and target[3] == 1
    assert np.isnan(target[4])  # nie prísny NORM a bez labelu → vynechať


def test_confusion_counts():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])
    c = at.confusion(y, s, threshold=0.5)
    assert (c["TP"], c["FP"], c["TN"], c["FN"]) == (2, 1, 2, 1)
    assert c["senzitivita"] == pytest.approx(2 / 3)
    assert c["špecificita"] == pytest.approx(2 / 3)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["falošné poplachy NORM"] == pytest.approx(1 / 3)


def test_threshold_is_inclusive():
    """Záznam s chybou presne rovnou prahu je anomália (>=)."""
    c = at.confusion(np.array([1]), np.array([0.5]), 0.5)
    assert c["TP"] == 1


def test_percentile_and_std_rules_use_only_norm():
    rng = np.random.default_rng(0)
    norm = rng.normal(1.0, 0.1, 10_000)
    abnormal = rng.normal(5.0, 0.1, 100)  # nesmie ovplyvniť percentily NORM
    y = np.r_[np.zeros_like(norm), np.ones_like(abnormal)].astype(int)
    th = at.candidate_thresholds(y, np.r_[norm, abnormal])
    assert th["p95"] == pytest.approx(np.quantile(norm, 0.95))
    assert th["mean+2std"] == pytest.approx(norm.mean() + 2 * norm.std(ddof=1))
    assert th["mean"] < th["p95"] < th["p99"]


def test_youden_separates_separable_data():
    y = np.array([0] * 50 + [1] * 50)
    s = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    th = at.candidate_thresholds(y, s)["youden"]
    c = at.confusion(y, s, th)
    assert c["senzitivita"] == 1.0 and c["špecificita"] == 1.0


def test_mean_rule_flags_many_normals_on_skewed_errors():
    """Pri zošikmenom rozdelení chyby označí priemer NORM veľa NORM záznamov ako anomáliu."""
    rng = np.random.default_rng(1)
    norm = rng.lognormal(mean=-4, sigma=0.8, size=5000)
    y = np.r_[np.zeros(5000), np.ones(10)].astype(int)
    s = np.r_[norm, np.full(10, 1.0)]
    th = at.candidate_thresholds(y, s)
    false_alarms = at.confusion(y, s, th["mean"])["falošné poplachy NORM"]
    assert 0.2 < false_alarms < 0.5


def test_needs_both_classes():
    with pytest.raises(ValueError, match="NORM aj patologické"):
        at.candidate_thresholds(np.zeros(5, int), np.arange(5.0))


# --- uložený prah v gite -----------------------------------------------------

SAVED = paths.PROJECT_ROOT / "docs" / "vysledky" / "ae_v1" / "prah.json"


def test_saved_threshold_documented():
    data = json.loads(SAVED.read_text(encoding="utf-8"))
    assert data["rule"] == "youden" and data["aggregation"] == "mean"
    assert data["calibrated_on"]["split"] == "val"
    m = data["validation_metrics"]
    assert m["TP"] + m["FN"] == data["calibrated_on"]["n_abnormal"]
    assert m["TN"] + m["FP"] == data["calibrated_on"]["n_strict_norm"]
    assert 0 < data["threshold"] < 1


@pytest.mark.skipif(
    not (paths.CHECKPOINT_DIR / "ae_v1" / "best.pt").exists(),
    reason="checkpoint ae_v1 nie je k dispozícii",
)
def test_saved_threshold_belongs_to_checkpoint():
    data = json.loads(SAVED.read_text(encoding="utf-8"))
    digest = hashlib.sha256((paths.CHECKPOINT_DIR / "ae_v1" / "best.pt").read_bytes()).hexdigest()
    assert data["checkpoint_sha256"] == digest
