"""Testy externej validácie na Georgia (issue #31)."""

import json

import numpy as np
import pandas as pd
import pytest

from ekg import contract, paths
from ekg.evaluation import external as ex

LABELS = list(contract.LABELS)
RESULTS = paths.PROJECT_ROOT / "docs" / "vysledky" / "externa"

HEADER = """E00001 12 500 5000
E00001.mat 16x1+24 1000.0(0)/mV 16 0 136 -28477 0 I
# Age: 65
# Sex: Female
# Dx: 426783006,164889003
# Rx: Unknown
"""


def test_parse_header():
    info = ex.parse_header(HEADER)
    assert info == {"fs": 500, "n_samples": 5000, "dx": ["426783006", "164889003"],
                    "age": 65, "sex": "female"}  # fmt: skip
    nan = ex.parse_header(HEADER.replace("Age: 65", "Age: NaN").replace("Female", "Unknown"))
    assert nan["age"] is None and nan["sex"] is None


@pytest.mark.parametrize(
    ("codes", "expected", "usable", "ref_a"),
    [
        (["426783006"], {}, True, 0.0),  # iba sínusový rytmus = NORM pre Model A
        (["426177001"], {}, True, None),  # sínusová bradykardia: B áno, referencia A nie
        (["164889003"], {"AFIB": 1}, True, 1.0),
        (["164890007"], {}, False, None),  # flutter nie je AFIB, ani bežný rytmus
        (["164934002", "426783006"], {"STTC": None}, True, None),  # iba nejasný nález T
        (["164934002", "428750005"], {"STTC": 1}, True, 1.0),  # nejasný + pozitívny
        (["59118001", "39732003"], {"CD": 1}, True, 1.0),  # RBBB + nenamapovaná os
        (["39732003"], {}, False, None),  # iba odchýlka osi – nevieme
        (["55827005", "164873001"], {"HYP": 1}, True, 1.0),
    ],
)
def test_labels_from_dx(codes, expected, usable, ref_a):
    out = ex.labels_from_dx(codes)
    for name in LABELS:
        want = expected.get(name, 0)
        if want is None:
            assert np.isnan(out[name]), name
        else:
            assert out[name] == want, name
    assert out["usable_classifier"] is usable
    assert np.isnan(out["reference_a"]) if ref_a is None else out["reference_a"] == ref_a


def test_mapping_consistent():
    labels = {label for _, label, _ in ex.MAPPING.values()}
    assert labels == set(LABELS)
    assert {kind for _, _, kind in ex.MAPPING.values()} == {"poz", "nej"}
    assert "164890007" not in ex.MAPPING  # flutter
    assert not set(ex.MAPPING) & ex.BENIGN_RHYTHMS


def test_resample_keeps_frequency_and_amplitude():
    t = np.arange(5000) / 500
    x = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1))  # 5 Hz
    y = ex.resample_to_model(x, 500)
    assert y.shape == (12, 1000) and y.dtype == np.float32
    t100 = np.arange(1000) / 100
    assert np.allclose(y[0, 50:-50], np.sin(2 * np.pi * 5 * t100)[50:-50], atol=0.01)
    noise = np.tile(np.sin(2 * np.pi * 120 * t), (12, 1))  # nad Nyquistom 50 Hz → odfiltruje
    assert np.abs(ex.resample_to_model(noise, 500)[:, 50:-50]).max() < 0.05
    assert ex.resample_to_model(x[:, :4999], 500) is None  # kratšie ako 10 s
    assert ex.resample_to_model(np.zeros((12, 1000)), 100).shape == (12, 1000)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.random((n, 5)) < 0.3
    frame = pd.DataFrame({n_: truth[:, j].astype(float) for j, n_ in enumerate(LABELS)})
    for j, name in enumerate(LABELS):
        frame[f"prob_{name}"] = np.clip(truth[:, j] * 0.4 + rng.random(n) * 0.6, 0, 1)
    frame.loc[:20, "STTC"] = np.nan  # nejasné
    abnormal = truth.any(axis=1)
    frame["reference_a"] = np.where(abnormal, 1.0, np.where(rng.random(n) < 0.7, 0.0, np.nan))
    frame["error"] = abnormal * 0.3 + rng.random(n)
    frame["usable_classifier"] = rng.random(n) < 0.9
    return frame


def test_evaluate_excludes_unclear_and_unusable():
    from sklearn.metrics import roc_auc_score

    frame = _frame()
    res = ex.evaluate(frame, np.full(5, 0.5), 0.8, n_bootstrap=20)
    clf = frame[frame["usable_classifier"]]
    sttc = clf[clf["STTC"].notna()]
    m = res["labels"]["STTC"]
    assert m["support"] + m["n_neg"] == len(sttc)
    assert m["n_unclear"] == int(clf["STTC"].isna().sum())
    assert m["AUROC"] == pytest.approx(roc_auc_score(sttc["STTC"], sttc["prob_STTC"]))
    assert len(m["AUROC_ci95"]) == 2
    a = res["model_a"]
    used = frame["reference_a"].notna()
    assert a["n_norm"] + a["n_abnormal"] == used.sum()
    y = frame.loc[used, "reference_a"] == 1
    assert a["AUROC"] == pytest.approx(roc_auc_score(y, frame.loc[used, "error"]))
    assert res["n_classifier"] == len(clf)


def test_positivity_by_code():
    frame = _frame(300)
    frame["dx"] = [["164889003"] if i % 2 else ["426783006", "39732003"] for i in range(300)]
    frame["is_anomaly"] = frame["error"] > 0.8
    tab = ex.positivity_by_code(frame, np.full(5, 0.5), min_records=100).set_index("skratka")
    assert set(tab.index) == {"AF", "NSR", "LAD"} and tab.loc["AF", "záznamov"] == 150
    odd = frame.iloc[1::2]
    assert tab.loc["AF", "B AFIB"] == pytest.approx((odd["prob_AFIB"] >= 0.5).mean())
    assert tab.loc["LAD", "mapovanie"] == "–" and tab.loc["AF", "mapovanie"] == "AFIB (poz)"


def test_no_test_set_or_ptbxl_signals_used():
    import inspect

    source = inspect.getsource(ex)
    assert "allow_test" not in source  # z PTB-XL iba validation (porovnanie signálov)
    assert 'split_ids("val"' in source and 'split_ids("test"' not in source


# --- uložené výsledky -----------------------------------------------------------------


@pytest.mark.skipif(not (RESULTS / "report.json").exists(), reason="výsledky ešte nie sú")
def test_saved_report_is_consistent():
    report = json.loads((RESULTS / "report.json").read_text("utf-8"))
    c = report["counts"]
    assert c["záznamov stiahnutých"] == 10344
    assert c["vyhodnotených"] + c["vynechaných"] == c["záznamov stiahnutých"]
    assert (
        c["vynechaných"]
        == c["vynechaných – kratšie ako 10 s"] + c["vynechaných – neplatný signál (NaN)"]
    )
    assert report["license"] == "CC BY 4.0" and report["doi"] == "10.13026/34va-7q14"
    for name in LABELS:
        m = report["labels"][name]
        assert m["support"] == c["pozitívnych (Model B)"][name]
    mapping = pd.read_csv(RESULTS / "mapovanie.csv", dtype={"SNOMED": str})
    assert set(mapping["SNOMED"]) == set(ex.MAPPING)
