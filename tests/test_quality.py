"""Testy kontroly kvality (issue #7)."""

import numpy as np
import pandas as pd
import pytest

from ekg import contract, quality
from ekg.paths import PROJECT_ROOT

T = np.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE
V5 = contract.LEADS.index("V5")


@pytest.fixture
def clean():
    """Umelé „EKG“: 1,2 Hz zložka s amplitúdou 0,5 mV + slabý šum vo všetkých zvodoch."""
    rng = np.random.default_rng(0)
    x = 0.5 * np.sin(2 * np.pi * 1.2 * T)[None, :] + rng.normal(0, 0.01, (12, contract.N_SAMPLES))
    return x.astype(np.float32)


def test_clean_signal_is_ok(clean):
    q = quality.assess(clean)
    assert q.status == "ok" and q.issues == []


def test_flat_lead_detected(clean):
    clean[V5] = 0.001  # odpojená elektróda
    assert quality.signal_issues(clean) == ["plochý zvod: V5"]


def test_extreme_amplitude_detected(clean):
    clean[0, 500] = 12.0
    issues = quality.signal_issues(clean)
    assert issues == ["extrémna amplitúda: I (12.0 mV)"]


def test_saturation_detected(clean):
    clean[V5, 300:315] = 2.5  # 150 ms orezaný signál
    assert "saturácia signálu: V5" in quality.signal_issues(clean)


def test_short_flat_bottom_is_not_saturation(clean):
    """Dno pomalej vlny s 5 rovnakými vzorkami nie je saturácia (prípad z PTB-XL)."""
    clean[V5] = -1.76 * np.exp(-(((T - 5) / 0.3) ** 2))
    clean[V5, 498:503] = clean[V5].min()
    assert quality.signal_issues(clean) == []


def test_multiple_issues_and_leads(clean):
    clean[V5] = 0.0
    clean[contract.LEADS.index("V6")] = 0.0
    clean[1, 10] = -15.0
    issues = quality.signal_issues(clean)
    assert issues[0] == "plochý zvod: V5, V6"
    assert issues[1].startswith("extrémna amplitúda: II")


def test_invalid_signal_still_rejected_by_contract(clean):
    clean[0, 0] = np.nan
    with pytest.raises(contract.InvalidInputError):
        quality.assess(clean)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" , I-V1,  ", "I-V1"),
        (" , alles,  ", "všetky zvody"),
        ("v4, v5", "V4, V5"),
        ("V1???", "V1?"),
        ("   ", None),
        (np.nan, None),
        (None, None),
    ],
)
def test_clean_annotation(raw, expected):
    assert quality._clean_annotation(raw) == expected


def test_annotations_add_issues(clean):
    row = {"static_noise": " , I-AVF,  ", "burst_noise": np.nan, "electrodes_problems": "V6"}
    q = quality.assess(clean, row)
    assert q.status == "warning"
    assert q.issues == ["statický šum (I-AVF)", "problém s elektródami (V6)"]


def test_assess_dataset_and_report(clean):
    flat = clean.copy()
    flat[V5] = 0.0
    signals = np.stack([clean, flat, clean])
    db = pd.DataFrame(
        {col: [np.nan, np.nan, " , V1,  "] for col in quality.ANNOTATIONS},
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )
    q = quality.assess_dataset(signals, np.array([1, 2, 3]), db)
    assert q["status"].tolist() == ["ok", "warning", "warning"]
    assert q.loc[2, "flat_lead"] and not q.loc[2, "static_noise"]
    assert q.loc[3, "static_noise"] and q.loc[3, "baseline_drift"]
    text = quality.report(q, pd.Series(["train", "val", "test"], index=[1, 2, 3]))
    assert "Vyradené záznamy: 0" in text


# --- uložený výsledok pre PTB-XL ---------------------------------------------

SAVED = PROJECT_ROOT / "manifests" / f"quality_v{quality.QUALITY_VERSION}.csv"


def test_saved_quality_manifest():
    q = pd.read_csv(SAVED, index_col="ecg_id", keep_default_na=False)
    assert len(q) == 21799
    assert set(q["status"]) == {"ok", "warning"}  # nič sa nevyradilo
    warned = q["status"] == "warning"
    assert (q.loc[warned, "issues"] != "").all()  # každé označenie má dôvod
    assert (q.loc[~warned, "issues"] == "").all()
    # jediný plochý zvod v PTB-XL – a PTB-XL ho má označený ako problém s elektródami
    assert q.index[q["flat_lead"]].tolist() == [12722]
    assert q.loc[12722, "electrodes_problems"]
