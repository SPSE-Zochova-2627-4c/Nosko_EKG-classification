"""Testy analýzy chýb autoencodera (issue #14) na malej umelej tabuľke."""

import numpy as np
import pandas as pd
import pytest

from ekg import contract
from ekg.evaluation import ae_analysis as aa

LABELS = list(contract.LABELS)


@pytest.fixture
def frame():
    """6 záznamov: 2 prísny NORM, 3 s patológiou, 1 vynechaný."""
    rows = [
        # ecg_id, error, strict_norm, MI, STTC, CD, HYP, AFIB
        (1, 0.01, True, 0, 0, 0, 0, 0),  # TN
        (2, 0.09, True, 0, 0, 0, 0, 0),  # FP
        (3, 0.20, False, 1, 0, 0, 0, 0),  # TP, iba MI
        (4, 0.02, False, 0, 1, 0, 0, 0),  # FN, iba STTC
        (5, 0.30, False, 1, 0, 1, 0, 0),  # TP, 2 labely
        (6, 0.50, False, 0, 0, 0, 0, 0),  # vynechaný
    ]
    df = pd.DataFrame(rows, columns=["ecg_id", "error", "strict_norm", *LABELS]).set_index("ecg_id")
    for i, lead in enumerate(contract.LEADS):
        df[f"error_{lead}"] = df["error"] * (1 + i / 10)
    return aa.add_predictions(df, threshold=0.05)


def test_outcomes(frame):
    assert frame["outcome"].to_dict() == {
        1: "TN",
        2: "FP",
        3: "TP",
        4: "FN",
        5: "TP",
        6: "vynechaný",
    }


def test_summary_metrics(frame):
    m = aa.summary_metrics(frame, 0.05)
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (2, 1, 1, 1)
    assert m["n_excluded"] == 1 and m["n_strict_norm"] == 2 and m["n_abnormal"] == 3
    assert m["senzitivita"] == pytest.approx(2 / 3)
    assert isinstance(m["TP"], int)


def test_detection_by_label(frame):
    det = aa.detection_by_label(frame)
    assert det.loc["MI", "záznamy"] == 2 and det.loc["MI", "zachytené"] == 1.0
    assert det.loc["MI", "iba tento label – záznamy"] == 1
    assert det.loc["STTC", "zachytené"] == 0.0
    assert np.isnan(det.loc["HYP", "zachytené"])  # žiadny záznam – metrika nedefinovaná, nie 0
    assert det.loc["2 a viac labelov", "záznamy"] == 1


def test_excluded_groups(frame):
    reasons = pd.Series({6: "šum alebo problém s elektródami"})
    excl = aa.excluded_groups(frame, reasons)
    assert excl.loc["šum alebo problém s elektródami", "záznamy"] == 1
    assert excl.loc["šum alebo problém s elektródami", "označené ako anomália"] == 1.0


def test_lead_error_profile(frame):
    prof = aa.lead_error_profile(frame)
    assert list(prof.columns) == list(contract.LEADS)
    assert prof.loc["prísny NORM", "I"] == pytest.approx(np.median([0.01, 0.09]))


def test_pick_examples(frame):
    ex = aa.pick_examples(frame)
    assert ex["TP"] == 5 and ex["FP"] == 2 and ex["FN"] == 4 and ex["TN"] == 1


def test_mi_localization(frame):
    codes = pd.Series({3: {"IMI": 100.0}, 5: {"ASMI": 100.0, "CLBBB": 100.0}})
    frame.loc[3, "error_V6"] = 10.0  # inferiórny MI má jednoznačne najhorší zvod V6
    loc = aa.mi_localization(frame, codes)
    assert loc.loc["inferiórny MI (IMI)", "záznamy"] == 1
    assert loc.loc["anteroseptálny MI (ASMI)", "záznamy"] == 1
    assert loc.loc["inferiórny MI (IMI)", "zvody s najväčšou chybou"].startswith("V6")
