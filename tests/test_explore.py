"""Testy prieskumu datasetu (issue #5) na malej umelej tabuľke."""

import numpy as np
import pandas as pd
import pytest

from ekg import contract
from ekg.data import explore

LABELS = list(contract.LABELS)


@pytest.fixture
def inputs():
    ids = pd.Index([1, 2, 3, 4, 5, 6], name="ecg_id")
    manifest = pd.DataFrame(
        {
            "patient_id": [10, 10, 11, 12, 13, 14],  # pacient 10 má 2 záznamy
            "split": ["train", "train", "train", "val", "test", "test"],
            "MI": [0, 1, 1, 0, 0, 0],
            "STTC": [0, 1, 0, 0, 1, 0],
            "CD": [0, 0, 1, 0, 0, 0],
            "HYP": [0, 0, 0, 0, 1, 0],
            "AFIB": [0, 0, 0, 0, 1, 0],
            "usable_classifier": [True, True, True, True, True, False],
            "strict_norm": [True, False, False, False, False, False],
        },
        index=ids,
    )
    db = pd.DataFrame(
        {
            "scp_codes": [{"NORM": 100}, {"IMI": 100}, {"IMI": 50}, {"NORM": 80}, {"NDT": 100}, {}],
            "age": [50.0, 300.0, 70.0, np.nan, 65.0, 40.0],
            "sex": [0, 1, 0, 1, 1, 0],
            "validated_by_human": [True, True, False, True, True, True],
            "height": [180, np.nan, np.nan, 160, np.nan, np.nan],
            "weight": [80, 70, np.nan, np.nan, np.nan, np.nan],
            "recording_date": ["1990-01-01"] * 5 + ["1995-06-01"],
            "device": ["A", "A", "B", "A", "B", "C"],
        },
        index=ids,
    )
    quality = pd.DataFrame({"status": ["ok", "warning", "ok", "ok", "warning", "ok"]}, index=ids)
    return db, manifest, quality


@pytest.fixture
def table(inputs):
    return explore.build_table(*inputs)


def test_build_table_age_300_is_not_a_real_age(table):
    assert table.loc[2, "age_over_89"] and np.isnan(table.loc[2, "age"])
    assert table.loc[1, "sex"] == "muž" and table.loc[2, "sex"] == "žena"
    assert table["NORM"].tolist() == [1, 0, 0, 1, 0, 0]
    assert table["n_labels"].tolist() == [0, 2, 2, 0, 3, 0]


def test_basic_counts(table, inputs):
    db = inputs[0]
    b = explore.basic_counts(table, db)
    assert b["záznamy"] == 6 and b["pacienti"] == 5
    assert b["záznamov na pacienta (max)"] == 2 and b["pacienti s 1 záznamom"] == 4
    assert b["recording_date (rozsah v CSV)"] == "1990-01-01 – 1995-06-01"
    assert b["typy prístrojov"] == 3


def test_missing_values_separate_anonymized_age(table, inputs):
    miss = explore.missing_values(table, inputs[0])
    assert miss.loc["vek chýba", "záznamy"] == 1  # iba skutočné NaN, nie vek 300
    assert miss.loc["vek > 89 (PTB-XL uvádza 300)", "záznamy"] == 1
    assert miss.loc["výška chýba", "záznamy"] == 4
    assert miss.loc["bez diagnostického kódu aj AFIB (vynechané z Modelu B)", "záznamy"] == 1


def test_label_counts_denominators(table):
    lc = explore.label_counts(table)
    assert lc.loc["MI", "train"] == 2 and lc.loc["MI", "train %"] == pytest.approx(100 * 2 / 3)
    assert lc.loc["MI", "spolu"] == 2 and lc.loc["MI", "spolu %"] == pytest.approx(100 * 2 / 6)
    assert lc.loc["prísny NORM", "spolu"] == 1
    assert lc["spolu"].dtype.kind == "i"


def test_labels_per_record(table):
    assert explore.labels_per_record(table).to_dict() == {0: 3, 2: 2, 3: 1}


def test_cooccurrence(table):
    counts, cond = explore.cooccurrence(table)
    assert counts.loc["MI", "STTC"] == 1 and counts.loc["MI", "MI"] == 2
    assert cond.loc["MI", "STTC"] == pytest.approx(0.5)  # P(STTC | MI)
    assert cond.loc["STTC", "MI"] == pytest.approx(0.5)  # P(MI | STTC)
    assert (np.diag(cond) == 1).all()
    assert counts.equals(counts.T)


def test_top_combinations(table):
    combos = explore.top_combinations(table)
    assert combos.loc["NORM", "záznamy"] == 2
    assert combos.loc["STTC + HYP + AFIB", "záznamy"] == 1
    assert combos.loc["bez labelu", "záznamy"] == 1
    assert combos["záznamy"].sum() == 6


def test_demographics_by_label(table):
    demo = explore.demographics_by_label(table)
    assert demo.loc["MI", "záznamy"] == 2
    assert demo.loc["MI", "vek medián"] == 70.0  # záznam s vekom 300 sa do mediánu nepočíta
    assert demo.loc["MI", "vek > 89 %"] == 50.0
    assert demo.loc["všetky", "kvalita warning %"] == pytest.approx(100 * 2 / 6)
