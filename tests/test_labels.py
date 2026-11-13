"""Testy mapovania labelov a výberu prísneho NORM (issue #4)."""

import pandas as pd
import pytest

from ekg import contract
from ekg.data import labels
from ekg.data.download import PTBXL_DIR


@pytest.fixture
def scp():
    """Výrez scp_statements.csv s rovnakou štruktúrou ako originál."""
    rows = {
        # kód: (diagnostic, form, rhythm, diagnostic_class)
        "NORM": (1, None, None, "NORM"),
        "IMI": (1, None, None, "MI"),
        "NDT": (1, 1, None, "STTC"),  # v PTB-XL je NDT diagnostický aj form kód
        "CLBBB": (1, None, None, "CD"),
        "LVH": (1, 1, None, "HYP"),
        "SR": (None, None, 1, None),
        "AFIB": (None, None, 1, None),
        "AFLT": (None, None, 1, None),
        "STACH": (None, None, 1, None),
        "PVC": (None, 1, None, None),
        "LVOLT": (None, 1, None, None),
    }
    df = pd.DataFrame.from_dict(
        rows, orient="index", columns=["diagnostic", "form", "rhythm", "diagnostic_class"]
    )
    df["description"] = df.index
    return df


EXAMPLES = {
    # ecg_id: (scp_codes, quality columns)
    1: ({"NORM": 100.0, "SR": 0.0}, {}),  # čistý NORM
    2: ({"NORM": 100.0, "SR": 0.0}, {"static_noise": ", I-V1,"}),  # NORM so šumom
    3: ({"NORM": 80.0, "SR": 0.0}, {}),  # NORM s nižšou istotou
    4: ({"NORM": 100.0, "IMI": 15.0, "SR": 0.0}, {}),  # NORM + nepravdepodobný MI
    5: ({"AFIB": 0.0, "NDT": 100.0}, {}),  # AFIB (likelihood 0) + STTC
    6: ({"IMI": 100.0, "LVH": 50.0, "CLBBB": 100.0, "SR": 0.0}, {}),  # viac labelov
    7: ({"AFLT": 0.0}, {}),  # iba flutter – žiadny label
    8: ({"NORM": 100.0, "AFIB": 0.0}, {}),  # NORM + AFIB
    9: ({"NORM": 100.0, "SR": 0.0, "PVC": 0.0}, {}),  # NORM s extrasystolou
    10: ({"NORM": 100.0, "STACH": 0.0}, {}),  # sínusová tachykardia
    11: ({"NORM": 100.0, "SR": 0.0}, {"pacemaker": "ja, pacemaker"}),
    12: ({"NORM": 100.0, "SR": 0.0, "LVOLT": 0.0}, {"baseline_drift": "   "}),  # prázdny text
}


@pytest.fixture
def db():
    quality_cols = (*labels.NOISE_COLUMNS, *labels.OTHER_QUALITY_COLUMNS)
    rows = []
    for ecg_id, (codes, quality) in EXAMPLES.items():
        row = {"ecg_id": ecg_id, "scp_codes": codes, "strat_fold": 1}
        row.update({col: quality.get(col) for col in quality_cols})
        rows.append(row)
    return pd.DataFrame(rows).set_index("ecg_id")


def row(targets, ecg_id):
    return {k: int(targets.loc[ecg_id, k]) for k in contract.LABELS}


# --- labely pre Model B ------------------------------------------------------


def test_label_map(scp):
    m = labels.label_map(scp)
    assert m == {"IMI": "MI", "NDT": "STTC", "CLBBB": "CD", "LVH": "HYP", "AFIB": "AFIB"}
    assert "AFLT" not in m and "NORM" not in m


def test_norm_record_has_all_zero_targets(db, scp):
    t = labels.make_targets(db, scp)
    assert row(t, 1) == dict.fromkeys(contract.LABELS, 0)
    assert t.loc[1, "usable"]


def test_low_likelihood_diagnostic_code_counts(db, scp):
    """Benchmark Strodthoff: kód sa počíta bez ohľadu na likelihood."""
    assert labels.make_targets(db, scp).loc[4, "MI"] == 1


def test_afib_with_zero_likelihood_counts(db, scp):
    t = labels.make_targets(db, scp)
    assert row(t, 5) == {"MI": 0, "STTC": 1, "CD": 0, "HYP": 0, "AFIB": 1}
    assert t.loc[8, "AFIB"] == 1


def test_multilabel(db, scp):
    t = labels.make_targets(db, scp)
    assert row(t, 6) == {"MI": 1, "STTC": 0, "CD": 1, "HYP": 1, "AFIB": 0}


def test_flutter_only_is_not_usable(db, scp):
    t = labels.make_targets(db, scp)
    assert row(t, 7) == dict.fromkeys(contract.LABELS, 0)
    assert not t.loc[7, "usable"]


# --- prísny NORM -------------------------------------------------------------


@pytest.mark.parametrize(
    "ecg_id, reason",
    [
        (1, None),
        (2, "šum alebo problém s elektródami"),
        (3, "nie je NORM s likelihood 100"),
        (4, "iný diagnostický kód"),
        (5, "nie je NORM s likelihood 100"),
        (8, "rytmus iný ako SR"),
        (9, "form kód"),
        (10, "rytmus iný ako SR"),
        (11, "baseline drift, extra beats alebo kardiostimulátor"),
        (12, "form kód"),  # prázdny text v baseline_drift sa nepočíta ako problém
    ],
)
def test_norm_exclusion_reason(db, scp, ecg_id, reason):
    assert labels.norm_exclusion_reason(db, scp)[ecg_id] == reason


def test_strict_norm_only_clean_record(db, scp):
    assert list(db.index[labels.strict_norm(db, scp)]) == [1]


# --- skutočný PTB-XL (preskočí sa, ak nie je stiahnutý) ----------------------

real = pytest.mark.skipif(
    not (PTBXL_DIR / "ptbxl_database.csv").exists(), reason="PTB-XL nie je stiahnutý"
)


@pytest.fixture(scope="module")
def real_data():
    from ekg.data import ptbxl

    return ptbxl.load_database(), ptbxl.load_scp_statements()


@real
def test_real_label_counts(real_data):
    db, scp = real_data
    t = labels.make_targets(db, scp)
    assert t[list(contract.LABELS)].sum().to_dict() == {
        "MI": 5469,
        "STTC": 5235,
        "CD": 4898,
        "HYP": 2649,
        "AFIB": 1514,
    }
    assert int((~t["usable"]).sum()) == 348


@real
def test_real_strict_norm_counts(real_data):
    db, scp = real_data
    norm = labels.strict_norm(db, scp)
    by_split = {
        "train": int((norm & (db.strat_fold <= 8)).sum()),
        "val": int((norm & (db.strat_fold == 9)).sum()),
        "test": int((norm & (db.strat_fold == 10)).sum()),
    }
    assert by_split == {"train": 3905, "val": 454, "test": 464}


@real
def test_real_mapping_covers_all_label_classes(real_data):
    _, scp = real_data
    table = labels.mapping_table(scp)
    assert len(table) == 71
    assert set(table["label"]) - {""} == set(contract.LABELS)
    assert table.loc["AFIB", "label"] == "AFIB" and table.loc["AFLT", "label"] == ""


@real
def test_committed_mapping_table_is_up_to_date(real_data):
    """docs/mapovanie_labelov_v1.0.csv musí zodpovedať kódu – inak treba zvýšiť verziu."""
    from ekg.paths import PROJECT_ROOT

    _, scp = real_data
    committed = pd.read_csv(
        PROJECT_ROOT / "docs" / f"mapovanie_labelov_v{labels.MAPPING_VERSION}.csv",
        index_col=0,
        keep_default_na=False,
    )
    generated = labels.mapping_table(scp).fillna("")
    assert committed.astype(str).equals(generated.astype(str))
