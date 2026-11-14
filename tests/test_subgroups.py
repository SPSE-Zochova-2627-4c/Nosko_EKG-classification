"""Testy analýzy podľa podskupín (issue #26)."""

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ekg import contract, paths
from ekg.evaluation import subgroups as sg

LABELS = list(contract.LABELS)
RESULTS = paths.PROJECT_ROOT / "docs" / "vysledky" / "podskupiny"
TEST_REPORT = paths.PROJECT_ROOT / "docs" / "vysledky" / "test" / "report.json"


def _frame(n=240, seed=0):
    """Syntetické predikcie s metadátami, labelmi a pacientmi (2 záznamy na pacienta)."""
    rng = np.random.default_rng(seed)
    truth = rng.random((n, 5)) < 0.3
    prob = np.clip(truth * 0.5 + rng.random((n, 5)) * 0.6, 0, 1)
    abnormal = truth.any(axis=1)
    frame = pd.DataFrame(
        {
            "error": abnormal * 0.4 + rng.random(n),
            "age": rng.integers(20, 89, n).astype(float),
            "age_over_89": False,
            "sex": rng.choice(["male", "female"], n),
            "quality_status": "ok",
            "quality_issues": [[] for _ in range(n)],
            "strict_norm": ~abnormal & (rng.random(n) < 0.8),
            "usable_classifier": rng.random(n) < 0.95,
            "patient_id": np.arange(n) // 2,
        },
        index=pd.Index(np.arange(1, n + 1), name="ecg_id"),
    )
    frame["is_anomaly"] = frame["error"] >= 0.7
    for j, name in enumerate(LABELS):
        frame[name] = truth[:, j].astype(int)
        frame[f"prob_{name}"] = prob[:, j]
        frame[f"positive_{name}"] = prob[:, j] >= 0.5
    return frame


# --- metriky --------------------------------------------------------------------------


def test_auroc_equals_sklearn_including_ties():
    rng = np.random.default_rng(1)
    y = rng.random(300) < 0.4
    s = np.round(rng.random(300) + y * 0.3, 1)  # veľa zhodných skóre
    assert sg.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert sg.auroc(np.zeros(5, bool), s[:5]) is None
    assert sg.auroc(np.ones(5, bool), s[:5]) is None


def test_f1_undefined_without_positives():
    assert sg.f1(np.array([0, 0, 0]), np.array([1, 0, 0])) is None
    assert sg.f1(np.array([1, 0, 1]), np.array([1, 1, 0])) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("key", "value", "cases", "records", "expected"),
    [
        (("B", "AUROC", "AFIB"), None, (0, 40), 40, ("nedefinovaná", "0 pozitívnych")),
        (("A", "AUROC", ""), None, (50, 0), 50, ("nedefinovaná", "0 prísnych NORM")),
        (("A", "špecificita", ""), 0.9, (5, 100), 105, ("ok", "")),  # potrebuje iba NORM
        (("A", "senzitivita", ""), 0.6, (5, 100), 105, ("malá skupina", "< 30 patológií")),
        (("B", "macro F1", ""), None, (0, 10), 0, ("nedefinovaná", "skupina nemá žiadny záznam")),
        (("B", "F1", "MI"), 0.7, (300, 2000), 2300, ("ok", "")),
        (
            ("A", "poplachy bez labelu", ""),
            0.2,
            (0, 12),
            50,
            ("malá skupina", "< 30 záznamov bez labelu"),
        ),
    ],
)
def test_status(key, value, cases, records, expected):
    assert sg.status(key, value, cases, records) == expected


def test_macro_requires_all_labels():
    a = sg.Arrays.from_frame(_frame())
    a.truth[:, 4] = False  # AFIB bez pozitívnych
    m = sg.metrics(a, np.arange(len(a.patient)))
    assert m[("B", "AUROC", "AFIB")] is None and m[("B", "macro AUROC", "")] is None
    assert m[("B", "macro F1", "")] is None and m[("B", "micro F1", "")] is not None


def test_metrics_match_direct_computation():
    frame = _frame()
    a = sg.Arrays.from_frame(frame)
    m = sg.metrics(a, np.arange(len(frame)))
    used = frame["strict_norm"] | (frame[LABELS].sum(axis=1) > 0)
    y = frame.loc[used, LABELS].sum(axis=1) > 0
    assert m[("A", "AUROC", "")] == pytest.approx(roc_auc_score(y, frame.loc[used, "error"]))
    anom = frame.loc[used, "is_anomaly"]
    assert m[("A", "senzitivita", "")] == pytest.approx(anom[y].mean())
    assert m[("A", "špecificita", "")] == pytest.approx((~anom[~y]).mean())
    clf = frame[frame["usable_classifier"]]
    aucs = [roc_auc_score(clf[n], clf[f"prob_{n}"]) for n in LABELS]
    assert m[("B", "macro AUROC", "")] == pytest.approx(np.mean(aucs))


# --- podskupiny -----------------------------------------------------------------------


def test_age_groups_boundaries_and_missing():
    frame = pd.DataFrame(
        {
            "age": [39, 40, 59, 60, 79, 80, None, None],
            "age_over_89": [False] * 6 + [True, False],
        }
    )
    assert sg.age_groups(frame).tolist() == [
        "< 40", "40–59", "40–59", "60–79", "60–79", "≥ 80", "≥ 80", "neznámy",
    ]  # fmt: skip


def test_subgroups_always_contain_missing_groups_and_overlapping_quality():
    frame = _frame(10)
    frame["quality_status"] = ["warning"] * 3 + ["ok"] * 7
    frame["quality_issues"] = [
        ["statický šum (I-AVR)", "drift základnej línie (V1)"],
        ["plochý zvod: V5"],
        ["nárazový šum (V2)"],
    ] + [[]] * 7
    groups = {(d, n): m for d, n, m in sg.subgroups(frame)}
    assert groups[("vek", "neznámy")].sum() == 0  # prázdna, ale v reporte je
    assert groups[("pohlavie", "neznáme")].sum() == 0
    assert groups[("kvalita (#7)", "s varovaním")].sum() == 3
    types = {n: groups[("typ problému kvality", n)].sum() for n in sg.QUALITY_TYPES}
    assert types == {
        "drift základnej línie": 1,
        "statický šum": 1,
        "nárazový šum": 1,
        "problém s elektródami": 0,
        "problém zo signálu": 1,
    }
    assert groups[("typ problému kvality", "statický šum")][0]  # záznam 0 má dva typy naraz
    assert groups[("typ problému kvality", "drift základnej línie")][0]


def test_group_sizes_add_up():
    frame = _frame()
    sizes = sg.group_sizes(frame, "val").set_index(["dimenzia", "skupina"])
    total = sizes.loc[sg.ALL, "záznamov"]
    assert sizes.loc["vek", "záznamov"].sum() == total == len(frame)
    assert sizes.loc["pohlavie", "záznamov"].sum() == total
    assert sizes.loc["kvalita (#7)", "záznamov"].sum() == total
    assert sizes.loc[sg.ALL, "pacientov"] == len(frame) // 2
    row = sizes.loc[sg.ALL]
    assert row["A prísny NORM"] + row["A patológia"] + row["A vynechaných"] == total
    assert row["B MI"] == frame.loc[frame["usable_classifier"], "MI"].sum()


def test_group_metrics_table():
    frame = _frame()
    table = sg.group_metrics(frame, "val", n_bootstrap=50)
    again = sg.group_metrics(frame, "val", n_bootstrap=50)
    pd.testing.assert_frame_equal(table, again)  # deterministické
    empty = table[(table["skupina"] == "neznáme")]
    assert (empty["stav"] == "nedefinovaná").all() and empty["hodnota"].isna().all()
    assert set(empty["dôvod"]) == {"skupina nemá žiadny záznam"}
    whole = table[(table["skupina"] == "všetky") & (table["stav"] != "nedefinovaná")]
    assert len(whole) == 4 + 3 + 15  # Model A 4, macro/micro 3, po labeloch 15
    defined = whole.dropna(subset=["ci95_dolna"])
    assert (defined["ci95_dolna"] <= defined["ci95_horna"]).all()
    assert len(defined) == len(whole)
    # pri 240 záznamoch má aspoň jeden label < 30 pozitívnych → stav „malá skupina“
    assert (table["stav"] == "malá skupina").any()


def test_bootstrap_resamples_whole_patients():
    frame = _frame(40)
    a = sg.Arrays.from_frame(frame)
    seen = []
    original = sg.metrics

    def spy(arrays, idx):
        seen.append(np.asarray(idx))
        return original(arrays, idx)

    sg.metrics = spy
    try:
        sg.bootstrap(a, 5, seed=0)
    finally:
        sg.metrics = original
    for idx in seen:
        patients = a.patient[idx]
        counts = pd.Series(idx).groupby(patients).size()
        assert (counts % 2 == 0).all()  # každý pacient má 2 záznamy – vždy idú spolu


# --- test množina: iba uložené predikcie ------------------------------------------------


def test_test_split_uses_only_saved_predictions():
    source = inspect.getsource(sg.load_test)
    for forbidden in ("Pipeline", "inference", "predict_batch", "ekg.models"):
        assert forbidden not in source
    assert "load_csv(test_predictions_path())" in source


@pytest.mark.skipif(
    not (paths.PROJECT_ROOT / "manifests" / "splits_v1.0.csv").exists(),
    reason="manifest nie je k dispozícii",
)
def test_all_records_row_reproduces_final_test_report():
    """Riadok „všetky“ na teste = report #22 (rovnaké predikcie, rovnaké metriky)."""
    from ekg.data import splits

    frame = sg.load_test(splits.load_manifest())
    a = sg.Arrays.from_frame(frame)
    m = sg.metrics(a, np.arange(len(frame)))
    report = json.loads(TEST_REPORT.read_text("utf-8"))
    ae, clf = report["autoencoder"], report["classifier"]
    assert m[("A", "AUROC", "")] == pytest.approx(ae["AUROC"], abs=1e-9)
    assert m[("A", "senzitivita", "")] == pytest.approx(ae["senzitivita"], abs=1e-9)
    assert m[("A", "špecificita", "")] == pytest.approx(ae["špecificita"], abs=1e-9)
    assert m[("B", "macro AUROC", "")] == pytest.approx(clf["macro"]["AUROC"], abs=1e-9)
    assert m[("B", "macro F1", "")] == pytest.approx(clf["macro"]["F1"], abs=1e-9)
    assert m[("B", "micro F1", "")] == pytest.approx(clf["micro"]["F1"], abs=1e-9)
    for name in LABELS:
        assert m[("B", "F1", name)] == pytest.approx(clf["labels"][name]["F1"], abs=1e-9)


# --- uložené výsledky -----------------------------------------------------------------


def test_saved_results_are_consistent():
    sizes = pd.read_csv(RESULTS / "velkosti_skupin.csv")
    table = pd.read_csv(RESULTS / "metriky.csv")
    report = json.loads((RESULTS / "report.json").read_text("utf-8"))
    assert report["n_records"] == {"val": 2183, "test": 2198}
    for split, n in report["n_records"].items():
        s = sizes[sizes["množina"] == split].set_index(["dimenzia", "skupina"])
        assert s.loc[sg.ALL, "záznamov"] == n
        for dim in ("vek", "pohlavie", "kvalita (#7)"):
            assert s.loc[dim, "záznamov"].sum() == n
        assert ("vek", "neznámy") in s.index and ("pohlavie", "neznáme") in s.index
    undefined = table[table["stav"] == "nedefinovaná"]
    assert undefined["hodnota"].isna().all() and undefined["dôvod"].notna().all()
    assert table.loc[table["stav"] != "nedefinovaná", "hodnota"].notna().all()
    assert set(table["stav"]) <= {"ok", "malá skupina", "nedefinovaná"}
