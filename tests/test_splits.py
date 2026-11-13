"""Testy rozdelenia na train / val / test (issue #6)."""

import pandas as pd
import pytest

from ekg.data import splits
from ekg.data.download import PTBXL_DIR


def make_manifest(rows):
    """rows: [(ecg_id, patient_id, split, usable, strict_norm), ...]"""
    df = pd.DataFrame(
        rows, columns=["ecg_id", "patient_id", "split", "usable_classifier", "strict_norm"]
    )
    return df.set_index("ecg_id")


@pytest.fixture
def manifest():
    return make_manifest(
        [
            (1, 10, "train", True, True),
            (2, 10, "train", True, False),  # ten istý pacient, ten istý split – OK
            (3, 11, "train", False, False),
            (4, 12, "val", True, True),
            (5, 13, "test", True, False),
        ]
    )


def test_assign_split():
    folds = pd.Series([1, 5, 8, 9, 10])
    assert list(splits.assign_split(folds)) == ["train", "train", "train", "val", "test"]


def test_assign_split_unknown_fold():
    with pytest.raises(ValueError, match="11"):
        splits.assign_split(pd.Series([1, 11]))


def test_no_leak_passes(manifest):
    splits.check_no_leak(manifest)


def test_patient_in_two_splits_is_detected(manifest):
    manifest.loc[4, "patient_id"] = 10  # pacient 10 je aj v train
    with pytest.raises(splits.SplitLeakError, match="1 pacientov"):
        splits.check_no_leak(manifest)


def test_duplicate_record_is_detected(manifest):
    with pytest.raises(splits.SplitLeakError, match="Duplicitné"):
        splits.check_no_leak(pd.concat([manifest, manifest.iloc[[0]]]))


def test_split_ids_and_subsets(manifest):
    assert list(splits.split_ids("train", manifest)) == [1, 2, 3]
    assert list(splits.split_ids("train", manifest, subset="classifier")) == [1, 2]
    assert list(splits.split_ids("train", manifest, subset="strict_norm")) == [1]
    assert list(splits.split_ids("val", manifest)) == [4]


def test_test_set_is_locked(manifest):
    with pytest.raises(splits.TestSetLockedError):
        splits.split_ids("test", manifest)
    assert list(splits.split_ids("test", manifest, allow_test=True)) == [5]


def test_unknown_split_or_subset(manifest):
    with pytest.raises(ValueError):
        splits.split_ids("validation", manifest)
    with pytest.raises(ValueError):
        splits.split_ids("train", manifest, subset="norm")


# --- skutočný PTB-XL a manifest v gite ---------------------------------------

real = pytest.mark.skipif(
    not (PTBXL_DIR / "ptbxl_database.csv").exists(), reason="PTB-XL nie je stiahnutý"
)


def test_committed_manifest_has_no_leak_and_expected_sizes():
    """Beží aj bez datasetu – manifest je v gite."""
    m = splits.load_manifest()
    sizes = m.groupby("split").size().to_dict()
    assert sizes == {"train": 17418, "val": 2183, "test": 2198}
    assert m.groupby("split")["patient_id"].nunique().to_dict() == {
        "train": 15023,
        "val": 1942,
        "test": 1904,
    }
    assert (m["split"] == "val").equals(m["strat_fold"] == 9)
    assert (m["split"] == "test").equals(m["strat_fold"] == 10)


@real
def test_committed_manifest_matches_data():
    assert splits.main(["--check"]) == 0
