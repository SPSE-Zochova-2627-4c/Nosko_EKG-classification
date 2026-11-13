import json

import numpy as np
import pytest

from ekg import contract as c
from ekg.paths import PROJECT_ROOT

EXAMPLE_JSON = PROJECT_ROOT / "docs" / "priklad_vystupu.json"


def make_signal(shape=(12, 1000)):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, size=shape)


# --- konštanty ---------------------------------------------------------------


def test_constants():
    assert c.LEADS == ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    assert c.LABELS == ("MI", "STTC", "CD", "HYP", "AFIB")
    assert (c.N_LEADS, c.N_SAMPLES, c.N_LABELS) == (12, 1000, 5)


# --- platný vstup ------------------------------------------------------------


def test_valid_signal_returns_float32():
    out = c.validate_signal(make_signal())
    assert out.shape == (12, 1000)
    assert out.dtype == np.float32


def test_from_wfdb_transposes():
    wfdb_signal = make_signal((1000, 12))
    out = c.from_wfdb(wfdb_signal)
    assert out.shape == (12, 1000)
    np.testing.assert_allclose(out[3], wfdb_signal[:, 3].astype(np.float32))


def test_from_wfdb_reorders_leads_by_name():
    wfdb_signal = make_signal((1000, 12))
    shuffled_names = list(reversed(c.LEADS))  # V6, V5, ..., I
    out = c.from_wfdb(wfdb_signal, sig_names=[n.lower() for n in shuffled_names])
    # zvod "I" bol v poslednom stĺpci
    np.testing.assert_allclose(out[0], wfdb_signal[:, 11].astype(np.float32))


def test_to_batch_shape():
    batch = c.to_batch([make_signal(), make_signal()])
    assert batch.shape == (2, 12, 1000)
    assert batch.dtype == np.float32


# --- neplatný vstup ----------------------------------------------------------


@pytest.mark.parametrize(
    "signal, message",
    [
        (make_signal((1000, 12)), "from_wfdb"),  # otočený záznam má nápovedu
        (make_signal((12, 500)), "Tvar"),
        (make_signal((11, 1000)), "Tvar"),
        (make_signal((1, 12, 1000)), "Tvar"),
        (np.full((12, 1000), "a"), "reálne"),
    ],
)
def test_invalid_shape_or_type(signal, message):
    with pytest.raises(c.InvalidInputError, match=message):
        c.validate_signal(signal)


def test_nan_reports_lead():
    signal = make_signal()
    signal[6, 100] = np.nan  # V1
    with pytest.raises(c.InvalidInputError, match="V1"):
        c.validate_signal(signal)


def test_wrong_sampling_rate():
    with pytest.raises(c.InvalidInputError, match="500 Hz"):
        c.validate_signal(make_signal(), sampling_rate=500)


def test_from_wfdb_missing_lead():
    names = list(c.LEADS)
    names[0] = "X"
    with pytest.raises(c.InvalidInputError, match="Chýbajú"):
        c.from_wfdb(make_signal((1000, 12)), sig_names=names)


# --- výstup ------------------------------------------------------------------


def test_example_json_roundtrip():
    """Príklad v dokumentácii musí zodpovedať kódu (načítanie a späť bez zmeny)."""
    data = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    result = c.PredictionResult.from_dict(data)
    assert json.loads(result.to_json()) == data
    assert list(data["labels"]) == list(c.LABELS)


def test_wrong_label_order_rejected():
    data = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    data["labels"] = dict(reversed(list(data["labels"].items())))
    with pytest.raises(ValueError, match="poradí"):
        c.PredictionResult.from_dict(data)


def test_probability_out_of_range_rejected():
    data = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    data["labels"]["MI"]["probability"] = 1.5
    with pytest.raises(ValueError, match="MI"):
        c.PredictionResult.from_dict(data)


def test_unknown_schema_version_rejected():
    data = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    data["schema_version"] = "0.9"
    with pytest.raises(ValueError, match="verzia"):
        c.PredictionResult.from_dict(data)


@pytest.mark.parametrize(
    "value, expected", [(0, "male"), (1, "female"), (None, None), (float("nan"), None)]
)
def test_sex_from_ptbxl(value, expected):
    assert c.sex_from_ptbxl(value) == expected
