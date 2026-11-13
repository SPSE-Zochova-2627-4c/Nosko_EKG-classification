"""Testy preprocessingu a cache signálov (issue #8)."""

import numpy as np
import pandas as pd
import pytest

from ekg import contract
from ekg.data import cache as cache_mod
from ekg.data import splits
from ekg.preprocessing import PARAMS_PATH, Preprocessor

T = np.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE  # 0 … 9,99 s


def ecg_like(rng, n=1, drift=0.0, scale=None):
    """Umelý signál: 10 Hz zložka + šum + voliteľný pomalý drift 0,1 Hz."""
    scale = np.linspace(0.1, 0.4, 12) if scale is None else scale
    x = np.sin(2 * np.pi * 10 * T)[None, None, :] * scale[None, :, None]
    x = x + rng.normal(0, 0.02, size=(n, 12, contract.N_SAMPLES))
    x = x + drift * np.sin(2 * np.pi * 0.1 * T)
    return x.astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- filter ------------------------------------------------------------------


def test_highpass_removes_drift_and_keeps_10hz(rng):
    p = Preprocessor()
    clean = ecg_like(rng)[0]
    drifted = clean + 1.0 * np.sin(2 * np.pi * 0.1 * T) + 0.5  # drift + DC posun
    filtered = p.filter(drifted)
    # mimo okrajov je výsledok takmer zhodný s čistým signálom bez driftu
    inner = slice(100, 900)
    assert np.abs(filtered[:, inner] - clean[:, inner]).max() < 0.05
    assert abs(filtered.mean()) < 0.01


def test_filter_single_and_batch_identical(rng):
    p = Preprocessor()
    batch = ecg_like(rng, n=3, drift=0.5)
    np.testing.assert_allclose(p.filter(batch)[1], p.filter(batch[1]), atol=1e-6)


def test_filter_can_be_disabled(rng):
    x = ecg_like(rng)[0]
    np.testing.assert_array_equal(Preprocessor(highpass_hz=None).filter(x), x)


# --- fit / transform ---------------------------------------------------------


def test_fit_gives_zero_mean_unit_std_per_lead(rng):
    data = ecg_like(rng, n=40, drift=0.3)
    p = Preprocessor().fit([data[:20], data[20:]])  # po dávkach
    out = p.transform(data)
    np.testing.assert_allclose(out.mean(axis=(0, 2)), 0, atol=1e-3)
    np.testing.assert_allclose(out.std(axis=(0, 2)), 1, atol=1e-3)
    assert out.dtype == np.float32 and out.shape == data.shape


def test_amplitude_differences_between_records_are_kept(rng):
    """Globálna normalizácia nezmaže, že jeden záznam má väčšiu amplitúdu (dôležité pre HYP)."""
    small = ecg_like(rng, scale=np.full(12, 0.1))
    big = ecg_like(rng, scale=np.full(12, 0.3))
    p = Preprocessor().fit([np.concatenate([small, big])])
    assert p.transform(big).std() > 2 * p.transform(small).std()


def test_params_depend_only_on_fit_data(rng):
    """Zmena validation/test dát nemení parametre – fit ich nikdy nevidí."""
    train = ecg_like(rng, n=10)
    p1 = Preprocessor().fit([train])
    p1.transform(ecg_like(rng, n=5) * 100)  # „iné“ validation dáta
    p2 = Preprocessor().fit([train])
    assert p1.mean == p2.mean and p1.std == p2.std


def test_zero_variance_lead_rejected(rng):
    data = ecg_like(rng, n=4)
    data[:, 7, :] = 0.0  # V2 plochý vo všetkých záznamoch
    with pytest.raises(ValueError, match="V2"):
        Preprocessor().fit([data])


def test_transform_requires_fit_and_valid_shape(rng):
    with pytest.raises(RuntimeError, match="fit"):
        Preprocessor().transform(ecg_like(rng)[0])
    p = Preprocessor().fit([ecg_like(rng, n=4)])
    with pytest.raises(contract.InvalidInputError):
        p.transform(np.zeros((1000, 12)))


def test_save_load_roundtrip(tmp_path, rng):
    data = ecg_like(rng, n=5)
    p = Preprocessor().fit([data], fitted_on={"split": "train"})
    p.save(tmp_path / "p.json")
    q = Preprocessor.load(tmp_path / "p.json")
    np.testing.assert_array_equal(p.transform(data), q.transform(data))
    assert q.fitted_on["split"] == "train"


def test_load_rejects_other_version(tmp_path, rng):
    p = Preprocessor().fit([ecg_like(rng, n=2)])
    p.version = "0.9"
    p.save(tmp_path / "p.json")
    with pytest.raises(ValueError, match="verziu"):
        Preprocessor.load(tmp_path / "p.json")


# --- cache -------------------------------------------------------------------


def test_cache_get_by_ids(tmp_path, rng):
    signals = ecg_like(rng, n=4)
    ids = np.array([7, 3, 11, 5])
    np.save(tmp_path / "s.npy", signals)
    np.save(tmp_path / "i.npy", ids)
    c = cache_mod.SignalCache(tmp_path / "s.npy", tmp_path / "i.npy")
    np.testing.assert_array_equal(c.get([11, 7]), signals[[2, 0]])
    with pytest.raises(KeyError, match="99"):
        c.get([99])
    manifest = pd.DataFrame(index=pd.Index(ids, name="ecg_id"))
    assert cache_mod.cache_matches(manifest, tmp_path / "i.npy", tmp_path / "s.npy")
    assert not cache_mod.cache_matches(manifest.iloc[::-1], tmp_path / "i.npy", tmp_path / "s.npy")


# --- uložené parametre v gite ------------------------------------------------


def test_committed_params_fitted_on_train_only():
    p = Preprocessor.load(PARAMS_PATH)
    assert p.fitted_on["split"] == "train"
    assert p.fitted_on["n_records"] == 17418
    assert p.highpass_hz == 0.5
    assert len(p.mean) == len(p.std) == 12 and min(p.std) > 0.05


real_cache = pytest.mark.skipif(
    not cache_mod.SIGNALS_PATH.exists(), reason="cache signálov nie je vytvorená"
)


@real_cache
def test_committed_params_match_refit_on_train():
    """Parametre v gite sa dajú presne zopakovať z train dát."""
    from ekg.preprocessing import _batches

    manifest = splits.load_manifest()
    c = cache_mod.SignalCache()
    refit = Preprocessor().fit(_batches(c, splits.split_ids("train", manifest)))
    committed = Preprocessor.load(PARAMS_PATH)
    np.testing.assert_allclose(refit.mean, committed.mean, atol=1e-9)
    np.testing.assert_allclose(refit.std, committed.std, rtol=1e-9)
