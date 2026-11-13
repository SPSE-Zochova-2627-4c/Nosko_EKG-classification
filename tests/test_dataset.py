"""Testy Datasetu, DataLoadera a kontrolného priechodu (issue #9) na umelom PTB-XL."""

import numpy as np
import pandas as pd
import pytest
import torch

from ekg import contract
from ekg.data import ptbxl, splits
from ekg.data.cache import SignalCache, build_cache
from ekg.data.check_pipeline import check_loader
from ekg.data.dataset import make_dataset, make_loader
from ekg.preprocessing import Preprocessor
from tests.test_ptbxl import write_fake_ptbxl

N = 12  # záznamov v umelom datasete


@pytest.fixture
def env(tmp_path):
    """Umelý PTB-XL + manifest + cache + preprocessor."""
    root = tmp_path / "ptb-xl"
    write_fake_ptbxl(root, n_records=N)
    db = ptbxl.load_database(root)
    rng = np.random.default_rng(1)
    ids = db.index.to_numpy()
    folds = np.array([1] * 8 + [9] * 2 + [10] * 2)
    manifest = pd.DataFrame(
        {
            "patient_id": ids + 1000,
            "strat_fold": folds,
            "split": splits.assign_split(pd.Series(folds, index=ids)).to_numpy(),
            "filename_lr": db["filename_lr"].to_numpy(),
            **{label: rng.integers(0, 2, N) for label in contract.LABELS},
            "usable_classifier": True,
            "strict_norm": [True, False] * 6,
        },
        index=pd.Index(ids, name="ecg_id"),
    )
    manifest.loc[manifest["strict_norm"], list(contract.LABELS)] = 0
    build_cache(manifest, root, tmp_path / "s.npy", tmp_path / "i.npy")
    cache = SignalCache(tmp_path / "s.npy", tmp_path / "i.npy")
    train = splits.split_ids("train", manifest)
    prep = Preprocessor().fit([cache.get(train)])
    return {"root": root, "manifest": manifest, "cache": cache, "preprocessor": prep}


def shared(env):
    return {k: env[k] for k in ("manifest", "cache", "preprocessor")}


def test_item_shape_and_targets(env):
    ds = make_dataset("train", "classifier", **shared(env))
    item = ds[3]
    assert item["signal"].shape == (12, 1000) and item["signal"].dtype == torch.float32
    expected = env["manifest"].loc[item["ecg_id"], list(contract.LABELS)].to_numpy(np.float32)
    np.testing.assert_array_equal(item["target"].numpy(), expected)


def test_batch_shapes(env):
    batch = next(iter(make_loader("train", batch_size=4, **shared(env))))
    assert batch["signal"].shape == (4, 12, 1000)
    assert batch["target"].shape == (4, 5)
    assert batch["ecg_id"].dtype == torch.int64


def test_preload_and_lazy_give_same_signals(env):
    a = make_dataset("val", preload=True, **shared(env))
    b = make_dataset("val", preload=False, **shared(env))
    for i in range(len(a)):
        torch.testing.assert_close(a[i]["signal"], b[i]["signal"])


def test_strict_norm_subset(env):
    ds = make_dataset("train", "strict_norm", **shared(env))
    assert len(ds) == 4 and not ds.targets.any()


def test_test_set_locked(env):
    with pytest.raises(splits.TestSetLockedError):
        make_dataset("test", **shared(env))
    assert len(make_dataset("test", allow_test=True, **shared(env))) == 2


def test_val_cannot_be_shuffled(env):
    with pytest.raises(ValueError, match="nemiešajú"):
        make_loader("val", shuffle=True, **shared(env))


def test_same_seed_same_order_different_seed_different_order(env):
    ds = make_dataset("train", **shared(env))

    def order(seed):
        return torch.cat(
            [b["ecg_id"] for b in make_loader("train", batch_size=3, seed=seed, dataset=ds)]
        )

    assert torch.equal(order(1), order(1))
    assert not torch.equal(order(1), order(2))


def test_check_loader_passes_on_correct_pipeline(env):
    loader = make_loader("train", batch_size=4, **shared(env))
    rng = np.random.default_rng(0)
    assert check_loader("t", loader, env["manifest"], env["preprocessor"], rng, env["root"]) == []


def test_check_loader_detects_shuffled_cache(env, tmp_path):
    """Ak by cache mala záznamy v inom poradí ako ids, kontrola to musí odhaliť."""
    signals = np.load(tmp_path / "s.npy")
    np.save(tmp_path / "s_bad.npy", signals[::-1].copy())
    bad_cache = SignalCache(tmp_path / "s_bad.npy", tmp_path / "i.npy")
    loader = make_loader(
        "train",
        manifest=env["manifest"],
        cache=bad_cache,
        preprocessor=env["preprocessor"],
    )
    problems = check_loader(
        "t", loader, env["manifest"], env["preprocessor"], np.random.default_rng(0), env["root"]
    )
    assert any("WFDB" in p for p in problems)


def test_check_loader_detects_wrong_targets(env):
    loader = make_loader("train", **shared(env))
    wrong = env["manifest"].copy()
    wrong["MI"] = 1 - wrong["MI"]
    problems = check_loader(
        "t", loader, wrong, env["preprocessor"], np.random.default_rng(0), env["root"]
    )
    assert any("manifestom" in p for p in problems)
