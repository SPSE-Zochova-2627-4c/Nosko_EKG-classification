"""Testy tréningu autoencodera (issue #11) na malých umelých dátach."""

import copy

import pandas as pd
import pytest
import torch

from ekg import contract
from ekg.models.autoencoder import ConvAutoencoder
from ekg.training.common import EarlyStopping, load_yaml
from ekg.training.train_autoencoder import DEFAULT_CONFIG, plot_curves, train

SHAPE = (contract.N_LEADS, contract.N_SAMPLES)
CPU = torch.device("cpu")


def fake_dataset(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 10, contract.N_SAMPLES)
    items = []
    for i in range(n):
        f = 1 + torch.rand(1, generator=g).item()
        x = torch.sin(2 * torch.pi * f * t).repeat(contract.N_LEADS, 1)
        x = x + 0.05 * torch.randn(*SHAPE, generator=g)
        items.append({"signal": x, "target": torch.zeros(5), "ecg_id": i})
    return items


@pytest.fixture
def cfg():
    c = load_yaml(DEFAULT_CONFIG)
    c["model"] = {"channels": [8, 16], "latent_channels": 4, "kernel_size": 5, "dropout": 0.1}
    c["data"]["batch_size"] = 4
    c["training"].update(max_epochs=4, early_stopping_patience=10)
    return c


@pytest.fixture
def data():
    return fake_dataset(12, 0), fake_dataset(4, 1)


def test_default_config_is_valid():
    from ekg.models.autoencoder import AutoencoderConfig

    c = load_yaml(DEFAULT_CONFIG)
    AutoencoderConfig(**c["model"])
    assert c["data"]["train"] == "train/strict_norm"
    assert c["data"]["val"] == "val/strict_norm"


def test_training_reduces_loss_and_saves_outputs(cfg, data, tmp_path):
    summary = train(cfg, *data, tmp_path / "ckpt", tmp_path / "out", device=CPU, verbose=False)
    metrics = pd.read_csv(tmp_path / "out" / "metrics.csv")
    assert list(metrics["epoch"]) == [1, 2, 3, 4]
    assert metrics["train_loss"].iloc[-1] < metrics["train_loss"].iloc[0]
    assert summary["best_val_loss"] == pytest.approx(metrics["val_loss"].min(), abs=1e-6)
    model, ckpt = ConvAutoencoder.load(tmp_path / "ckpt" / "best.pt")
    assert ckpt["extra"]["epoch"] == summary["best_epoch"]
    plot_curves(tmp_path / "out" / "metrics.csv", tmp_path / "out" / "krivky.png")
    assert (tmp_path / "out" / "krivky.png").stat().st_size > 0


def test_resume_gives_same_result_as_uninterrupted(cfg, data, tmp_path):
    """Tréning prerušený po 2. epoche a obnovený = tréning 4 epochy bez prerušenia."""
    full = tmp_path / "full"
    train(cfg, *data, full / "c", full / "o", device=CPU, verbose=False)

    part = tmp_path / "part"
    r = train(cfg, *data, part / "c", part / "o", device=CPU, verbose=False, stop_after_epoch=2)
    assert r == {"interrupted": True, "epoch": 2}
    train(cfg, *data, part / "c", part / "o", resume=True, device=CPU, verbose=False)

    a = torch.load(full / "c" / "last.pt", weights_only=True)["model"]
    b = torch.load(part / "c" / "last.pt", weights_only=True)["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
    ma = pd.read_csv(full / "o" / "metrics.csv").drop(columns="seconds")
    mb = pd.read_csv(part / "o" / "metrics.csv").drop(columns="seconds")
    pd.testing.assert_frame_equal(ma, mb)


def test_resume_without_checkpoint_fails(cfg, data, tmp_path):
    with pytest.raises(FileNotFoundError, match="last.pt"):
        train(cfg, *data, tmp_path / "c", tmp_path / "o", resume=True, device=CPU, verbose=False)


def test_early_stopping_stops(cfg, data, tmp_path):
    c = copy.deepcopy(cfg)
    c["optimizer"]["lr"] = 0.0  # nič sa nezlepší
    c["training"].update(max_epochs=20, early_stopping_patience=2)
    summary = train(c, *data, tmp_path / "c", tmp_path / "o", device=CPU, verbose=False)
    assert summary["stopped_early"] and summary["epochs_run"] == 3  # 1 najlepšia + 2 bez zlepšenia


def test_early_stopping_logic():
    es = EarlyStopping(patience=2, min_delta=0.1)
    assert es.step(1.0, 1)
    assert not es.step(0.95, 2)  # zlepšenie menšie ako min_delta
    assert not es.should_stop
    assert not es.step(0.99, 3)
    assert es.should_stop and es.best_epoch == 1
