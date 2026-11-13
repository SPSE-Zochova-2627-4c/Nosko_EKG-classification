"""Testy tréningu klasifikátora (issue #16) na malých umelých dátach."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ekg import contract
from ekg.models.classifier import ClassifierConfig, ResNet1dClassifier
from ekg.training import train_classifier as tc
from ekg.training.common import load_yaml

SHAPE = (contract.N_LEADS, contract.N_SAMPLES)
CPU = torch.device("cpu")


def fake_dataset(n, seed):
    """Label j je pozitívny, keď má zvod j sínus s vyššou frekvenciou – dá sa naučiť."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 10, contract.N_SAMPLES)
    items = []
    for i in range(n):
        y = (torch.rand(contract.N_LABELS, generator=g) < 0.4).float()
        y[i % contract.N_LABELS] = 1.0  # každý label má aspoň jeden pozitívny príklad
        x = 0.05 * torch.randn(*SHAPE, generator=g)
        for j in range(contract.N_LABELS):
            x[j] += torch.sin(2 * torch.pi * (3.0 if y[j] else 1.0) * t)
        items.append({"signal": x, "target": y, "ecg_id": i})
    return items


@pytest.fixture
def cfg():
    c = load_yaml(tc.DEFAULT_CONFIG)
    c["model"] = {
        "stem_channels": 8,
        "stem_kernel": 7,
        "channels": [8, 16],
        "blocks": [1, 1],
        "kernel_size": 5,
        "dropout": 0.1,
    }
    c["data"]["batch_size"] = 8
    c["training"].update(max_epochs=4, early_stopping_patience=10, amp="none")
    return c


@pytest.fixture
def data():
    return fake_dataset(40, 0), fake_dataset(20, 1)


def test_default_config_is_valid():
    c = load_yaml(tc.DEFAULT_CONFIG)
    ClassifierConfig(**c["model"])
    assert c["data"]["train"] == "train/classifier"
    assert c["data"]["val"] == "val/classifier"
    assert c["imbalance"]["pos_weight"] in ("train", "none")


def test_pos_weight_is_neg_over_pos():
    y = np.array([[1, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0]], np.float32)
    # pozitívne: MI 1, STTC 1, CD 3, HYP 1, AFIB 1 zo 4
    w = tc.compute_pos_weight(y)
    torch.testing.assert_close(w, torch.tensor([3.0, 3.0, 1 / 3, 3.0, 3.0]))
    counts = tc.label_counts(y)
    assert counts["CD"] == {"pos": 3, "neg": 1, "share": 0.75}


def test_pos_weight_requires_positive_examples():
    y = np.zeros((3, 5), np.float32)
    y[:, :4] = 1
    with pytest.raises(ValueError, match="AFIB"):
        tc.compute_pos_weight(y)


def test_metrics_perfect_and_random():
    y = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]] * 10, np.float32)
    perfect = tc.classification_metrics(y, y * 0.9 + 0.05)
    assert perfect["val_macro_auroc"] == pytest.approx(1.0)
    assert perfect["auprc_MI"] == pytest.approx(1.0)
    flipped = tc.classification_metrics(y, 1 - y)
    assert flipped["val_macro_auroc"] == pytest.approx(0.0)


def test_bce_matches_formula():
    p = np.array([[0.9, 0.2]])
    y = np.array([[1.0, 0.0]])
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert tc.bce(p, y) == pytest.approx(expected)


def test_test_split_is_refused(cfg, data, tmp_path):
    c = copy.deepcopy(cfg)
    c["data"]["val"] = "test/classifier"
    with pytest.raises(ValueError, match="Test"):
        tc.train(c, *data, tmp_path / "c", tmp_path / "o", device=CPU, verbose=False)


@pytest.mark.parametrize("mode", ["train", "none"])
def test_training_learns_and_saves_outputs(cfg, data, tmp_path, mode):
    c = copy.deepcopy(cfg)
    c["imbalance"]["pos_weight"] = mode
    c["training"]["max_epochs"] = 6
    summary = tc.train(c, *data, tmp_path / "ckpt", tmp_path / "out", device=CPU, verbose=False)

    metrics = pd.read_csv(tmp_path / "out" / "metrics.csv")
    assert list(metrics.columns) == tc.FIELDS
    assert list(metrics["epoch"]) == [1, 2, 3, 4, 5, 6]
    assert metrics["train_loss"].iloc[-1] < metrics["train_loss"].iloc[0]
    best = summary["best_val_macro_auroc"]
    assert best == pytest.approx(metrics["val_macro_auroc"].max(), abs=1e-4)
    assert best > 0.8  # úloha je ľahká

    model, ckpt = ResNet1dClassifier.load(tmp_path / "ckpt" / "best.pt")
    assert ckpt["extra"]["epoch"] == summary["best_epoch"]
    assert ckpt["extra"]["amp"] == "none"
    assert (ckpt["extra"]["pos_weight"] is None) == (mode == "none")

    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["pos_weight_mode"] == mode
    assert set(report["train_label_counts"]) == set(contract.LABELS)
    assert report["best_epoch"] == summary["best_epoch"]
    assert report["best_val"]["val_macro_auroc"] == pytest.approx(best, abs=1e-4)

    tc.plot_curves(tmp_path / "out" / "metrics.csv", tmp_path / "out" / "krivky.png")
    assert (tmp_path / "out" / "krivky.png").stat().st_size > 0


def test_resume_gives_same_result_as_uninterrupted(cfg, data, tmp_path):
    """Tréning prerušený po 2. epoche a obnovený = tréning 4 epochy bez prerušenia (CPU)."""
    full = tmp_path / "full"
    tc.train(cfg, *data, full / "c", full / "o", device=CPU, verbose=False)

    part = tmp_path / "part"
    r = tc.train(cfg, *data, part / "c", part / "o", device=CPU, verbose=False, stop_after_epoch=2)
    assert r == {"interrupted": True, "epoch": 2}
    tc.train(cfg, *data, part / "c", part / "o", resume=True, device=CPU, verbose=False)

    a = torch.load(full / "c" / "last.pt", weights_only=True)["model"]
    b = torch.load(part / "c" / "last.pt", weights_only=True)["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
    ma = pd.read_csv(full / "o" / "metrics.csv").drop(columns="seconds")
    mb = pd.read_csv(part / "o" / "metrics.csv").drop(columns="seconds")
    pd.testing.assert_frame_equal(ma, mb)


def test_early_stopping_on_auroc(cfg, data, tmp_path):
    c = copy.deepcopy(cfg)
    c["optimizer"]["lr"] = 0.0  # nič sa nezlepší
    c["training"].update(max_epochs=20, early_stopping_patience=2)
    summary = tc.train(c, *data, tmp_path / "c", tmp_path / "o", device=CPU, verbose=False)
    assert summary["stopped_early"] and summary["epochs_run"] == 3


def test_overrides_and_compare(cfg, data, tmp_path, monkeypatch, capsys):
    args = tc.argparse.Namespace(run_name="x", pos_weight="none", amp="none", seed=7, max_epochs=2)
    c = tc.apply_overrides(copy.deepcopy(cfg), args)
    assert (c["run_name"], c["imbalance"]["pos_weight"], c["seed"]) == ("x", "none", 7)
    assert c["training"]["max_epochs"] == 2

    monkeypatch.setattr(tc.paths, "OUTPUT_DIR", tmp_path)
    tc.train(c, *data, tmp_path / "ckpt", tmp_path / "x", device=CPU, verbose=False)
    assert tc.compare_main(["x", "chyba"]) == 0
    out = capsys.readouterr().out
    assert "x | none | none" in out and "chyba: chýba" in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nie je dostupné")
def test_training_on_gpu_with_bf16(cfg, data, tmp_path):
    c = copy.deepcopy(cfg)
    c["training"]["amp"] = "bf16"
    gpu = torch.device("cuda")
    summary = tc.train(c, *data, tmp_path / "c", tmp_path / "o", device=gpu, verbose=False)
    assert summary["report"]["amp"] == "bf16"
    assert np.isfinite(summary["best_val_macro_auroc"])
