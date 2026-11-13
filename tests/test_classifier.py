"""Testy Modelu B – 1D ResNet multi-label klasifikátora (issue #15)."""

import pytest
import torch

from ekg import contract
from ekg.models.autoencoder import ConvAutoencoder
from ekg.models.classifier import (
    ClassifierConfig,
    ResNet1dClassifier,
    loss_fn,
)

SHAPE = (contract.N_LEADS, contract.N_SAMPLES)
SMALL = ClassifierConfig(stem_channels=16, channels=(16, 32), blocks=(1, 1), dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ResNet1dClassifier()


def test_output_is_five_logits(model):
    assert model(torch.randn(4, *SHAPE)).shape == (4, contract.N_LABELS)


def test_predict_proba_in_unit_interval_and_independent(model):
    p = model.predict_proba(torch.randn(8, *SHAPE))
    assert p.shape == (8, 5)
    assert torch.all((p >= 0) & (p <= 1))
    # nezávislé sigmoidy: súčet NIE je vynútene 1 (to by bol softmax)
    assert not torch.allclose(p.sum(dim=1), torch.ones(8), atol=1e-3)


def test_predict_proba_restores_mode(model):
    model.train()
    model.predict_proba(torch.randn(2, *SHAPE))
    assert model.training


def test_multiple_positive_labels_possible():
    """Sieť sa vie naučiť záznam s MI aj CD naraz (obe P > 0.5)."""
    torch.manual_seed(0)
    m = ResNet1dClassifier(SMALL)
    x = torch.randn(4, *SHAPE)
    y = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn()(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    p = m.predict_proba(x)
    assert loss.item() < 0.1 * first
    assert torch.equal((p > 0.5).float(), y)  # vrátane riadku s 2 a s 5 pozitívnymi labelmi


def test_backward_reaches_all_parameters(model):
    x = torch.randn(4, *SHAPE)
    y = torch.randint(0, 2, (4, 5)).float()
    loss_fn()(model(x), y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_bce_matches_manual_formula(model):
    logits = torch.tensor([[2.0, -1.0, 0.0, 3.0, -2.0]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])
    p = torch.sigmoid(logits)
    manual = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
    torch.testing.assert_close(loss_fn()(logits, y), manual)


def test_pos_weight_increases_loss_for_missed_positive():
    logits = torch.tensor([[-2.0, 0.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    weighted = loss_fn(torch.tensor([5.0, 1, 1, 1, 1]))(logits, y)
    assert weighted > loss_fn()(logits, y)


def test_features_for_gradcam(model):
    assert model.features(torch.randn(2, *SHAPE)).shape == (2, 256, 63)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"channels": (64, 128), "blocks": (2,)}, "rovnakú dĺžku"),
        ({"kernel_size": 8}, "nepárne"),
        ({"n_labels": 4}, "labelov"),
    ],
)
def test_invalid_config(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        ClassifierConfig(**kwargs)


def test_checkpoint_roundtrip_identical(model, tmp_path):
    x = torch.randn(3, *SHAPE)
    model.train()
    model(x)  # zmení BatchNorm štatistiky
    model.eval()
    with torch.no_grad():
        expected = model(x)
    model.save(tmp_path / "clf.pt", epoch=1)
    loaded, ckpt = ResNet1dClassifier.load(tmp_path / "clf.pt")
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)
    assert ckpt["contract"]["labels"] == list(contract.LABELS)
    assert loaded.config == model.config


def test_checkpoint_types_not_interchangeable(model, tmp_path):
    model.save(tmp_path / "clf.pt")
    ConvAutoencoder().save(tmp_path / "ae.pt")
    with pytest.raises(ValueError, match="resnet1d_classifier"):
        ConvAutoencoder.load(tmp_path / "clf.pt")
    with pytest.raises(ValueError, match="conv_autoencoder"):
        ResNet1dClassifier.load(tmp_path / "ae.pt")
