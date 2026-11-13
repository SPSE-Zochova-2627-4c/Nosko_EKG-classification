"""Testy Modelu A – 1D konvolučného autoencodera (issue #10)."""

import pytest
import torch
from torch import nn

from ekg import contract
from ekg.models import checkpoint
from ekg.models.autoencoder import AutoencoderConfig, ConvAutoencoder

SHAPE = (contract.N_LEADS, contract.N_SAMPLES)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ConvAutoencoder()


def test_output_shape_equals_input(model):
    x = torch.randn(3, *SHAPE)
    assert model(x).shape == x.shape


def test_latent_shape_and_compression(model):
    z = model.encode(torch.randn(2, *SHAPE))
    assert z.shape == (2, 16, 125)
    assert model.config.compression == pytest.approx(6.0)


@pytest.mark.parametrize("channels, latent", [((16, 32), 8), ((32, 64, 128), 4)])
def test_other_configs_keep_shape(channels, latent):
    m = ConvAutoencoder(AutoencoderConfig(channels=channels, latent_channels=latent))
    assert m(torch.randn(2, *SHAPE)).shape == (2, *SHAPE)


def test_invalid_config():
    with pytest.raises(ValueError, match="nepárny"):
        AutoencoderConfig(kernel_size=6)
    with pytest.raises(ValueError, match="deliť"):
        AutoencoderConfig(channels=(8, 8, 8, 8))  # 1000 / 16 nie je celé číslo


def test_backward_reaches_all_parameters(model):
    x = torch.randn(4, *SHAPE)
    nn.functional.mse_loss(model(x), x).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_can_overfit_small_batch(model):
    """Ak sa model nevie naučiť ani 4 záznamy, je v architektúre chyba."""
    torch.manual_seed(1)
    t = torch.linspace(0, 10, contract.N_SAMPLES)
    x = torch.stack(
        [torch.sin(2 * torch.pi * f * t).repeat(contract.N_LEADS, 1) for f in (1.0, 1.3, 1.7, 2.1)]
    )
    x = x + 0.05 * torch.randn_like(x)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    model.eval()
    with torch.no_grad():
        final = nn.functional.mse_loss(model(x), x).item()
    assert final < 0.1 * first, (first, final)


def test_checkpoint_roundtrip_gives_identical_output(model, tmp_path):
    x = torch.randn(2, *SHAPE)
    model.train()
    model(x)  # zmení BatchNorm štatistiky, aby sa overilo, že sa uložia tiež
    model.eval()
    with torch.no_grad():
        expected = model(x)
    model.save(tmp_path / "ae.pt", epoch=3, val_loss=0.12)
    loaded, ckpt = ConvAutoencoder.load(tmp_path / "ae.pt")
    assert not loaded.training
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)
    assert ckpt["extra"] == {"epoch": 3, "val_loss": 0.12}
    assert ckpt["versions"]["preprocessing"] == "1.0"
    assert loaded.config == model.config


def test_checkpoint_keeps_custom_config(tmp_path):
    cfg = AutoencoderConfig(channels=(16, 32), latent_channels=8, kernel_size=5)
    ConvAutoencoder(cfg).save(tmp_path / "ae.pt")
    loaded, _ = ConvAutoencoder.load(tmp_path / "ae.pt")
    assert loaded.config == cfg


def test_checkpoint_rejects_other_model_type(model, tmp_path):
    checkpoint.save(tmp_path / "x.pt", "classifier", {}, model)
    with pytest.raises(ValueError, match="classifier"):
        ConvAutoencoder.load(tmp_path / "x.pt")


def test_checkpoint_rejects_other_contract(model, tmp_path):
    model.save(tmp_path / "ae.pt")
    ckpt = torch.load(tmp_path / "ae.pt", weights_only=True)
    ckpt["contract"]["leads"] = list(reversed(ckpt["contract"]["leads"]))
    torch.save(ckpt, tmp_path / "zle.pt")
    with pytest.raises(ValueError, match="kontraktom"):
        ConvAutoencoder.load(tmp_path / "zle.pt")
