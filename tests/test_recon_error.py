"""Testy rekonštrukčnej chyby (issue #12)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ekg import contract
from ekg.models import recon_error as re
from ekg.models.autoencoder import AutoencoderConfig, ConvAutoencoder

SHAPE = (contract.N_LEADS, contract.N_SAMPLES)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(3, *SHAPE)


def test_identical_reconstruction_has_zero_error(x):
    assert torch.equal(re.record_error(x, x.clone()), torch.zeros(3))
    assert torch.equal(re.per_lead_error(x, x.clone()), torch.zeros(3, 12))
    assert torch.equal(re.error_over_time(x, x.clone()), torch.zeros(3, 1000))


def test_constant_offset_gives_offset_squared(x):
    """x̂ = x + c  =>  e = c² pre každý zvod aj záznam."""
    c = 0.5
    torch.testing.assert_close(re.record_error(x, x + c), torch.full((3,), c**2))
    torch.testing.assert_close(re.per_lead_error(x, x + c), torch.full((3, 12), c**2))


def test_offset_in_one_lead_mean_vs_max(x):
    """Chyba 2.0 iba vo V1 počas celého záznamu: zvod V1 = 4, priemer = 4/12, max = 4."""
    recon = x.clone()
    v1 = contract.LEADS.index("V1")
    recon[:, v1, :] += 2.0
    lead = re.per_lead_error(x, recon)
    expected = torch.zeros(3, 12)
    expected[:, v1] = 4.0
    torch.testing.assert_close(lead, expected)
    torch.testing.assert_close(re.record_error(x, recon, "mean"), torch.full((3,), 4.0 / 12))
    torch.testing.assert_close(re.record_error(x, recon, "max"), torch.full((3,), 4.0))


def test_short_deviation_localized_in_time(x):
    """Odchýlka 1.0 iba v 0,5 s (50 vzoriek) vo všetkých zvodoch: e = 50/1000 = 0.05."""
    recon = x.clone()
    recon[:, :, 300:350] += 1.0
    torch.testing.assert_close(re.record_error(x, recon), torch.full((3,), 0.05))
    over_time = re.error_over_time(x, recon)
    assert torch.all(over_time[:, 300:350] == 1.0)
    assert torch.all(over_time[:, :300] == 0) and torch.all(over_time[:, 350:] == 0)


def test_analytic_sine_example():
    """x = sin, x̂ = 0  =>  e = priemer sin² = 0.5 (celý počet periód)."""
    t = torch.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE
    x = torch.sin(2 * torch.pi * 1.0 * t).repeat(1, 12, 1)
    torch.testing.assert_close(re.record_error(x, torch.zeros_like(x)), torch.tensor([0.5]))


def test_error_is_mse_over_all_values(x):
    recon = x + 0.1 * torch.randn_like(x)
    expected = torch.stack([torch.nn.functional.mse_loss(recon[i], x[i]) for i in range(3)])
    torch.testing.assert_close(re.record_error(x, recon), expected)


def test_one_score_per_record_independent_of_batch(x):
    recon = x + 0.3 * torch.randn_like(x)
    together = re.record_error(x, recon)
    alone = torch.cat([re.record_error(x[i : i + 1], recon[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(together, alone)


def test_numpy_input_works(x):
    assert re.record_error(x.numpy(), (x + 1).numpy()).shape == (3,)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_invalid_values_rejected(x, bad):
    recon = x.clone()
    recon[0, 4, 10] = float(bad)
    with pytest.raises(ValueError, match="NaN"):
        re.record_error(x, recon)


def test_shape_mismatch_rejected(x):
    with pytest.raises(ValueError, match="Tvar"):
        re.record_error(x, x[:, :, :500])
    with pytest.raises(ValueError, match="Očakáva"):
        re.record_error(x[0], x[0])  # chýba rozmer dávky


def test_unknown_aggregation(x):
    with pytest.raises(ValueError, match="agregácia"):
        re.record_error(x, x, "median")


def test_lead_dict_order():
    d = re.lead_dict(np.arange(12))
    assert list(d) == list(contract.LEADS) and d["V6"] == 11.0


# --- compute_errors cez loader -----------------------------------------------


def fake_loader(n=10, batch_size=4):
    torch.manual_seed(1)
    items = [
        {"signal": torch.randn(*SHAPE), "target": torch.zeros(5), "ecg_id": 100 + i}
        for i in range(n)
    ]
    return DataLoader(items, batch_size=batch_size, shuffle=False)


def test_compute_errors_matches_direct_computation():
    torch.manual_seed(0)
    model = ConvAutoencoder(AutoencoderConfig(channels=(8, 16), latent_channels=4))
    model.train()
    loader = fake_loader()
    result = re.compute_errors(model, loader)
    assert model.training  # pôvodný režim sa obnoví
    assert list(result.ecg_ids) == list(range(100, 110))
    assert result.per_lead.shape == (10, 12)

    model.eval()
    with torch.no_grad():
        x = torch.stack([item["signal"] for item in loader.dataset])
        direct = re.per_lead_error(x, model(x)).numpy()
    np.testing.assert_allclose(result.per_lead, direct, rtol=1e-5)

    frame = result.to_frame()
    assert frame.index.name == "ecg_id" and "error_V1" in frame.columns
    np.testing.assert_allclose(frame["error"], direct.mean(axis=1), rtol=1e-5)


def test_compute_errors_is_deterministic():
    model = ConvAutoencoder(AutoencoderConfig(channels=(8, 16), latent_channels=4))
    a = re.compute_errors(model, fake_loader()).record()
    b = re.compute_errors(model, fake_loader(batch_size=3)).record()  # iná veľkosť dávky
    np.testing.assert_allclose(a, b, rtol=1e-5)
