"""Testy máp dôležitosti a chyby v čase (issue #27)."""

import numpy as np
import pytest
import torch
from torch import nn

from ekg import contract, explain, inference, paths
from ekg.models.classifier import ResNet1dClassifier
from tests.test_inference import _labels, _signal, _write_run

LABELS = list(contract.LABELS)
EXAMPLE = paths.PROJECT_ROOT / "examples" / "00017_lr"
REAL = (paths.CHECKPOINT_DIR / "clf_v1" / "thresholds.json").exists() and (
    paths.CHECKPOINT_DIR / "ae_v1" / "threshold.json"
).exists()


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    _write_run(root)
    return inference.Pipeline.load("ae", "clf", checkpoint_dir=root)


@pytest.fixture(scope="module")
def expl(pipe):
    return explain.explain(pipe, _signal(1), "z1")


class Linear(nn.Module):
    """logit_j = Σ w_j · x – pre lineárny model je IG presne x · w (pri základnej línii 0)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(5, 12, 1000))

    def forward(self, x):
        return torch.einsum("bct,jct->bj", x, self.w)


# --- metódy ---------------------------------------------------------------------------


def test_integrated_gradients_exact_for_linear_model():
    model = Linear()
    x = torch.randn(1, 12, 1000)
    attrs, delta = explain.integrated_gradients(model, x, steps=8)
    expected = (x[0] * model.w).detach().numpy()
    assert attrs.shape == (5, 12, 1000)
    assert np.allclose(attrs, expected, atol=1e-5)
    assert (delta < 1e-4).all()


def test_integrated_gradients_completeness_on_resnet():
    torch.manual_seed(0)
    model = ResNet1dClassifier().eval()
    x = torch.randn(1, 12, 1000)
    attrs, delta = explain.integrated_gradients(model, x)
    with torch.no_grad():
        diff = (model(x) - model(torch.zeros_like(x)))[0].numpy()
    assert np.allclose(attrs.sum(axis=(1, 2)), diff, rtol=0.05, atol=0.02)
    assert delta.shape == (5,)


def test_gradcam_shape_range_and_label_specific():
    torch.manual_seed(0)
    model = ResNet1dClassifier().eval()
    cams = explain.gradcam(model, torch.randn(1, 12, 1000))
    assert cams.shape == (5, contract.N_SAMPLES)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-9
    assert not np.allclose(cams[0], cams[1])  # iná mapa pre iný label


def test_upsample_keeps_values_at_centers():
    cam = np.array([0.0, 1.0, 0.0, 0.5])
    up = explain._upsample(cam, 8)  # stredy úsekov sú vzorky 1 a 2, 3 a 4, …
    assert up.shape == (8,) and up.max() <= 1 and up[0] == 0 and up[-1] == 0.5


# --- súhrny ---------------------------------------------------------------------------


def test_smooth_keeps_shape_and_constants():
    x = np.ones((12, 1000)) * 3
    assert np.allclose(explain.smooth(x), 3)
    spike = np.zeros(1000)
    spike[500] = 10
    s = explain.smooth(spike)
    assert s.shape == (1000,) and s.sum() == pytest.approx(10) and s.max() == pytest.approx(1)


def test_lead_shares_and_region():
    a = np.zeros((12, 1000))
    a[contract.LEADS.index("V1")] = 3.0
    a[contract.LEADS.index("II")] = 1.0
    a[contract.LEADS.index("I")] = -5.0  # záporná atribúcia sa nepočíta
    shares = explain.lead_shares(a)
    assert sum(shares.values()) == pytest.approx(1)
    assert shares["V1"] == pytest.approx(0.75) and shares["I"] == 0
    assert explain.region_share(a, explain.INFERIOR) == pytest.approx(0.25)
    flat = explain.lead_shares(-np.ones((12, 1000)))
    assert all(v == pytest.approx(1 / 12) for v in flat.values())
    assert explain.top_leads(a, 2) == "V1 75 %, II 25 %"


def test_heat_in_unit_range():
    rng = np.random.default_rng(0)
    h = explain.heat(rng.normal(size=(12, 1000)))
    assert h.shape == (12, 1000) and h.min() >= 0 and h.max() == pytest.approx(1)
    assert (explain.heat(np.zeros((12, 1000))) == 0).all()


# --- celý záznam ----------------------------------------------------------------------


def test_explain_matches_pipeline_output(pipe, expl):
    """Mapy sa počítajú z rovnakých modelov a preprocessingu ako výstup (#19)."""
    r = pipe.predict(_signal(1), "z1")
    assert expl.anomaly_score == pytest.approx(r.anomaly.anomaly_score, rel=1e-5)
    assert np.allclose(expl.recon_error.mean(axis=1), list(r.anomaly.error_per_lead.values()),
                       rtol=1e-5)  # fmt: skip
    for name in LABELS:
        assert expl.probabilities[name] == pytest.approx(r.labels[name].probability, abs=1e-6)
        assert expl.attributions[name].shape == (12, 1000)
        assert expl.gradcam[name].shape == (1000,)
    filtered, _ = pipe.reconstruct(_signal(1))
    assert np.allclose(expl.signal_mv, filtered, atol=1e-5)


def test_explain_restores_threads(pipe):
    before = torch.get_num_threads()
    explain.explain(pipe, _signal(2), steps=4)
    assert torch.get_num_threads() == before


def test_explain_rejects_invalid_input(pipe):
    with pytest.raises(contract.InvalidInputError):
        explain.explain(pipe, np.zeros((12, 999), np.float32))


def test_view_and_plot(expl):
    import matplotlib.pyplot as plt

    for target in (explain.MODEL_A, "MI"):
        v = explain.view(expl, target)
        assert v["heat"].shape == (12, 1000) and v["strip"].shape == (1000,)
        fig = explain.plot_explanation(expl, target)
        assert len(fig.axes) == 1 + 12 + 1  # pás, 12 zvodov, farebná škála
        assert any("nie lekárske vysvetlenie" in t.get_text() for t in fig.texts)
        plt.close(fig)
    with pytest.raises(ValueError, match="Neznámy"):
        explain.view(expl, "NORM")


def test_default_target(pipe):
    r = pipe.predict(_signal(), "a")
    r.labels = _labels("CD", "AFIB")
    r.labels["AFIB"].probability = 0.95
    assert explain.default_target(r) == "AFIB"
    r.labels = _labels()
    assert explain.default_target(r) == explain.MODEL_A


# --- CLI ------------------------------------------------------------------------------


def test_cli_refuses_test_set(monkeypatch, capsys):
    from ekg.data import splits

    test_id = int(splits.load_manifest().query("split == 'test'").index[0])
    monkeypatch.setattr(inference.Pipeline, "load", classmethod(lambda cls, *a, **k: None))
    called = []
    monkeypatch.setattr(explain, "explain", lambda *a, **k: called.append(1))
    assert explain.main(["--ecg-id", str(test_id)]) == 1  # TestSetLockedError je OSError
    assert "Chyba" in capsys.readouterr().err and not called


@pytest.mark.skipif(not REAL, reason="checkpointy ae_v1 / clf_v1 nie sú k dispozícii")
def test_cli_example_file(tmp_path, capsys):
    assert explain.main([str(EXAMPLE), "--label", "AFIB", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "zaznam_00017_lr_AFIB.png").exists()
    out = capsys.readouterr().out
    assert "P = 0.998" in out and "Úplnosť IG" in out
