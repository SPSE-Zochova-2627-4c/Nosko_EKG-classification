"""Testy spoločnej inference oboch modelov (issue #19)."""

import hashlib
import json

import jsonschema
import numpy as np
import pytest
import torch

from ekg import contract, inference, paths
from ekg.contract import LabelResult
from ekg.data import splits
from ekg.label_thresholds import THRESHOLDS_VERSION
from ekg.models.autoencoder import ConvAutoencoder
from ekg.models.classifier import ResNet1dClassifier

SCHEMA = json.loads(
    (paths.PROJECT_ROOT / "docs" / "schema" / f"vystup_v{contract.SCHEMA_VERSION}.schema.json")
    .read_text(encoding="utf-8")
)  # fmt: skip
CLF_THRESHOLDS = [0.5, 0.6, 0.7, 0.4, 0.9]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root, ae_threshold=0.5, wrong_sha=False):
    """Malé netrénované modely + prahy v tvare, aký ukladajú #13 a #17."""
    torch.manual_seed(0)
    (root / "ae").mkdir(parents=True)
    (root / "clf").mkdir(parents=True)
    ConvAutoencoder().save(root / "ae" / "best.pt", epoch=1)
    ResNet1dClassifier().save(root / "clf" / "best.pt", epoch=2)
    ae_sha = "0" * 64 if wrong_sha else _sha(root / "ae" / "best.pt")
    (root / "ae" / "threshold.json").write_text(
        json.dumps({"checkpoint_sha256": ae_sha, "threshold": ae_threshold, "aggregation": "mean"}),
        encoding="utf-8",
    )
    labels = {n: {"threshold": t} for n, t in zip(contract.LABELS, CLF_THRESHOLDS, strict=True)}
    (root / "clf" / "thresholds.json").write_text(
        json.dumps(
            {
                "version": THRESHOLDS_VERSION,
                "checkpoint_sha256": _sha(root / "clf" / "best.pt"),
                "label_order": list(contract.LABELS),
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    _write_run(root)
    return inference.Pipeline.load("ae", "clf", checkpoint_dir=root)


def _signal(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE
    return (np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.normal(size=(12, 1000))).astype(np.float32)


# --- pravidlo súladu ------------------------------------------------------------


def _labels(*positive):
    return {
        n: LabelResult(
            probability=0.9 if n in positive else 0.1, threshold=0.5, positive=n in positive
        )
        for n in contract.LABELS
    }


def test_consistency_rule_all_cases():
    assert inference.consistency_rule(True, _labels("MI")).models_agree
    assert inference.consistency_rule(False, _labels()).models_agree
    assert inference.consistency_rule(True, _labels("MI")).note is None

    c = inference.consistency_rule(False, _labels("MI", "AFIB"))
    assert not c.models_agree and "MI (P = 0.90)" in c.note and "AFIB" in c.note
    assert "STTC" not in c.note

    c = inference.consistency_rule(True, _labels())
    assert not c.models_agree and "neoznačil žiadnu" in c.note


# --- pipeline -------------------------------------------------------------------


def test_output_complete_and_valid(pipe):
    r = pipe.predict(_signal(), "zaznam-1")
    assert r.record_id == "zaznam-1" and r.schema_version == contract.SCHEMA_VERSION
    a = r.anomaly
    assert a.anomaly_score == a.reconstruction_error > 0
    assert a.is_anomaly == (a.anomaly_score >= a.threshold) and a.threshold == 0.5
    assert list(a.error_per_lead) == list(contract.LEADS)
    assert np.mean(list(a.error_per_lead.values())) == pytest.approx(a.reconstruction_error)
    assert list(r.labels) == list(contract.LABELS)
    for (name, lab), th in zip(r.labels.items(), CLF_THRESHOLDS, strict=True):
        assert 0 <= lab.probability <= 1 and lab.threshold == th, name
        assert lab.positive == (lab.probability >= th)
    assert r.consistency == inference.consistency_rule(a.is_anomaly, r.labels)
    assert r.model_versions["autoencoder"].startswith("ae (epocha 1, sha256 ")
    assert r.model_versions["classifier"].startswith("clf (epocha 2, sha256 ")
    jsonschema.validate(
        {"schema_version": contract.SCHEMA_VERSION, "results": [r.to_dict()]}, SCHEMA
    )


def test_matches_models_directly(pipe):
    """Pipeline = preprocessing z train + modely; nič iné do výsledku nevstupuje."""
    x = _signal(3)
    r = pipe.predict(x, "a")
    xp = torch.from_numpy(pipe.preprocessor.transform(x[None]))
    with torch.no_grad():
        err = ((pipe.autoencoder(xp) - xp) ** 2).mean().item()
        probs = torch.sigmoid(pipe.classifier(xp))[0].numpy()
    assert r.anomaly.reconstruction_error == pytest.approx(err, rel=1e-5)
    got = [r.labels[n].probability for n in contract.LABELS]
    assert np.allclose(got, probs, atol=1e-6)


def test_reproducible_and_independent_of_batch(pipe):
    signals = [_signal(i) for i in range(4)]
    batch = pipe.predict_batch(signals, ["a", "b", "c", "d"])
    again = pipe.predict_batch(signals, ["a", "b", "c", "d"])
    assert batch == again
    single = pipe.predict(signals[2], "c")
    assert single == batch[2]  # bitovo rovnaké – nezáleží na ostatných záznamoch v dávke


def test_threads_restored(pipe):
    before = torch.get_num_threads()
    pipe.predict(_signal(), "a")
    assert torch.get_num_threads() == before


def test_classifier_runs_even_when_not_anomaly(tmp_path):
    _write_run(tmp_path, ae_threshold=1e9)  # Model A nikdy nehlási anomáliu
    pipe = inference.Pipeline.load("ae", "clf", checkpoint_dir=tmp_path)
    r = pipe.predict(_signal(), "a")
    assert not r.anomaly.is_anomaly
    assert len(r.labels) == 5 and all(lab.probability > 0 for lab in r.labels.values())


def test_quality_warning_does_not_block(pipe):
    x = _signal()
    x[10] = 0.0  # plochý zvod V5
    r = pipe.predict(x, "a")
    assert r.quality.status == "warning" and any("V5" in i for i in r.quality.issues)


@pytest.mark.parametrize(
    ("signal", "match"),
    [
        (np.zeros((1000, 12), np.float32), "from_wfdb"),
        (np.zeros((12, 999), np.float32), "Tvar"),
        (np.full((12, 1000), np.nan, np.float32), "NaN"),
    ],
)
def test_invalid_input_rejected(pipe, signal, match):
    with pytest.raises(contract.InvalidInputError, match=match):
        pipe.predict(signal, "zly")


def test_wrong_threshold_checkpoint_rejected(tmp_path):
    _write_run(tmp_path, wrong_sha=True)
    with pytest.raises(inference.PipelineError, match="SHA256"):
        inference.Pipeline.load("ae", "clf", checkpoint_dir=tmp_path)


def test_missing_checkpoint_has_hint(tmp_path):
    with pytest.raises(inference.PipelineError, match="Release"):
        inference.Pipeline.load("ae", "clf", checkpoint_dir=tmp_path)


# --- záznamy PTB-XL -------------------------------------------------------------


def test_ptbxl_path_detection(tmp_path):
    from ekg.data import ptbxl

    assert inference.ptbxl_id(ptbxl.PTBXL_DIR / "records100" / "00000" / "00009_lr") == 9
    assert inference.ptbxl_id(tmp_path / "00009_lr") is None  # mimo PTB-XL
    assert inference.ptbxl_id(ptbxl.PTBXL_DIR / "records100" / "00000" / "moj") is None


def test_test_set_locked():
    test_id = int(splits.load_manifest().query("split == 'test'").index[0])
    with pytest.raises(splits.TestSetLockedError):
        list(inference.ptbxl_records([test_id]))


@pytest.mark.skipif(
    not (paths.CHECKPOINT_DIR / "clf_v1" / "thresholds.json").exists()
    or not (paths.CHECKPOINT_DIR / "ae_v1" / "threshold.json").exists(),
    reason="checkpointy ae_v1 / clf_v1 nie sú k dispozícii",
)
def test_real_models_use_saved_thresholds():
    pipe = inference.Pipeline.load()
    ae = json.loads((paths.CHECKPOINT_DIR / "ae_v1" / "threshold.json").read_text("utf-8"))
    clf = json.loads((paths.CHECKPOINT_DIR / "clf_v1" / "thresholds.json").read_text("utf-8"))
    assert pipe.anomaly_threshold == ae["threshold"]
    assert pipe.label_thresholds.tolist() == [
        clf["labels"][n]["threshold"] for n in contract.LABELS
    ]
    assert clf["checkpoint_sha256"][:12] in pipe.model_versions["classifier"]
