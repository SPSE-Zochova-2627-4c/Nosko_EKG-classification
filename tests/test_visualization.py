"""Testy zobrazenia výsledkov a web aplikácie (issue #20)."""

import io
import json

import numpy as np
import pytest
import torch

from ekg import app_support, contract, inference, paths, visualization
from tests.test_inference import _signal, _write_run

APP = paths.PROJECT_ROOT / "streamlit_app.py"
HAS_MODELS = all(
    (paths.CHECKPOINT_DIR / run / name).exists()
    for run, name in (("ae_v1", "threshold.json"), ("clf_v1", "thresholds.json"))
)


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    _write_run(root)
    return inference.Pipeline.load("ae", "clf", checkpoint_dir=root)


# --- rekonštrukcia ----------------------------------------------------------------


def test_reconstruction_matches_anomaly_score(pipe):
    """Zobrazená rekonštrukcia je presne tá, z ktorej sa počíta anomaly score."""
    x = _signal(1)
    result = pipe.predict(x, "a")
    filtered, recon = pipe.reconstruct(x)
    assert filtered.shape == recon.shape == (12, 1000)
    std = np.asarray(pipe.preprocessor.std, dtype=np.float32)[:, None]
    per_lead = (((filtered - recon) / std) ** 2).mean(axis=1)
    expected = [result.anomaly.error_per_lead[lead] for lead in contract.LEADS]
    assert np.allclose(per_lead, expected, rtol=1e-4)
    assert per_lead.mean() == pytest.approx(result.anomaly.anomaly_score, rel=1e-4)
    # signál po filtrovaní v mV = filter(x), nie normalizovaný
    assert np.allclose(filtered, pipe.preprocessor.filter(x), atol=1e-5)


def test_reconstruct_rejects_invalid_and_restores_threads(pipe):
    before = torch.get_num_threads()
    with pytest.raises(contract.InvalidInputError):
        pipe.reconstruct(np.zeros((1000, 12), np.float32))
    pipe.reconstruct(_signal())
    assert torch.get_num_threads() == before


# --- obrázok ----------------------------------------------------------------------


def _report(pipe, **kw):
    x = _signal(2)
    result = pipe.predict(x, "rec-7")
    filtered, recon = pipe.reconstruct(x)
    return result, visualization.plot_report(filtered, recon, result, **kw)


def test_report_shows_all_leads_with_time_axis(pipe):
    result, fig = _report(pipe)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    for lead in contract.LEADS:
        name = visualization.lead_label(lead)
        assert any(t.startswith(f"{name} ") for t in titles), name
        assert any(f"chyba {result.anomaly.error_per_lead[lead]:.4f}" in t for t in titles)
    lead_axes = [ax for ax in fig.axes if ax.get_title(loc="left").split(" ")[0] in
                 {visualization.lead_label(lead) for lead in contract.LEADS}]  # fmt: skip
    assert len(lead_axes) == 12
    assert all(ax.get_xlim() == (0, 10) for ax in lead_axes)
    assert sum(ax.get_xlabel() == "čas (s)" for ax in lead_axes) == 2
    assert all(len(ax.lines) == 2 for ax in lead_axes)  # signál + rekonštrukcia


def test_report_texts_match_pipeline(pipe):
    result, fig = _report(pipe)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    texts += " ".join(t.get_text() for t in fig.texts)
    assert f"{result.anomaly.anomaly_score:.4f}" in texts
    assert f"prah {result.anomaly.threshold:.4f}" in texts
    for lab in result.labels.values():
        assert f"{lab.probability:.3f}  (prah {lab.threshold:.2f})" in texts
    assert result.disclaimer in texts and "rec-7" in fig._suptitle.get_text()


def test_report_without_reconstruction(pipe):
    _, fig = _report(pipe, show_reconstruction=False)
    leads = [ax for ax in fig.axes if ax.get_xlim() == (0, 10)]
    assert all(len(ax.lines) == 1 for ax in leads)


def test_report_rejects_wrong_shape(pipe):
    result = pipe.predict(_signal(), "a")
    with pytest.raises(ValueError, match="tvar"):
        visualization.plot_report(np.zeros((1000, 12)), np.zeros((12, 1000)), result)


def test_verdict():
    labels = {
        n: contract.LabelResult(0.9 if n == "CD" else 0.1, 0.5, n == "CD") for n in contract.LABELS
    }
    r = contract.PredictionResult(
        record_id="x",
        model_versions={},
        anomaly=contract.AnomalyResult(0.1, True, 0.1, 0.05, dict.fromkeys(contract.LEADS, 0.1)),
        labels=labels,
        consistency=contract.Consistency(True),
    )
    assert visualization.verdict(r) == "Model A: anomália · Model B: CD"


# --- nahrávanie súborov -----------------------------------------------------------


def _npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_upload_npy():
    x = _signal()
    rid, sig = app_support.signal_from_upload([("moj.npy", _npy(x))])
    assert rid == "moj" and np.array_equal(sig, x)


def test_upload_wfdb(tmp_path):
    import wfdb

    x = _signal()
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    wfdb.wrsamp("rec", fs=100, units=["mV"] * 12, sig_name=names, p_signal=x.T.astype(float),
                fmt=["16"] * 12, write_dir=str(tmp_path))  # fmt: skip
    files = [(f"rec{ext}", (tmp_path / f"rec{ext}").read_bytes()) for ext in (".hea", ".dat")]
    rid, sig = app_support.signal_from_upload(files)
    assert rid == "rec" and sig.shape == (12, 1000)
    assert np.allclose(sig, x, atol=1e-2)  # 16-bitové kvantovanie WFDB


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ([], "žiadny"),
        ([("a.txt", b"x")], "Nepodporovaný"),
        ([("a.hea", b"x")], "práve jeden"),
        ([("a.hea", b"x"), ("b.dat", b"y")], "rovnaké meno"),
        ([("a.npy", b"nie je npy")], "NumPy"),
        ([("a.npy", _npy(np.zeros((1000, 12), np.float32)))], "from_wfdb"),
        ([("a.npy", _npy(np.zeros((12, 500), np.float32)))], "Tvar"),
    ],
)
def test_upload_invalid(files, match):
    with pytest.raises(contract.InvalidInputError, match=match):
        app_support.signal_from_upload(files)


def test_tables(pipe):
    r = pipe.predict(_signal(), "a")
    t = app_support.results_table(r)
    assert list(t["Porucha"]) == list(contract.LABELS)
    assert list(t["Prah"]) == [lab.threshold for lab in r.labels.values()]
    assert app_support.lead_errors_table(r)["Zvod"].tolist()[3:6] == ["aVR", "aVL", "aVF"]


def test_examples_are_not_in_test_set():
    from ekg.data import splits

    manifest = splits.load_manifest()
    assert all(manifest.loc[i, "split"] == "val" for i in app_support.EXAMPLES)


# --- CLI a web aplikácia (skutočné modely a dataset) ------------------------------


@pytest.mark.skipif(not HAS_MODELS, reason="checkpointy ae_v1 / clf_v1 nie sú k dispozícii")
def test_cli_writes_png(tmp_path):
    out = tmp_path / "r.png"
    assert visualization.main(["--ecg-id", "10", "-o", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert visualization.main(["--ecg-id", "9", "-o", str(out)]) == 1  # test množina


@pytest.mark.skipif(not HAS_MODELS, reason="checkpointy ae_v1 / clf_v1 nie sú k dispozícii")
def test_app_matches_pipeline():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    expected = inference.Pipeline.load().predict(
        *[(s, rid, m, a) for rid, s, m, a in inference.ptbxl_records([10])][0]
    )
    assert metrics["Anomaly score (Model A)"] == f"{expected.anomaly.anomaly_score:.4f}"
    assert metrics["Súlad modelov"] == "áno"
    probs = at.dataframe[0].value["Pravdepodobnosť"].tolist()
    assert probs == pytest.approx([lab.probability for lab in expected.labels.values()])

    at.number_input(key="ecg_id").set_value(127).run()
    assert not at.exception and at.warning[-1].proto.title == "Modely nesúhlasia"

    at.number_input(key="ecg_id").set_value(9).run()
    assert at.error and "test množiny" in at.error[0].value

    at.segmented_control(key="source").set_value("Vlastný súbor").run()
    assert not at.exception and at.info


def test_saved_examples_json_valid():
    """Príklady výstupu z #19 sa dajú načítať aj s aktuálnou schémou."""
    from ekg import export

    path = paths.PROJECT_ROOT / "docs" / "vysledky" / "inference" / "priklad_inference.json"
    results = export.load_json(path)
    assert [r.record_id for r in results] == [str(i) for i in app_support.EXAMPLES]
    assert json.loads(path.read_text("utf-8"))["schema_version"] == contract.SCHEMA_VERSION
