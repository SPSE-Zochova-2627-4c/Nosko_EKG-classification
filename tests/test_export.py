"""Testy metadát a exportu do JSON/CSV (issue #21)."""

import copy
import csv
import json
import math

import jsonschema
import pytest
import torch

from ekg import contract, export
from ekg.models.autoencoder import ConvAutoencoder
from ekg.paths import PROJECT_ROOT

DOCS = PROJECT_ROOT / "docs"
SCHEMA = json.loads(
    (DOCS / "schema" / f"vystup_v{contract.SCHEMA_VERSION}.schema.json").read_text(encoding="utf-8")
)


@pytest.fixture
def results():
    """Príklad z dokumentácie + druhý záznam s chýbajúcimi metadátami a problémami kvality."""
    data = json.loads((DOCS / "priklad_vystupu.json").read_text(encoding="utf-8"))
    a = contract.PredictionResult.from_dict(data)
    b = copy.deepcopy(a)
    b.record_id = "B"
    b.metadata = contract.Metadata(age=None, age_over_89=True, sex=None, source=None)
    b.quality = contract.Quality(
        "warning", ["plochý zvod: V5", "extrémna amplitúda: I (12.3 mV), V2 (10.1 mV)"]
    )
    b.consistency = contract.Consistency(models_agree=True, note=None)
    b.anomaly.reconstruction_error = 0.1 + 0.2  # 0.30000000000000004 – presnosť sa nesmie stratiť
    b.model_versions = {"autoencoder": "ae_v1"}  # klasifikátor chýba
    return [a, b]


# --- metadáta ----------------------------------------------------------------


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"age": 56.0, "sex": 1}, (56, False, "female")),
        ({"age": 300.0, "sex": 0}, (None, True, "male")),  # PTB-XL anonymizácia > 89 rokov
        ({"age": float("nan"), "sex": float("nan")}, (None, False, None)),
        ({}, (None, False, None)),
    ],
)
def test_metadata_from_ptbxl(row, expected):
    m = export.metadata_from_ptbxl(row)
    assert (m.age, m.age_over_89, m.sex) == expected
    assert m.source == "PTB-XL"


# --- JSON --------------------------------------------------------------------


def test_json_roundtrip(results, tmp_path):
    export.to_json(results, tmp_path / "v.json")
    assert export.load_json(tmp_path / "v.json") == results


def test_json_matches_schema(results, tmp_path):
    export.to_json(results, tmp_path / "v.json")
    jsonschema.validate(json.loads((tmp_path / "v.json").read_text(encoding="utf-8")), SCHEMA)


@pytest.mark.parametrize(
    "path, value",
    [
        (("labels", "MI", "probability"), 1.5),
        (("metadata", "age"), 300),
        (("quality", "status"), "bad"),
        (("anomaly", "error_per_lead", "V1"), -0.1),
    ],
)
def test_schema_rejects_invalid(results, tmp_path, path, value):
    export.to_json(results, tmp_path / "v.json")
    data = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    target = data["results"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)


def test_schema_rejects_missing_label(results, tmp_path):
    export.to_json(results, tmp_path / "v.json")
    data = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    del data["results"][0]["labels"]["AFIB"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)


def test_old_schema_1_0_still_loads_and_upgrades(results):
    old = results[0].to_dict()
    old["schema_version"] = "1.0"
    del old["metadata"]["age_over_89"]  # v 1.0 pole neexistovalo
    loaded = contract.PredictionResult.from_dict(old)
    assert loaded.schema_version == "1.1" and loaded.metadata.age_over_89 is False


# --- CSV ---------------------------------------------------------------------


def test_csv_roundtrip_exact(results, tmp_path):
    export.to_csv(results, tmp_path / "v.csv")
    loaded = export.load_csv(tmp_path / "v.csv")
    assert loaded == results
    assert loaded[1].anomaly.reconstruction_error == 0.1 + 0.2  # bitovo presne


def test_csv_columns_and_missing_values(results, tmp_path):
    export.to_csv(results, tmp_path / "v.csv")
    with open(tmp_path / "v.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == tuple(export.CSV_COLUMNS)
        rows = list(reader)
    b = rows[1]
    assert b["age"] == "" and b["sex"] == "" and b["age_over_89"] == "true"  # nič vymyslené
    assert b["model_classifier"] == "" and b["consistency_note"] == ""
    assert b["quality_issues"] == "plochý zvod: V5 | extrémna amplitúda: I (12.3 mV), V2 (10.1 mV)"
    assert rows[0]["positive_MI"] == "true" and rows[0]["quality_issues"] == ""


def test_csv_rejects_separator_in_issue(results, tmp_path):
    results[0].quality.issues = ["a | b"]
    with pytest.raises(ValueError, match="oddeľovač"):
        export.to_csv(results, tmp_path / "v.csv")


def test_csv_rejects_unknown_model(results, tmp_path):
    results[0].model_versions["iny_model"] = "x"
    with pytest.raises(ValueError, match="iny_model"):
        export.to_csv(results, tmp_path / "v.csv")


def test_csv_missing_column(results, tmp_path):
    row = export.to_row(results[0])
    del row["prob_HYP"]
    with pytest.raises(ValueError, match="prob_HYP"):
        export.from_row(row)


def test_documented_examples_load():
    [a] = [
        contract.PredictionResult.from_dict(json.loads((DOCS / "priklad_vystupu.json").read_text()))
    ]
    csv_results = export.load_csv(DOCS / "priklad_exportu.csv")
    assert csv_results[0] == a
    assert csv_results[1].metadata.age_over_89 and csv_results[1].metadata.age is None
    assert not math.isnan(csv_results[0].anomaly.reconstruction_error)


# --- zmena schémy nesmie zneplatniť natrénované modely -----------------------


def test_checkpoint_with_older_schema_version_loads(tmp_path):
    ConvAutoencoder().save(tmp_path / "ae.pt")
    ckpt = torch.load(tmp_path / "ae.pt", weights_only=True)
    ckpt["contract"]["schema_version"] = "1.0"
    torch.save(ckpt, tmp_path / "old.pt")
    ConvAutoencoder.load(tmp_path / "old.pt")  # nesmie vyhodiť chybu
