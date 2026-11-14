"""Metadáta záznamu a export výsledkov do JSON a CSV (issue #21).

JSON – úplný, vnorený formát (presne štruktúra PredictionResult z kontraktu):
    {"schema_version": "1.2", "created_at": "...", "results": [ {…}, {…} ]}

CSV – plochá tabuľka, jeden riadok = jeden záznam, stĺpce v CSV_COLUMNS
(popis v docs/export.md). Chýbajúca hodnota = prázdna bunka, nikdy odhad.

Oba formáty sa dajú načítať späť (load_json, load_csv) na rovnaké PredictionResult.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from ekg import contract
from ekg.contract import (
    AnomalyResult,
    Consistency,
    LabelResult,
    Metadata,
    PredictionResult,
    Quality,
)

PTBXL_ANONYMIZED_AGE = 300  # PTB-XL uvádza 300 pre pacientov starších ako 89 rokov
ISSUE_SEPARATOR = " | "

MODEL_KEYS = ("autoencoder", "classifier")
OLD_WITHOUT_SCORE = ("1.0", "1.1")  # CSV bez stĺpca anomaly_score (pridaný v schéme 1.2)
CSV_COLUMNS = (
    ["record_id", "schema_version", "anomaly_score", "is_anomaly", "reconstruction_error"]
    + ["anomaly_threshold"]
    + [f"error_{lead}" for lead in contract.LEADS]
    + [f"prob_{label}" for label in contract.LABELS]
    + [f"threshold_{label}" for label in contract.LABELS]
    + [f"positive_{label}" for label in contract.LABELS]
    + ["models_agree", "consistency_note", "quality_status", "quality_issues"]
    + ["age", "age_over_89", "sex", "source"]
    + [f"model_{key}" for key in MODEL_KEYS]
    + ["disclaimer"]
)


# --- metadáta ----------------------------------------------------------------


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def metadata_from_ptbxl(row) -> Metadata:
    """Vek a pohlavie z riadku ptbxl_database.csv; nič sa nedopĺňa odhadom."""
    age = row.get("age")
    over_89 = not _missing(age) and int(age) == PTBXL_ANONYMIZED_AGE
    return Metadata(
        age=None if _missing(age) or over_89 else int(age),
        age_over_89=over_89,
        sex=contract.sex_from_ptbxl(row.get("sex")),
        source="PTB-XL",
    )


# --- JSON --------------------------------------------------------------------


def to_json(results: list[PredictionResult], path: Path) -> None:
    data = {
        "schema_version": contract.SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": contract.DISCLAIMER,
        "results": [r.to_dict() for r in results],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> list[PredictionResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") not in contract.SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Nepodporovaná verzia súboru {data.get('schema_version')!r}.")
    return [PredictionResult.from_dict(item) for item in data["results"]]


# --- CSV ---------------------------------------------------------------------


def _cell(value) -> str:
    """Hodnota do CSV: None → prázdne, bool → true/false, float → presný zápis (repr)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def to_row(r: PredictionResult) -> dict[str, str]:
    for key in r.model_versions:
        if key not in MODEL_KEYS:
            raise ValueError(f"Neznámy model {key!r} – CSV pozná {MODEL_KEYS}.")
    for issue in r.quality.issues:
        if ISSUE_SEPARATOR in issue:
            raise ValueError(f"Problém kvality obsahuje oddeľovač {ISSUE_SEPARATOR!r}: {issue}")
    row = {
        "record_id": r.record_id,
        "schema_version": r.schema_version,
        "anomaly_score": r.anomaly.anomaly_score,
        "is_anomaly": r.anomaly.is_anomaly,
        "reconstruction_error": r.anomaly.reconstruction_error,
        "anomaly_threshold": r.anomaly.threshold,
        **{f"error_{lead}": v for lead, v in r.anomaly.error_per_lead.items()},
        **{f"prob_{k}": v.probability for k, v in r.labels.items()},
        **{f"threshold_{k}": v.threshold for k, v in r.labels.items()},
        **{f"positive_{k}": v.positive for k, v in r.labels.items()},
        "models_agree": r.consistency.models_agree,
        "consistency_note": r.consistency.note,
        "quality_status": r.quality.status,
        "quality_issues": ISSUE_SEPARATOR.join(r.quality.issues) or None,
        "age": r.metadata.age,
        "age_over_89": r.metadata.age_over_89,
        "sex": r.metadata.sex,
        "source": r.metadata.source,
        **{f"model_{key}": r.model_versions.get(key) for key in MODEL_KEYS},
        "disclaimer": r.disclaimer,
    }
    return {col: _cell(row[col]) for col in CSV_COLUMNS}


def to_csv(results: list[PredictionResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow(to_row(r))


def _parse_bool(text: str) -> bool:
    if text not in ("true", "false"):
        raise ValueError(f"Očakáva sa true/false, nie {text!r}.")
    return text == "true"


def _opt(text: str) -> str | None:
    return text if text != "" else None


def from_row(row: dict[str, str]) -> PredictionResult:
    if row.get("schema_version") not in contract.SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Nepodporovaná verzia schémy {row.get('schema_version')!r}.")
    if row["schema_version"] in OLD_WITHOUT_SCORE and "anomaly_score" not in row:
        row = {**row, "anomaly_score": row.get("reconstruction_error", "")}
    missing = [c for c in CSV_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"CSV nemá stĺpce: {missing}")
    issues = row["quality_issues"]
    return PredictionResult(
        record_id=row["record_id"],
        model_versions={k: row[f"model_{k}"] for k in MODEL_KEYS if row[f"model_{k}"] != ""},
        anomaly=AnomalyResult(
            anomaly_score=float(row["anomaly_score"]),
            is_anomaly=_parse_bool(row["is_anomaly"]),
            reconstruction_error=float(row["reconstruction_error"]),
            threshold=float(row["anomaly_threshold"]),
            error_per_lead={lead: float(row[f"error_{lead}"]) for lead in contract.LEADS},
        ),
        labels={
            k: LabelResult(
                probability=float(row[f"prob_{k}"]),
                threshold=float(row[f"threshold_{k}"]),
                positive=_parse_bool(row[f"positive_{k}"]),
            )
            for k in contract.LABELS
        },
        consistency=Consistency(
            models_agree=_parse_bool(row["models_agree"]), note=_opt(row["consistency_note"])
        ),
        quality=Quality(
            status=row["quality_status"], issues=issues.split(ISSUE_SEPARATOR) if issues else []
        ),
        metadata=Metadata(
            age=int(row["age"]) if row["age"] else None,
            age_over_89=_parse_bool(row["age_over_89"]),
            sex=_opt(row["sex"]),
            source=_opt(row["source"]),
        ),
        disclaimer=row["disclaimer"],
    )


def load_csv(path: Path) -> list[PredictionResult]:
    with open(path, newline="", encoding="utf-8") as f:
        return [from_row(row) for row in csv.DictReader(f)]
