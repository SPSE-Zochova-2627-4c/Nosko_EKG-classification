"""Spoločná inference oboch modelov (issue #19).

Jeden 12-zvodový záznam (12, 1000) v mV → kompletný výstup podľa kontraktu (PredictionResult):

    1. overenie vstupu (contract.validate_signal) – neplatný vstup = InvalidInputError
    2. kontrola kvality (#7) – nikdy záznam neodmietne, iba pridá varovanie
    3. preprocessing s parametrami z train množiny (#8)
    4. Model A: rekonštrukčná chyba po zvodoch → anomaly score, prah z validation (#13)
    5. Model B: 5 pravdepodobností a rozhodnutí s prahmi z validation (#17) – beží VŽDY,
       bez ohľadu na Model A (žiadny gating), takže výstup má vždy všetkých 5 skóre
    6. pravidlo súladu modelov (consistency_rule)

Načítané prahy sa overia voči SHA256 checkpointov a všetky verzie (preprocessing, mapovanie
labelov, rozdelenie dát) musia sedieť, inak sa pipeline nespustí. Predvolene beží na CPU vo
float32, takže opakovaný beh dá bitovo rovnaký výsledok.

    from ekg.inference import Pipeline
    pipe = Pipeline.load()                      # ae_v1 + clf_v1
    result = pipe.predict(signal, record_id="moj-zaznam")
    print(result.to_json())

Spustenie:  uv run ekg-predict <cesta k WFDB záznamu bez prípony> [--json out.json] [--csv out.csv]
            uv run ekg-predict --ecg-id 9 [--ecg-id 12 …]   (záznam z PTB-XL, nie z test množiny)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ekg import contract, paths, quality
from ekg.contract import (
    AnomalyResult,
    Consistency,
    LabelResult,
    Metadata,
    PredictionResult,
)

DEFAULT_AE_RUN = "ae_v1"
DEFAULT_CLF_RUN = "clf_v1"
RELEASE_HINT = "Stiahni model z GitHub Release model-<run> (pozri README) do checkpoints/<run>/."


class PipelineError(RuntimeError):
    """Pipeline sa nedá zostaviť (chýba checkpoint, prahy nepatria k modelu, iná verzia…)."""


# --- pravidlo súladu ----------------------------------------------------------


def consistency_rule(is_anomaly: bool, labels: dict[str, LabelResult]) -> Consistency:
    """Súlad Modelu A a Modelu B.

    Modely sa zhodujú, ak buď (a) A hlási anomáliu a B označil aspoň jeden label, alebo
    (b) A anomáliu nehlási a B neoznačil žiadny label. Rozhoduje iba porovnanie s prahmi
    (žiadne pásmo okolo prahu), aby bolo pravidlo jednoduché a overiteľné. Pri nesúlade
    ostávajú výsledky oboch modelov nezmenené – žiadny model druhý neprebije.
    """
    positive = [name for name, r in labels.items() if r.positive]
    if is_anomaly == bool(positive):
        return Consistency(models_agree=True, note=None)
    if positive:
        found = ", ".join(f"{name} (P = {labels[name].probability:.2f})" for name in positive)
        return Consistency(
            models_agree=False,
            note=f"Model A nehlási anomáliu, ale Model B označil {found} – "
            "záznam odporúčame skontrolovať.",
        )
    return Consistency(
        models_agree=False,
        note="Model A hlási anomáliu, ale Model B neoznačil žiadnu z porúch "
        f"{', '.join(contract.LABELS)} – môže ísť o inú poruchu, šum alebo falošný poplach.",
    )


# --- načítanie ----------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path) -> Path:
    if not path.exists():
        raise PipelineError(f"Chýba súbor {path}. {RELEASE_HINT}")
    return path


def _model_version(run: str, ckpt: dict, digest: str) -> str:
    return f"{run} (epocha {ckpt['extra'].get('epoch')}, sha256 {digest[:12]})"


@dataclass
class Pipeline:
    autoencoder: torch.nn.Module
    classifier: torch.nn.Module
    preprocessor: object
    anomaly_threshold: float
    aggregation: str
    label_thresholds: np.ndarray  # (5,) v poradí contract.LABELS
    model_versions: dict[str, str]
    device: torch.device

    @classmethod
    def load(
        cls,
        ae_run: str = DEFAULT_AE_RUN,
        clf_run: str = DEFAULT_CLF_RUN,
        device: str | torch.device = "cpu",
        checkpoint_dir: Path | None = None,
    ) -> Pipeline:
        """Načíta oba checkpointy, prah anomálie, prahy labelov a preprocessing; všetko overí."""
        from ekg import label_thresholds
        from ekg.models.autoencoder import ConvAutoencoder
        from ekg.models.classifier import ResNet1dClassifier
        from ekg.preprocessing import PREPROCESSING_VERSION, Preprocessor

        root = Path(checkpoint_dir or paths.CHECKPOINT_DIR)
        ae_path = _require(root / ae_run / "best.pt")
        clf_path = _require(root / clf_run / "best.pt")
        ae, ae_ckpt = ConvAutoencoder.load(ae_path)
        clf, clf_ckpt = ResNet1dClassifier.load(clf_path)
        ae_sha, clf_sha = _sha256(ae_path), _sha256(clf_path)

        ae_th = json.loads(_require(root / ae_run / "threshold.json").read_text("utf-8"))
        if ae_th["checkpoint_sha256"] != ae_sha:
            raise PipelineError(f"Prah anomálie nepatrí k checkpointu {ae_run} (iný SHA256).")
        clf_th = label_thresholds.load_thresholds(_require(root / clf_run / "thresholds.json"))
        if clf_th["checkpoint_sha256"] != clf_sha:
            raise PipelineError(f"Prahy labelov nepatria k checkpointu {clf_run} (iný SHA256).")

        for name, ckpt in (("autoencoder", ae_ckpt), ("klasifikátor", clf_ckpt)):
            if ckpt["versions"]["preprocessing"] != PREPROCESSING_VERSION:
                raise PipelineError(
                    f"{name} bol natrénovaný s preprocessingom "
                    f"{ckpt['versions']['preprocessing']}, kód má {PREPROCESSING_VERSION}."
                )
        shared = ("preprocessing", "label_mapping", "splits")
        if any(ae_ckpt["versions"][k] != clf_ckpt["versions"][k] for k in shared):
            raise PipelineError("Modely boli natrénované s inými verziami dát.")

        device = torch.device(device)
        return cls(
            autoencoder=ae.to(device).eval(),
            classifier=clf.to(device).eval(),
            preprocessor=Preprocessor.load(),
            anomaly_threshold=float(ae_th["threshold"]),
            aggregation=ae_th["aggregation"],
            label_thresholds=np.array(
                [clf_th["labels"][name]["threshold"] for name in contract.LABELS]
            ),
            model_versions={
                "autoencoder": _model_version(ae_run, ae_ckpt, ae_sha),
                "classifier": _model_version(clf_run, clf_ckpt, clf_sha),
            },
            device=device,
        )

    # --- predikcia ------------------------------------------------------------

    @torch.no_grad()
    def scores(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Surové výstupy pre overené signály (B, 12, 1000) v mV.

        Vráti (chyba záznamu (B,), chyba po zvodoch (B, 12), pravdepodobnosti (B, 5)).

        Každý záznam sa počíta samostatne (dávka 1) a na CPU v 1 vlákne. Výsledok záznamu
        tak nezávisí od toho, s akými inými záznamami prišiel, ani od počtu jadier počítača
        (pri väčšej dávke sa poradie sčítania vo float32 mení a P sa líši v ~1e-7).
        Navyše v PyTorch 2.14 (ROCm build) trvá dávka 1 pri 16 vláknach ~0,3 s, v 1 vlákne
        ~2–15 ms na model.
        """
        from ekg.models.recon_error import aggregate, per_lead_error

        x_all = torch.from_numpy(self.preprocessor.transform(signals))
        threads = torch.get_num_threads()
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        try:
            lead_err, probs = [], []
            for i in range(len(x_all)):
                x = x_all[i : i + 1].to(self.device)
                lead_err.append(per_lead_error(x, self.autoencoder(x)).cpu())
                probs.append(torch.sigmoid(self.classifier(x).float()).cpu())
        finally:
            torch.set_num_threads(threads)
        lead_err = torch.cat(lead_err)
        record_err = aggregate(lead_err, self.aggregation)
        return record_err.numpy(), lead_err.numpy(), torch.cat(probs).numpy()

    def predict_batch(
        self,
        signals: list[np.ndarray] | np.ndarray,
        record_ids: list[str],
        metadata: list[Metadata | None] | None = None,
        annotations: list | None = None,
    ) -> list[PredictionResult]:
        """Výstupy pre viac záznamov naraz. Neplatný záznam vyhodí InvalidInputError."""
        from ekg.models.recon_error import lead_dict

        if len(signals) != len(record_ids):
            raise ValueError("Počet signálov a record_ids sa musí zhodovať.")
        batch = contract.to_batch(signals)  # overí každý záznam
        n = len(batch)
        metadata = metadata or [None] * n
        annotations = annotations or [None] * n
        record_err, lead_err, probs = self.scores(batch)
        results = []
        for i in range(n):
            error = float(record_err[i])
            is_anomaly = error >= self.anomaly_threshold
            labels = {
                name: LabelResult(
                    probability=float(probs[i, j]),
                    threshold=float(self.label_thresholds[j]),
                    positive=bool(probs[i, j] >= self.label_thresholds[j]),
                )
                for j, name in enumerate(contract.LABELS)
            }
            results.append(
                PredictionResult(
                    record_id=str(record_ids[i]),
                    model_versions=dict(self.model_versions),
                    anomaly=AnomalyResult(
                        anomaly_score=error,
                        is_anomaly=bool(is_anomaly),
                        reconstruction_error=error,
                        threshold=self.anomaly_threshold,
                        error_per_lead=lead_dict(lead_err[i]),
                    ),
                    labels=labels,
                    consistency=consistency_rule(bool(is_anomaly), labels),
                    quality=quality.assess(batch[i], annotations[i]),
                    metadata=metadata[i] or Metadata(),
                )
            )
        return results

    def predict(
        self,
        signal: np.ndarray,
        record_id: str,
        metadata: Metadata | None = None,
        annotations=None,
    ) -> PredictionResult:
        """Výstup pre jeden záznam (12, 1000) v mV."""
        return self.predict_batch([signal], [record_id], [metadata], [annotations])[0]


# --- načítanie záznamov a CLI --------------------------------------------------


def read_wfdb(path: str | Path) -> np.ndarray:
    """WFDB záznam (cesta bez prípony) → (12, 1000) v poradí kontraktu."""
    import wfdb

    record = wfdb.rdrecord(str(path))
    return contract.from_wfdb(record.p_signal, record.sig_name, int(record.fs))


def ptbxl_id(path: str | Path) -> int | None:
    """ecg_id, ak cesta vedie na záznam PTB-XL (records100/…/00009_lr), inak None."""
    import re

    from ekg.data import ptbxl

    path = Path(path).resolve()
    match = re.fullmatch(r"(\d{5})_lr", path.name)
    if match and path.is_relative_to(Path(ptbxl.PTBXL_DIR).resolve()):
        return int(match.group(1))
    return None


def ptbxl_records(ecg_ids: list[int]):
    """Signál, metadáta a anotácie kvality záznamov PTB-XL; test množina je zamknutá (#22)."""
    from ekg import export
    from ekg.data import ptbxl, splits

    manifest = splits.load_manifest()
    locked = [i for i in ecg_ids if manifest.loc[i, "split"] == "test"]
    if locked:
        raise splits.TestSetLockedError(
            f"Záznamy {locked} patria do test množiny, ktorá je zamknutá pre #22."
        )
    db = ptbxl.load_database()
    for ecg_id in ecg_ids:
        row = db.loc[ecg_id]
        signal = read_wfdb(ptbxl.PTBXL_DIR / row["filename_lr"])
        yield str(ecg_id), signal, export.metadata_from_ptbxl(row), row


def main(argv: list[str] | None = None) -> int:
    from ekg import export

    parser = argparse.ArgumentParser(description="Inference oboch modelov na EKG zázname.")
    parser.add_argument("records", nargs="*", help="WFDB záznam (cesta bez prípony .hea/.dat)")
    parser.add_argument("--ecg-id", type=int, action="append", default=[], help="záznam PTB-XL")
    parser.add_argument("--ae-run", default=DEFAULT_AE_RUN)
    parser.add_argument("--clf-run", default=DEFAULT_CLF_RUN)
    parser.add_argument("--device", default="cpu", help="cpu (predvolené, reprodukovateľné)/cuda")
    parser.add_argument("--json", type=Path, help="uloží výsledky do JSON")
    parser.add_argument("--csv", type=Path, help="uloží výsledky do CSV")
    args = parser.parse_args(argv)
    if not args.records and not args.ecg_id:
        parser.error("zadaj aspoň jeden WFDB záznam alebo --ecg-id")

    try:
        pipe = Pipeline.load(args.ae_run, args.clf_run, args.device)
        # záznam PTB-XL zadaný cestou sa spracuje ako --ecg-id (metadáta, zámok test množiny)
        ecg_ids = list(args.ecg_id) + [i for p in args.records if (i := ptbxl_id(p)) is not None]
        files = [p for p in args.records if ptbxl_id(p) is None]
        items = [(Path(p).name, read_wfdb(p), Metadata(source=Path(p).name), None)
                 for p in files]  # fmt: skip
        items += list(ptbxl_records(ecg_ids))
        ids, signals, meta, ann = (list(col) for col in zip(*items, strict=True))
        results = pipe.predict_batch(signals, ids, meta, ann)
    except (PipelineError, contract.InvalidInputError, OSError, KeyError) as exc:
        print(f"Chyba: {exc}", file=sys.stderr)
        return 1

    for r in results:
        print(summary(r))
    if args.json:
        export.to_json(results, args.json)
        print(f"JSON: {args.json}")
    if args.csv:
        export.to_csv(results, args.csv)
        print(f"CSV: {args.csv}")
    return 0


def summary(r: PredictionResult) -> str:
    """Čitateľný prehľad jedného výsledku."""
    a = r.anomaly
    lines = [
        f"Záznam {r.record_id}",
        f"  anomaly score {a.anomaly_score:.4f} (prah {a.threshold:.4f}) → "
        f"{'ANOMÁLIA' if a.is_anomaly else 'v norme'}",
    ]
    for name, lab in r.labels.items():
        mark = "ÁNO" if lab.positive else "nie"
        lines.append(f"  {name:5s} P = {lab.probability:.3f} (prah {lab.threshold:.3f}) {mark}")
    lines.append(f"  súlad modelov: {'áno' if r.consistency.models_agree else 'NIE'}")
    if r.consistency.note:
        lines.append(f"    {r.consistency.note}")
    if r.quality.issues:
        lines.append(f"  kvalita: {r.quality.status} – {'; '.join(r.quality.issues)}")
    lines.append(f"  {r.disclaimer}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
