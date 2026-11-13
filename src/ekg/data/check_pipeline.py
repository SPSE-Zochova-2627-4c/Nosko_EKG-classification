"""Kontrolný priechod celou dátovou pipeline (issue #9).

Spustenie:  uv run ekg-check-data

Pre každú množinu a podmnožinu prejde všetky dávky a overí:
- tvar (B, 12, 1000) float32, konečné hodnoty, ciele (B, 5) iba 0/1,
- každý záznam podmnožiny sa objaví práve raz (úplnosť, bez duplicít),
- ciele v dávke = labely v manifeste pre dané ecg_id (väzba anotácia–záznam),
- signál v dávke = záznam načítaný priamo z WFDB + preprocessing (väzba signál–záznam,
  na náhodnej vzorke – overuje, že cache nepomiešala poradie),
- prísny NORM má všetky ciele 0,
- žiadny pacient nie je vo viac ako jednej množine,
- rovnaký seed dá rovnaké poradie dávok, validation je vždy v rovnakom poradí.

Test množina sa tu otvára (allow_test=True) IBA na technickú kontrolu tvarov a väzieb –
nevyhodnocuje sa na nej žiadny model, takže to nie je ladenie na test dátach.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from ekg import contract
from ekg.data import ptbxl, splits
from ekg.data.cache import SignalCache
from ekg.data.dataset import LABEL_COLUMNS, make_dataset, make_loader
from ekg.preprocessing import Preprocessor

N_WFDB_SAMPLES = 25  # na koľkých náhodných záznamoch z každej množiny porovnať s WFDB


def check_loader(name, loader, manifest, prep, rng, root=ptbxl.PTBXL_DIR) -> list[str]:
    problems = []
    seen = []
    targets_by_id = manifest[LABEL_COLUMNS]
    for batch in loader:
        x, y, ids = batch["signal"], batch["target"], batch["ecg_id"]
        if x.dtype != torch.float32 or tuple(x.shape[1:]) != (contract.N_LEADS, contract.N_SAMPLES):
            problems.append(f"{name}: signál {tuple(x.shape)} {x.dtype}")
        if not torch.isfinite(x).all():
            problems.append(f"{name}: NaN/inf v signáli")
        if (
            tuple(y.shape[1:]) != (contract.N_LABELS,)
            or not torch.isin(y, torch.tensor([0.0, 1.0])).all()
        ):
            problems.append(f"{name}: ciele {tuple(y.shape)} mimo 0/1")
        expected = targets_by_id.loc[ids.numpy()].to_numpy(dtype=np.float32)
        if not np.array_equal(y.numpy(), expected):
            problems.append(f"{name}: ciele nesedia s manifestom")
        seen.append(ids.numpy())
    seen = np.concatenate(seen)
    dataset_ids = loader.dataset.ecg_ids
    if len(seen) != len(np.unique(seen)) or set(seen) != set(dataset_ids):
        problems.append(f"{name}: nie každý záznam je v epoche práve raz")

    sample = rng.choice(dataset_ids, size=min(N_WFDB_SAMPLES, len(dataset_ids)), replace=False)
    files = manifest.loc[sample, "filename_lr"]
    direct = prep.transform(np.stack([ptbxl.load_signal(f, root) for f in files]))
    index = {e: i for i, e in enumerate(dataset_ids)}
    from_loader = np.stack([loader.dataset[index[e]]["signal"].numpy() for e in sample])
    if not np.allclose(direct, from_loader, atol=1e-5):
        problems.append(f"{name}: signál z loadera ≠ signál z WFDB (cache pomiešala záznamy?)")
    return problems


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    manifest = splits.load_manifest()
    cache, prep = SignalCache(), Preprocessor.load()
    shared = {"manifest": manifest, "cache": cache, "preprocessor": prep}

    combos = [(s, sub) for s in splits.SPLITS for sub in ("classifier", "strict_norm")]
    problems, patients = [], {}
    print(
        f"{'množina':<6} {'podmnožina':<12} {'záznamy':>8} {'dávky':>6}  pozitívne {LABEL_COLUMNS}"
    )
    for split, subset in combos:
        ds = make_dataset(split, subset, allow_test=(split == "test"), **shared)
        loader = make_loader(split, subset, batch_size=256, dataset=ds)
        problems += check_loader(f"{split}/{subset}", loader, manifest, prep, rng)
        if subset == "strict_norm" and ds.targets.any():
            problems.append(f"{split}/strict_norm: prísny NORM má pozitívny label")
        patients.setdefault(split, set()).update(ds.patient_ids.tolist())
        pos = ds.targets.sum(axis=0).astype(int).tolist()
        print(f"{split:<6} {subset:<12} {len(ds):>8} {len(loader):>6}  {pos}")

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patients[a] & patients[b]
        if overlap:
            problems.append(f"{len(overlap)} pacientov je v {a} aj {b}")

    val_ds = make_dataset("val", "classifier", **shared)
    order = [torch.cat([b["ecg_id"] for b in make_loader("val", dataset=val_ds)]) for _ in range(2)]
    if not torch.equal(*order):
        problems.append("validation loader nie je deterministický")
    train_small = make_dataset("train", "strict_norm", **shared)
    first = [
        next(iter(make_loader("train", dataset=train_small, seed=7)))["ecg_id"] for _ in range(2)
    ]
    if not torch.equal(*first):
        problems.append("rovnaký seed nedal rovnaké poradie train dávok")

    if problems:
        print(f"\nKontrola ZLYHALA ({len(problems)}):")
        for p in problems:
            print("  -", p)
        return 1
    print(
        "\nOK: tvary, ciele, väzba signál–záznam (porovnanie s WFDB), úplnosť, prísny NORM,"
        "\n    nulové prekrývanie pacientov a determinizmus sú v poriadku."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
