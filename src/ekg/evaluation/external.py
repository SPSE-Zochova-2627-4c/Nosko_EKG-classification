"""Externá validácia zmrazeného systému na datasete Georgia (G12EC) – issue #31.

Dataset: The Georgia 12-lead ECG Challenge (G12EC) Database, súčasť PhysioNet/Computing in
Cardiology Challenge 2021, verzia 1.0.3 (DOI 10.13026/34va-7q14), licencia CC BY 4.0.
10 344 záznamov z Emory University (Atlanta, USA), 12 zvodov, 500 Hz, 5 – 10 s, diagnózy ako
kódy SNOMED CT v hlavičke (# Dx). Iná krajina, nemocnica aj prístroj ako PTB-XL (Nemecko).

Stiahnutie (~1,25 GB, overenie SHA256 z oficiálneho SHA256SUMS.txt): docs/externa_validacia.md

Postup – NIČ sa nedotrénováva ani neladí:
    1. diagnózy → labely MI, STTC, CD, HYP, AFIB rovnako, ako ich triedi PTB-XL (MAPPING nižšie):
       pozitívny = kód zodpovedá diagnostickému kódu PTB-XL v danej supertriede,
       nejasný   = v PTB-XL je to iba „tvarový“ nález (napr. abnormálna vlna T) – záznam sa pre
                   tento label nehodnotí (ani pozitívny, ani negatívny),
    2. 500 Hz → 100 Hz polyfázovým filtrom (scipy resample_poly, antialiasing), 5000 → 1000 vzoriek;
       záznamy kratšie ako 10 s sa vynechajú (dopĺňanie by vytvorilo umelý plochý úsek),
       rovnako záznamy, ktoré odmietne kontrakt (#2, napr. NaN v signáli),
    3. zmrazená pipeline (#19) s uloženým train preprocessingom, modelmi a prahmi (#22),
    4. metriky ako na PTB-XL teste (#22) a porovnanie s ním.
Model B hodnotí iba záznamy, pri ktorých vieme všetky labely: aspoň jeden pozitívny label, alebo
iba bežné rytmy bez poruchy (sínusový rytmus, sínusová bradykardia / tachykardia / arytmia);
záznamy iba s inými, nenamapovanými kódmi (napr. PAC, odchýlka osi) sa vynechajú – rovnako ako
PTB-XL vynecháva záznamy bez diagnostického kódu (#8).
Model A: referenčný NORM = záznam iba so sínusovým rytmom (Georgia nemá kód „normálne EKG“),
patológia = aspoň jeden pozitívny label.

Spustenie:  uv run ekg-external [--limit N]
Výstupy:    docs/vysledky/externa/, predikcie v outputs/externa/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ekg import contract, paths

LABELS = list(contract.LABELS)
SOURCE = "Georgia (G12EC), PhysioNet Challenge 2021 v1.0.3"
SOURCE_FS = 500
N_BOOTSTRAP = 1000
SEED = 31

# SNOMED CT kód → (skratka, label, typ). Typ "poz" = pozitívny, "nej" = nejasný pre label.
# Zdroj kódov a počtov: dx_mapping_scored.csv / dx_mapping_unscored.csv (Challenge 2021).
MAPPING: dict[str, tuple[str, str, str]] = {
    # AFIB – rytmus; flutter (164890007) sa do AFIB nezaratúva, rovnako ako v PTB-XL (AFLT)
    "164889003": ("AF", "AFIB", "poz"),
    "195080001": ("AFAFL", "AFIB", "poz"),  # fibrilácia a flutter
    # MI
    "164865005": ("MI", "MI", "poz"),
    "164917005": ("QAb", "MI", "nej"),  # Q vlny – v PTB-XL tvarový kód QWAVE
    # STTC – PTB-XL: NDT, NST_, LNGQT, ISC*, EL, DIG, ANEUR
    "428750005": ("NSSTTA", "STTC", "poz"),
    "111975006": ("LQT", "STTC", "poz"),
    "426434006": ("AnMIs", "STTC", "poz"),
    "425419005": ("IIs", "STTC", "poz"),
    "425623009": ("LIs", "STTC", "poz"),
    "413444003": ("AMIs", "STTC", "poz"),
    "164934002": ("TAb", "STTC", "nej"),  # PTB-XL TAB_ – tvarový
    "59931005": ("TInv", "STTC", "nej"),  # INVT – tvarový
    "164930006": ("STIAb", "STTC", "nej"),
    "429622005": ("STD", "STTC", "nej"),  # STD_ – tvarový
    "164931005": ("STE", "STTC", "nej"),  # STE_ – tvarový
    # CD – PTB-XL: CLBBB, CRBBB, ILBBB, IRBBB, LAFB, LPFB, IVCD, 1AVB, 2AVB, 3AVB, WPW
    "164909002": ("LBBB", "CD", "poz"),
    "733534002": ("CLBBB", "CD", "poz"),
    "59118001": ("RBBB", "CD", "poz"),
    "713427006": ("CRBBB", "CD", "poz"),
    "713426002": ("IRBBB", "CD", "poz"),
    "251120003": ("ILBBB", "CD", "poz"),
    "6374002": ("BBB", "CD", "poz"),
    "445118002": ("LAnFB", "CD", "poz"),
    "445211001": ("LPFB", "CD", "poz"),
    "698252002": ("NSIVCB", "CD", "poz"),
    "270492004": ("IAVB", "CD", "poz"),
    "195042002": ("IIAVB", "CD", "poz"),
    "233917008": ("AVB", "CD", "poz"),
    "27885002": ("CHB", "CD", "poz"),
    "74390002": ("WPW", "CD", "poz"),
    "164947007": ("LPR", "CD", "nej"),  # predĺžené PR – v PTB-XL tvarový LPR
    # HYP – PTB-XL: LVH, RVH, LAO/LAE, RAO/RAE, SEHYP
    "164873001": ("LVH", "HYP", "poz"),
    "89792004": ("RVH", "HYP", "poz"),
    "266249003": ("VH", "HYP", "poz"),
    "67741000119109": ("LAE", "HYP", "poz"),
    "253352002": ("LAA", "HYP", "poz"),
    "195126007": ("AH", "HYP", "poz"),
    "253339007": ("RAAb", "HYP", "poz"),
    "446358003": ("RAH", "HYP", "poz"),
    "55827005": ("LVHV", "HYP", "nej"),  # napäťové kritérium – v PTB-XL tvarový VCLVH
}
# nenamapované kódy, ktoré sa v Georgia vyskytujú často (iba na pomenovanie v rozbore)
OTHER_CODES = {
    "426783006": "NSR",
    "426177001": "SB",
    "427084000": "STach",
    "427393009": "SA",
    "164890007": "AFL",
    "39732003": "LAD",
    "47665007": "RAD",
    "284470004": "PAC",
    "17338001": "VPB",
    "251146004": "LQRSV",
    "428417006": "ERe",
}
SINUS = "426783006"
BENIGN_RHYTHMS = {SINUS, "426177001", "427084000", "427393009"}  # NSR, SB, STach, SA


def georgia_dir() -> Path:
    return paths.DATA_DIR / "raw" / "georgia"


def results_dir() -> Path:
    return paths.PROJECT_ROOT / "docs" / "vysledky" / "externa"


# --- hlavička a labely ----------------------------------------------------------------


def parse_header(text: str) -> dict:
    """Hlavička WFDB z Challenge 2021 → fs, počet vzoriek, kódy Dx, vek, pohlavie."""
    lines = text.strip().splitlines()
    first = lines[0].split()
    info = {"fs": int(float(first[2])), "n_samples": int(first[3]), "dx": [], "age": None,
            "sex": None}  # fmt: skip
    for line in lines:
        if not line.startswith("#"):
            continue
        key, _, value = line[1:].partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "dx":
            info["dx"] = [c.strip() for c in value.split(",") if c.strip()]
        elif key == "age":
            try:
                age = float(value)
            except ValueError:
                age = float("nan")
            info["age"] = None if np.isnan(age) else int(age)
        elif key == "sex":
            info["sex"] = {"male": "male", "female": "female"}.get(value.lower())
    return info


def labels_from_dx(codes: list[str]) -> dict:
    """Kódy SNOMED → {label: 1 / 0 / NaN}, referencia Modelu A a použiteľnosť pre Model B."""
    out = {name: 0.0 for name in LABELS}
    for code in codes:
        if code in MAPPING and MAPPING[code][2] == "poz":
            out[MAPPING[code][1]] = 1.0
    for code in codes:  # nejasný nález → label sa nehodnotí, ak nie je aj jasne pozitívny
        if code in MAPPING and MAPPING[code][2] == "nej" and out[MAPPING[code][1]] == 0.0:
            out[MAPPING[code][1]] = np.nan
    positive = any(out[n] == 1.0 for n in LABELS)
    unclear = {c for c, (_, _, kind) in MAPPING.items() if kind == "nej"}
    # bez pozitívneho labelu musíme vedieť, že ide iba o rytmus bez poruchy (+ nejasné nálezy,
    # ktoré majú pre svoj label NaN); iný nenamapovaný kód = nevieme, záznam sa vynechá
    out["usable_classifier"] = bool(positive or set(codes) <= BENIGN_RHYTHMS | unclear)
    out["reference_a"] = 1.0 if positive else (0.0 if set(codes) == {SINUS} else np.nan)
    return out


def resample_to_model(signal: np.ndarray, fs: int) -> np.ndarray | None:
    """(12, N) pri fs Hz → (12, 1000) pri 100 Hz; kratší záznam ako 10 s → None."""
    from math import gcd

    from scipy.signal import resample_poly

    if signal.shape[1] < contract.DURATION_S * fs:
        return None
    x = signal[:, : contract.DURATION_S * fs]
    if fs != contract.SAMPLING_RATE:
        g = gcd(contract.SAMPLING_RATE, fs)
        x = resample_poly(x, contract.SAMPLING_RATE // g, fs // g, axis=1)
    return x.astype(np.float32)


def read_record(path: Path) -> tuple[np.ndarray, dict]:
    """Záznam Georgia (cesta bez prípony) → signál (12, N) v mV v poradí kontraktu + hlavička."""
    import wfdb

    info = parse_header(Path(f"{path}.hea").read_text(encoding="utf-8"))
    record = wfdb.rdrecord(str(path))
    names = [s.upper() for s in record.sig_name]
    signal = record.p_signal[:, [names.index(lead) for lead in contract.LEADS]].T
    return signal, info


def list_records(root: Path | None = None) -> list[Path]:
    root = Path(root or georgia_dir())
    return sorted(p.with_suffix("") for p in root.rglob("*.hea"))


# --- metriky --------------------------------------------------------------------------


def evaluate(frame: pd.DataFrame, thresholds: np.ndarray, anomaly_threshold: float,
             n_bootstrap: int = N_BOOTSTRAP) -> dict:  # fmt: skip
    """Metriky Modelu A a B ako na PTB-XL teste (#22); bootstrap po záznamoch (Georgia nemá
    identifikátor pacienta)."""
    from ekg.evaluation.clf_report import label_metrics
    from ekg.evaluation.final_evaluation import cluster_bootstrap
    from ekg.evaluation.subgroups import auroc

    groups = np.arange(len(frame))
    ref = frame["reference_a"].to_numpy(float)
    used = ~np.isnan(ref)
    y, s = ref[used] == 1, frame["error"].to_numpy(float)[used]
    d = s >= anomaly_threshold
    model_a = {
        "n_norm": int((~y).sum()),
        "n_abnormal": int(y.sum()),
        "AUROC": auroc(y, s),
        "senzitivita": float(d[y].mean()) if y.any() else None,
        "špecificita": float((~d[~y]).mean()) if (~y).any() else None,
    }
    if n_bootstrap and y.any() and (~y).any():
        model_a["AUROC_ci95"] = cluster_bootstrap(
            groups[used], lambda i: auroc(y[i], s[i]), n=n_bootstrap, seed=SEED
        )

    clf = frame[frame["usable_classifier"]]
    per_label = {}
    for j, name in enumerate(LABELS):
        known = clf[name].notna()
        yt = clf.loc[known, name].to_numpy(int)
        p = clf.loc[known, f"prob_{name}"].to_numpy(float)
        m = label_metrics(yt, p, thresholds[j])
        m["n_unclear"] = int((~known).sum())
        if n_bootstrap and 0 < yt.sum() < len(yt):
            m["AUROC_ci95"] = cluster_bootstrap(
                np.arange(len(yt)), lambda i, yt=yt, p=p: auroc(yt[i], p[i]),
                n=n_bootstrap, seed=SEED + j + 1,
            )  # fmt: skip
        per_label[name] = m
    return {"model_a": model_a, "labels": per_label, "n_classifier": int(len(clf))}


def comparison_with_ptbxl(result: dict) -> pd.DataFrame:
    """Rovnaké metriky na PTB-XL teste (#22) a na Georgia."""
    test = json.loads(
        (paths.PROJECT_ROOT / "docs" / "vysledky" / "test" / "report.json").read_text("utf-8")
    )
    rows = []
    for key in ("AUROC", "senzitivita", "špecificita"):
        rows.append(("Model A", key, test["autoencoder"][key], result["model_a"][key]))
    for name in LABELS:
        for key in ("AUROC", "F1", "recall", "precision", "specificity"):
            rows.append(
                (f"Model B {name}", key, test["classifier"]["labels"][name][key],
                 result["labels"][name].get(key))
            )  # fmt: skip
    frame = pd.DataFrame(rows, columns=["model", "metrika", "PTB-XL test", "Georgia"])
    frame["rozdiel"] = frame["Georgia"] - frame["PTB-XL test"]
    return frame


def mapping_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mapovanie s počtami záznamov v Georgia (všetky stiahnuté záznamy)."""
    counts = pd.Series([c for codes in frame["dx"] for c in codes]).value_counts()
    rows = [
        {"SNOMED": code, "skratka": abbr, "label": label,
         "typ": "pozitívny" if kind == "poz" else "nejasný (nehodnotí sa)",
         "záznamov": int(counts.get(code, 0))}
        for code, (abbr, label, kind) in MAPPING.items()
    ]  # fmt: skip
    return pd.DataFrame(rows).sort_values(["label", "typ", "záznamov"], ascending=[1, 0, 0])


def code_name(code: str) -> str:
    return MAPPING[code][0] if code in MAPPING else OTHER_CODES.get(code, code)


def positivity_by_code(data: pd.DataFrame, thresholds: np.ndarray, min_records: int = 100):
    """Pre každý kód Georgia (≥ min_records záznamov): podiel záznamov, ktoré Model B označí
    daným labelom a Model A ako anomáliu. Ukáže, ktoré diagnózy vyvolávajú falošné poplachy."""
    long = data.assign(code=data["dx"]).explode("code")
    rows = []
    for code, part in long.groupby("code"):
        if len(part) < min_records:
            continue
        mapped = f"{MAPPING[code][1]} ({MAPPING[code][2]})" if code in MAPPING else "–"
        row = {"SNOMED": code, "skratka": code_name(code), "mapovanie": mapped,
               "záznamov": len(part)}  # fmt: skip
        for j, name in enumerate(LABELS):
            row[f"B {name}"] = float((part[f"prob_{name}"] >= thresholds[j]).mean())
        row["A anomália"] = float(part["is_anomaly"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("záznamov", ascending=False)


def domain_shift(pipe, georgia_normal: list[np.ndarray]) -> pd.DataFrame:
    """Rozdiel signálov: prísny NORM PTB-XL (validation) vs. Georgia iba so sínusovým rytmom.
    Medián rozkmitu (mV) po zvodoch a rekonštrukčná chyba autoencodera."""
    from ekg.data import splits
    from ekg.data.cache import SignalCache

    manifest = splits.load_manifest()
    ids = [i for i in splits.split_ids("val", manifest) if manifest.loc[i, "strict_norm"]]
    ptb = SignalCache().get(ids)
    geo = np.stack(georgia_normal)

    def ptp(x):
        return np.median(x.max(axis=2) - x.min(axis=2), axis=0)

    _, err_p, _ = pipe.scores(ptb)
    _, err_g, _ = pipe.scores(geo)
    frame = pd.DataFrame(
        {"PTB-XL rozkmit (mV)": ptp(ptb), "Georgia rozkmit (mV)": ptp(geo),
         "PTB-XL chyba AE": np.median(err_p, axis=0), "Georgia chyba AE": np.median(err_g, axis=0)},
        index=pd.Index(contract.LEADS, name="zvod"),
    )  # fmt: skip
    frame["pomer rozkmitu"] = frame["Georgia rozkmit (mV)"] / frame["PTB-XL rozkmit (mV)"]
    frame["pomer chyby"] = frame["Georgia chyba AE"] / frame["PTB-XL chyba AE"]
    record_p, record_g = err_p.mean(axis=1), err_g.mean(axis=1)
    frame.attrs = {
        "n_ptbxl": len(ptb),
        "n_georgia": len(geo),
        "anomaly_rate_ptbxl": float((record_p >= pipe.anomaly_threshold).mean()),
        "anomaly_rate_georgia": float((record_g >= pipe.anomaly_threshold).mean()),
        "median_error_ptbxl": float(np.median(record_p)),
        "median_error_georgia": float(np.median(record_g)),
    }
    return frame


# --- beh ------------------------------------------------------------------------------


def load_dataset(limit: int | None = None) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Všetky záznamy: hlavička, labely, signál 100 Hz (None = vynechaný)."""
    rows, signals = [], []
    for path in list_records()[:limit]:
        signal, info = read_record(path)
        x, reason = resample_to_model(signal, info["fs"]), None
        if x is None:
            reason = "kratší ako 10 s"
        else:
            try:
                x = contract.validate_signal(x)
            except contract.InvalidInputError as exc:
                x, reason = None, f"neplatný signál: {exc}"
        rows.append({"record": path.name, "length_s": info["n_samples"] / info["fs"],
                     "age": info["age"], "sex": info["sex"], "dx": info["dx"],
                     **labels_from_dx(info["dx"]), "included": x is not None,
                     "exclusion": reason})  # fmt: skip
        signals.append(x)
    return pd.DataFrame(rows).set_index("record"), signals


def main(argv: list[str] | None = None) -> int:
    from ekg import export
    from ekg.inference import Pipeline

    parser = argparse.ArgumentParser(description="Externá validácia na Georgia (#31).")
    parser.add_argument("--limit", type=int, help="iba prvých N záznamov (skúška)")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args(argv)
    if not list_records():
        print(f"Chýba dataset v {georgia_dir()} – postup stiahnutia: docs/externa_validacia.md")
        return 1

    from ekg.evaluation.subgroups import frozen_inputs_unchanged

    problems = frozen_inputs_unchanged()
    if problems:
        print("Modely, prahy alebo dáta sa od zmrazenia (#22) zmenili:\n" + "\n".join(problems))
        return 1

    frame, signals = load_dataset(args.limit)
    keep = frame["included"].to_numpy()
    pipe = Pipeline.load()
    ids = list(frame.index[keep])
    meta = [
        contract.Metadata(
            age=int(frame.loc[i, "age"]) if pd.notna(frame.loc[i, "age"]) else None,
            sex=frame.loc[i, "sex"],
            source=SOURCE,
        )
        for i in ids
    ]
    results = pipe.predict_batch([s for s, k in zip(signals, keep, strict=True) if k], ids, meta)
    pred = pd.DataFrame(
        {
            "error": [r.anomaly.anomaly_score for r in results],
            "is_anomaly": [r.anomaly.is_anomaly for r in results],
            "quality_warning": [r.quality.status == "warning" for r in results],
            **{f"prob_{n}": [r.labels[n].probability for r in results] for n in LABELS},
        },
        index=pd.Index(ids, name="record"),
    )
    data = frame[keep].join(pred)
    result = evaluate(data, pipe.label_thresholds, pipe.anomaly_threshold, args.bootstrap)

    out_docs, out_pred = results_dir(), paths.OUTPUT_DIR / "externa"
    out_docs.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    export.to_csv(results, out_pred / "predikcie_georgia.csv")
    data.drop(columns="dx").assign(dx=data["dx"].map(",".join)).to_csv(
        out_pred / "georgia_tabulka.csv"
    )
    mapping_table(frame).to_csv(out_docs / "mapovanie.csv", index=False)
    positivity_by_code(data, pipe.label_thresholds).to_csv(
        out_docs / "pozitivita_podla_kodu.csv", index=False, float_format="%.3f"
    )
    normal = [s for s, ok, ref in zip(signals, keep, frame["reference_a"], strict=True)
              if ok and ref == 0]  # fmt: skip
    shift = domain_shift(pipe, normal)
    shift.to_csv(out_docs / "posun_domeny.csv", float_format="%.4f")
    comparison = comparison_with_ptbxl(result)
    comparison.to_csv(out_docs / "porovnanie_ptbxl.csv", index=False, float_format="%.4f")
    table = pd.DataFrame(result["labels"]).T.drop(columns="undefined")
    table.index.name = "label"
    table.to_csv(out_docs / "metriky_po_labeloch.csv", float_format="%.4f")
    counts = {
        "záznamov stiahnutých": int(len(frame)),
        "vynechaných": int((~frame["included"]).sum()),
        "vynechaných – kratšie ako 10 s": int((frame["exclusion"] == "kratší ako 10 s").sum()),
        "vynechaných – neplatný signál (NaN)": int(
            frame["exclusion"].fillna("").str.startswith("neplatný").sum()
        ),
        "vyhodnotených": int(keep.sum()),
        "Model B – použiteľných": result["n_classifier"],
        "Model A – iba sínusový rytmus": result["model_a"]["n_norm"],
        "Model A – patológia": result["model_a"]["n_abnormal"],
        "pozitívnych (Model B)": {
            n: int((data.loc[data["usable_classifier"], n] == 1).sum()) for n in LABELS
        },  # fmt: skip
        "varovanie kvality": int(data["quality_warning"].sum()),
    }
    report = {
        "dataset": SOURCE,
        "doi": "10.13026/34va-7q14",
        "license": "CC BY 4.0",
        "resampling": "scipy.signal.resample_poly 500 → 100 Hz, prvých 10 s",
        "model_versions": pipe.model_versions,
        "bootstrap": f"{args.bootstrap}× po záznamoch (bez ID pacienta)",
        "counts": counts,
        "domain_shift": shift.attrs,
        **result,
    }
    from ekg.evaluation.subgroups import _clean

    (out_docs / "report.json").write_text(
        json.dumps(_clean(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(_clean(counts), indent=2, ensure_ascii=False))
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
