"""Výsledky podľa podskupín: vek, pohlavie a kvalita signálu (issue #26).

Opisná analýza ZMRAZENÝCH modelov a prahov (#22) – nič sa tu neladí ani nemení. Skript odmietne
bežať, ak sa od zmrazenia zmenili modely, prahy, manifesty alebo mapovanie labelov.

Zdroj predikcií:
    val   zmrazená pipeline (#19) na všetkých validation záznamoch (CPU, po jednom zázname)
          → outputs/podskupiny/predikcie_val.csv
    test  UŽ ULOŽENÉ predikcie z #22 (docs/vysledky/test/predikcie_test.csv); na test množine sa
          nič znova nepočíta. Z test množiny sa načítajú iba labely a patient_id
          (split_ids("test", allow_test=True)) a overí sa, že sedia s uloženými predikciami.

Podskupiny sú dané vopred, bez pohľadu na výsledky (vekové skupiny rovnaké ako v #18):
    vek                  < 40, 40–59, 60–79, ≥ 80 (vek > 89 je v PTB-XL anonymizovaný → ≥ 80),
                         neznámy
    pohlavie             muž, žena, neznáme
    kvalita (#7)         bez varovania, s varovaním
    typ problému kvality drift základnej línie, statický šum, nárazový šum, problém s elektródami,
                         problém zo signálu (plochý zvod, extrémna amplitúda, saturácia);
                         záznam môže mať viac typov naraz – tieto skupiny sa prekrývajú
Skupiny chýbajúcich údajov (neznámy vek / pohlavie) sú v reporte vždy, aj keď majú 0 záznamov.

Pre každú skupinu: počty záznamov, pacientov a labelov; Model A (AUROC, senzitivita, špecificita
a podiel poplachov na záznamoch bez labelu – prísny NORM podľa #13 vylučuje záznamy so šumom,
takže v skupinách so šumom je to jediný údaj o poplachoch) a Model B (macro AUROC, macro F1,
micro F1, po labeloch AUROC, F1, recall) s 95 % intervalom z bootstrapu po pacientoch. Stav metriky:
    nedefinovaná   chýbajú pozitívne alebo negatívne prípady (hodnota prázdna + dôvod)
    malá skupina   < 30 pozitívnych alebo < 30 negatívnych prípadov (rovnaká hranica ako #17);
                   hodnota je uvedená, ale nedá sa z nej robiť záver
    ok

Spustenie:  uv run ekg-subgroups [--bootstrap 1000]
Výstupy:    docs/vysledky/podskupiny/
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ekg import contract, paths
from ekg.evaluation.clf_report import AGE_BINS, AGE_NAMES

LABELS = list(contract.LABELS)
AE_RUN, CLF_RUN = "ae_v1", "clf_v1"
MIN_CASES = 30  # pod touto hranicou je metrika „malá skupina“ (ako fallback v #17)
N_BOOTSTRAP = 1000
SEED = 26
AGE_UNKNOWN = "neznámy"
SEX_NAMES = {"male": "muž", "female": "žena"}
SEX_UNKNOWN = "neznáme"
QUALITY_TYPES = {
    "drift základnej línie": ("drift základnej línie",),
    "statický šum": ("statický šum",),
    "nárazový šum": ("nárazový šum",),
    "problém s elektródami": ("problém s elektródami",),
    "problém zo signálu": ("plochý zvod", "extrémna amplitúda", "saturácia"),
}
ALL = ("všetky", "všetky")


def results_dir() -> Path:
    return paths.PROJECT_ROOT / "docs" / "vysledky" / "podskupiny"


def test_predictions_path() -> Path:
    return paths.PROJECT_ROOT / "docs" / "vysledky" / "test" / "predikcie_test.csv"


# --- predikcie → tabuľka ------------------------------------------------------------


def frame_from_results(results: list[contract.PredictionResult]) -> pd.DataFrame:
    """Jeden riadok na záznam: skóre a rozhodnutia oboch modelov + metadáta a kvalita."""
    rows = []
    for r in results:
        row = {
            "ecg_id": int(r.record_id),
            "error": r.anomaly.anomaly_score,
            "is_anomaly": r.anomaly.is_anomaly,
            "age": r.metadata.age,
            "age_over_89": bool(r.metadata.age_over_89),
            "sex": r.metadata.sex,
            "quality_status": r.quality.status,
            "quality_issues": list(r.quality.issues),
        }
        for name in LABELS:
            row[f"prob_{name}"] = r.labels[name].probability
            row[f"positive_{name}"] = r.labels[name].positive
        rows.append(row)
    return pd.DataFrame(rows).set_index("ecg_id")


def attach_truth(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Doplní skutočné labely, prísny NORM, použiteľnosť pre Model B a pacienta (manifest #8)."""
    out = frame.copy()
    m = manifest.loc[out.index]
    out[LABELS] = m[LABELS].astype(int).to_numpy()
    out["strict_norm"] = m["strict_norm"].astype(bool).to_numpy()
    out["usable_classifier"] = m["usable_classifier"].astype(bool).to_numpy()
    out["patient_id"] = m["patient_id"].to_numpy()
    return out


# --- podskupiny ---------------------------------------------------------------------


def age_groups(frame: pd.DataFrame) -> pd.Series:
    """Vekové skupiny z #18; vek > 89 (anonymizovaný) patrí do ≥ 80, chýbajúci = neznámy."""
    age = pd.to_numeric(frame["age"], errors="coerce").where(~frame["age_over_89"], 90.0)
    groups = pd.cut(age, bins=list(AGE_BINS), labels=list(AGE_NAMES), right=False)
    return groups.astype(object).where(age.notna(), AGE_UNKNOWN)


def quality_types(issues: list[str]) -> set[str]:
    return {
        name
        for name, prefixes in QUALITY_TYPES.items()
        if any(issue.startswith(prefixes) for issue in issues)
    }


def subgroups(frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(dimenzia, skupina, maska) – poradie je pevné, prázdne skupiny chýbajúcich údajov ostanú."""
    out = [(*ALL, np.ones(len(frame), bool))]
    ages = age_groups(frame).to_numpy()
    for name in (*AGE_NAMES, AGE_UNKNOWN):
        out.append(("vek", name, ages == name))
    sex = frame["sex"].map(SEX_NAMES).fillna(SEX_UNKNOWN).to_numpy()
    for name in (*SEX_NAMES.values(), SEX_UNKNOWN):
        out.append(("pohlavie", name, sex == name))
    warning = (frame["quality_status"] == "warning").to_numpy()
    out.append(("kvalita (#7)", "bez varovania", ~warning))
    out.append(("kvalita (#7)", "s varovaním", warning))
    types = frame["quality_issues"].map(quality_types)
    for name in QUALITY_TYPES:
        out.append(("typ problému kvality", name, types.map(lambda t, n=name: n in t).to_numpy()))
    return out


# --- metriky ------------------------------------------------------------------------


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    """AUROC cez poradia (Mann–Whitney), zhodné so sklearn; None bez oboch tried."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1(y: np.ndarray, decision: np.ndarray) -> float | None:
    """F1; None, ak nie je žiadny pozitívny prípad (bez nich F1 nehovorí nič o rozpoznaní)."""
    y, d = np.asarray(y, bool), np.asarray(decision, bool)
    if not y.any():
        return None
    tp, fp, fn = np.sum(d & y), np.sum(d & ~y), np.sum(~d & y)
    return float(2 * tp / (2 * tp + fp + fn))


def _rate(decision: np.ndarray) -> float | None:
    return float(decision.mean()) if len(decision) else None


def _mean_or_none(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))


@dataclass
class Arrays:
    """Polia jednej skupiny pre rýchly výpočet metrík (aj v bootstrape)."""

    target_a: np.ndarray  # 1 patológia, 0 prísny NORM, NaN vynechaný (#13)
    error: np.ndarray
    anomaly: np.ndarray
    clf: np.ndarray  # záznam patrí Modelu B
    truth: np.ndarray  # (n, 5) bool
    prob: np.ndarray  # (n, 5)
    decision: np.ndarray  # (n, 5) bool
    patient: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Arrays:
        abnormal = frame[LABELS].sum(axis=1).to_numpy() > 0
        target = np.full(len(frame), np.nan)
        target[frame["strict_norm"].to_numpy(bool)] = 0
        target[abnormal] = 1
        return cls(
            target_a=target,
            error=frame["error"].to_numpy(float),
            anomaly=frame["is_anomaly"].to_numpy(bool),
            clf=frame["usable_classifier"].to_numpy(bool),
            truth=frame[LABELS].to_numpy(bool),
            prob=frame[[f"prob_{n}" for n in LABELS]].to_numpy(float),
            decision=frame[[f"positive_{n}" for n in LABELS]].to_numpy(bool),
            patient=frame["patient_id"].to_numpy(),
        )


def metrics(a: Arrays, idx: np.ndarray) -> dict[tuple[str, str, str], float | None]:
    """Všetky metriky pre záznamy idx. Kľúč = (model, metrika, label)."""
    out: dict[tuple[str, str, str], float | None] = {}
    ta = a.target_a[idx]
    used = ~np.isnan(ta)
    y, err, anom = ta[used] == 1, a.error[idx][used], a.anomaly[idx][used]
    out[("A", "AUROC", "")] = auroc(y, err)
    out[("A", "senzitivita", "")] = _rate(anom[y])
    out[("A", "špecificita", "")] = _rate(~anom[~y])
    # poplachy na záznamoch bez labelu (aj mimo prísneho NORM) – jediný údaj o poplachoch tam,
    # kde prísny NORM nie je (napr. záznamy so šumom: prísny NORM šum vylučuje, #13)
    out[("A", "poplachy bez labelu", "")] = _rate(a.anomaly[idx][~a.truth[idx].any(axis=1)])

    c = a.clf[idx]
    truth, prob, dec = a.truth[idx][c], a.prob[idx][c], a.decision[idx][c]
    for j, name in enumerate(LABELS):
        out[("B", "AUROC", name)] = auroc(truth[:, j], prob[:, j])
        out[("B", "F1", name)] = f1(truth[:, j], dec[:, j])
        out[("B", "recall", name)] = _rate(dec[:, j][truth[:, j]])
    out[("B", "macro AUROC", "")] = _mean_or_none([out[("B", "AUROC", n)] for n in LABELS])
    out[("B", "macro F1", "")] = _mean_or_none([out[("B", "F1", n)] for n in LABELS])
    out[("B", "micro F1", "")] = f1(truth.ravel(), dec.ravel())
    return out


# ktoré prípady metrika potrebuje: senzitivita/recall/F1 pozitívne, špecificita negatívne,
# AUROC oboje
NEEDS = {
    "AUROC": ("pos", "neg"),
    "macro AUROC": ("pos", "neg"),
    "senzitivita": ("pos",),
    "recall": ("pos",),
    "F1": ("pos",),
    "macro F1": ("pos",),
    "micro F1": ("pos",),
    "špecificita": ("neg",),
    "poplachy bez labelu": ("neg",),
}


def case_counts(a: Arrays) -> dict[tuple[str, str, str], tuple[int, int]]:
    """(pozitívnych, negatívnych) pre každú metriku – podľa nich sa určuje stav metriky.

    Model A: pozitívne = patológia, negatívne = prísny NORM. Model B: po labeloch; pri macro
    počty najslabšieho labelu, pri micro súčet cez všetky labely."""
    ta = a.target_a
    model_a = (int(np.sum(ta == 1)), int(np.sum(ta == 0)))
    out = {("A", m, ""): model_a for m in ("AUROC", "senzitivita", "špecificita")}
    n_no_label = int((~a.truth.any(axis=1)).sum())
    out[("A", "poplachy bez labelu", "")] = (0, n_no_label)
    truth = a.truth[a.clf]
    pos = truth.sum(axis=0).astype(int)
    neg = (len(truth) - pos).astype(int)
    for j, name in enumerate(LABELS):
        for metric in ("AUROC", "F1", "recall"):
            out[("B", metric, name)] = (int(pos[j]), int(neg[j]))
    least = (int(pos.min()), int(neg.min())) if len(truth) else (0, 0)
    out[("B", "macro AUROC", "")] = least
    out[("B", "macro F1", "")] = least
    out[("B", "micro F1", "")] = (int(pos.sum()), int(neg.sum()))
    return out


def status(key: tuple[str, str, str], value, cases: tuple[int, int], records: int) -> tuple:
    """(stav, dôvod) metriky podľa počtu prípadov, ktoré metrika potrebuje."""
    model, metric, _ = key
    count = dict(zip(("pos", "neg"), cases, strict=True))
    name = {"pos": "pozitívnych", "neg": "negatívnych"}
    if model == "A":
        name = {"pos": "patológií", "neg": "prísnych NORM"}
        if metric == "poplachy bez labelu":
            name = {"neg": "záznamov bez labelu"}
    if value is None:
        if records == 0:
            return "nedefinovaná", "skupina nemá žiadny záznam"
        if metric.startswith("macro"):
            return "nedefinovaná", "aspoň jeden label nemá potrebné prípady"
        missing = next(k for k in NEEDS[metric] if count[k] == 0)
        return "nedefinovaná", f"0 {name[missing]}"
    small = [k for k in NEEDS[metric] if count[k] < MIN_CASES]
    if small:
        prefix = "label s najmenej prípadmi má " if metric.startswith("macro") else ""
        return "malá skupina", prefix + f"< {MIN_CASES} {name[small[0]]}"
    return "ok", ""


def _seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8")) ^ SEED


def bootstrap(a: Arrays, n: int, seed: int) -> dict[tuple[str, str, str], list | None]:
    """95 % interval každej metriky; prevzorkujú sa celí PACIENTI (ako #22).

    Opakovania, v ktorých metrika nie je definovaná, sa vynechajú; ak je ich viac ako polovica,
    interval sa neuvádza (None)."""
    if n == 0 or len(a.patient) == 0:
        return {}
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(a.patient, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inverse))]
    members = [order[bounds[k] : bounds[k + 1]] for k in range(len(unique))]
    samples: dict[tuple[str, str, str], list[float]] = {}
    for _ in range(n):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[k] for k in chosen])
        for key, value in metrics(a, idx).items():
            samples.setdefault(key, [])
            if value is not None:
                samples[key].append(value)
    out = {}
    for key, values in samples.items():
        if len(values) * 2 < n:
            out[key] = None
        else:
            lo, hi = np.quantile(values, [0.025, 0.975])
            out[key] = [float(lo), float(hi)]
    return out


# --- tabuľky ------------------------------------------------------------------------


def group_sizes(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for dim, name, mask in subgroups(frame):
        g = frame[mask]
        a = Arrays.from_frame(g)
        clf = g[g["usable_classifier"]]
        rows.append(
            {
                "množina": split,
                "dimenzia": dim,
                "skupina": name,
                "záznamov": len(g),
                "pacientov": int(g["patient_id"].nunique()),
                "A prísny NORM": int(np.sum(a.target_a == 0)),
                "A patológia": int(np.sum(a.target_a == 1)),
                "A vynechaných": int(np.isnan(a.target_a).sum()),
                "B záznamov": len(clf),
                "B bez labelu": int((clf[LABELS].sum(axis=1) == 0).sum()),
                **{f"B {n}": int(clf[n].sum()) for n in LABELS},
                "malá skupina": len(g) < MIN_CASES,
            }
        )
    return pd.DataFrame(rows)


def group_metrics(frame: pd.DataFrame, split: str, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Dlhá tabuľka: jeden riadok = (skupina, model, metrika, label) s CI, počtami a stavom."""
    rows = []
    for dim, name, mask in subgroups(frame):
        a = Arrays.from_frame(frame[mask])
        point = metrics(a, np.arange(int(mask.sum())))
        cases = case_counts(a)
        ci = bootstrap(a, n_bootstrap, _seed(split, dim, name))
        for key, value in point.items():
            model, metric, label = key
            state, reason = status(key, value, cases[key], int(mask.sum()))
            interval = ci.get(key) if value is not None else None
            rows.append(
                {
                    "množina": split,
                    "dimenzia": dim,
                    "skupina": name,
                    "model": model,
                    "metrika": metric,
                    "label": label,
                    "hodnota": value,
                    "ci95_dolna": interval[0] if interval else None,
                    "ci95_horna": interval[1] if interval else None,
                    "pozitívnych": cases[key][0],
                    "negatívnych": cases[key][1],
                    "stav": state,
                    "dôvod": reason,
                }
            )
    return pd.DataFrame(rows)


def overview(table: pd.DataFrame) -> pd.DataFrame:
    """Prehľad hlavných metrík: skupiny × (množina, metrika) s intervalom v jednej bunke."""
    main = [
        ("A", "AUROC", ""),
        ("A", "senzitivita", ""),
        ("A", "poplachy bez labelu", ""),
        ("B", "macro AUROC", ""),
        ("B", "macro F1", ""),
        ("B", "micro F1", ""),
    ]
    rows = []
    for (dim, name), part in table.groupby(["dimenzia", "skupina"], sort=False):
        row = {"dimenzia": dim, "skupina": name}
        for split in ("val", "test"):
            for model, metric, label in main:
                sel = part[
                    (part["množina"] == split)
                    & (part["model"] == model)
                    & (part["metrika"] == metric)
                    & (part["label"] == label)
                ]
                if sel.empty:
                    continue
                r = sel.iloc[0]
                row[f"{split} {model} {metric}"] = _cell(r)
        rows.append(row)
    return pd.DataFrame(rows)


def _cell(r: pd.Series) -> str:
    if r["stav"] == "nedefinovaná":
        return "–"
    text = f"{r['hodnota']:.3f}"
    if pd.notna(r["ci95_dolna"]):
        text += f" ({r['ci95_dolna']:.3f}–{r['ci95_horna']:.3f})"
    return text + (" *" if r["stav"] == "malá skupina" else "")


def plot_forest(table: pd.DataFrame, sizes: pd.DataFrame, path: Path) -> None:
    """Hlavné metriky oboch modelov po skupinách s 95 % CI; val vs. test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("A", "AUROC", "Model A – AUROC"),
        ("A", "senzitivita", "Model A – senzitivita"),
        ("A", "poplachy bez labelu", "Model A – poplachy bez labelu"),
        ("B", "macro AUROC", "Model B – macro AUROC"),
        ("B", "micro F1", "Model B – micro F1"),
    ]
    # skupiny s < MIN_CASES záznamami v oboch množinách sú iba v tabuľkách (v grafe by ich
    # interval cez celú os prekryl ostatné); prázdne skupiny chýbajúcich údajov ostávajú
    records = sizes.groupby(["dimenzia", "skupina"], sort=False)["záznamov"].max()
    groups = [g for g, n in records.items() if n >= MIN_CASES or n == 0]
    colors = {"val": "#2a78d6", "test": "#eb6834"}  # kategorické farby 1 a 2
    offset = {"val": -0.17, "test": 0.17}
    fig, axes = plt.subplots(1, 5, figsize=(17, 0.36 * len(groups) + 1.8), sharey=True)
    for ax, (model, metric, title) in zip(axes, panels, strict=True):
        sel = table[(table["model"] == model) & (table["metrika"] == metric)]
        for split in ("val", "test"):
            part = sel[sel["množina"] == split].set_index(["dimenzia", "skupina"])
            overall = part.loc[ALL, "hodnota"]
            ax.axvline(overall, color=colors[split], lw=0.8, ls=":", alpha=0.8)
            for k, g in enumerate(groups):
                if g not in part.index or pd.isna(part.loc[g, "hodnota"]):
                    continue
                r = part.loc[g]
                small = r["stav"] == "malá skupina"
                y = k + offset[split]
                if pd.notna(r["ci95_dolna"]):
                    ax.plot([r["ci95_dolna"], r["ci95_horna"]], [y, y], color=colors[split], lw=1.4)
                ax.plot(
                    r["hodnota"],
                    y,
                    "o",
                    ms=5,
                    color="white" if small else colors[split],
                    markeredgecolor=colors[split],
                    label=split if k == 0 else None,
                )
        ax.set_title(title, fontsize=10.5, color="#222222")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.tick_params(colors="#444444", labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    labels = [name if dim == ALL[0] else f"{dim}: {name}" for dim, name in groups]
    axes[0].set_yticks(range(len(groups)), labels, fontsize=8.5)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", color=colors[s], lw=1.4, label=s) for s in ("val", "test")
    ] + [
        plt.Line2D(
            [], [], marker="o", color="grey", mfc="white", lw=0, label=f"< {MIN_CASES} prípadov"
        )
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "Výsledky podľa podskupín (95 % CI z bootstrapu po pacientoch; bodkovane = celá množina;"
        f" chýbajúci bod = nedefinovaná metrika; skupiny s < {MIN_CASES} záznamami iba v tabuľke)",
        fontsize=10,
        x=0.01,
        ha="left",
        color="#222222",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- dáta a beh ---------------------------------------------------------------------


def frozen_inputs_unchanged() -> list[str]:
    """Rozdiely oproti zmrazeniu #22 mimo kódu (modely, prahy, manifesty, mapovanie, uv.lock)."""
    from ekg.evaluation import final_evaluation as fe

    freeze = json.loads((fe.results_dir() / "zmrazenie.json").read_text("utf-8"))
    problems = fe.check_freeze(freeze, fe.post_test_changes())
    return [p for p in problems if " src/" not in p]


def load_val(manifest: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Zmrazená pipeline (#19) na všetkých validation záznamoch; uloží CSV predikcií."""
    from ekg import export
    from ekg.data import ptbxl, splits
    from ekg.data.cache import SignalCache
    from ekg.inference import Pipeline

    ids = splits.split_ids("val", manifest)
    pipe = Pipeline.load(AE_RUN, CLF_RUN)
    db = ptbxl.load_database()
    results = pipe.predict_batch(
        SignalCache().get(ids),
        [str(i) for i in ids],
        [export.metadata_from_ptbxl(db.loc[i]) for i in ids],
        [db.loc[i] for i in ids],
    )
    out.mkdir(parents=True, exist_ok=True)
    export.to_csv(results, out / "predikcie_val.csv")
    return attach_truth(frame_from_results(results), manifest)


def load_test(manifest: pd.DataFrame) -> pd.DataFrame:
    """Uložené predikcie z #22 + skutočné labely. Nová inferencia na test množine sa NEROBÍ."""
    from ekg import export
    from ekg.data import splits

    # #26: test množina sa otvára iba kvôli labelom k už uloženým predikciám z #22
    ids = splits.split_ids("test", manifest, allow_test=True)
    frame = frame_from_results(export.load_csv(test_predictions_path()))
    if set(frame.index) != set(ids):
        raise ValueError("Uložené predikcie z #22 nesedia so záznamami test množiny.")
    return attach_truth(frame, manifest)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    from ekg.data import splits

    parser = argparse.ArgumentParser(description="Výsledky podľa veku, pohlavia a kvality (#26).")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="počet opakovaní")
    args = parser.parse_args(argv)

    problems = frozen_inputs_unchanged()
    if problems:
        print("Modely, prahy alebo dáta sa od zmrazenia (#22) zmenili – analýza sa nespustí:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    manifest = splits.load_manifest()
    frames = {
        "val": load_val(manifest, paths.OUTPUT_DIR / "podskupiny"),
        "test": load_test(manifest),
    }
    sizes = pd.concat([group_sizes(f, s) for s, f in frames.items()], ignore_index=True)
    table = pd.concat(
        [group_metrics(f, s, args.bootstrap) for s, f in frames.items()], ignore_index=True
    )

    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    sizes.to_csv(out / "velkosti_skupin.csv", index=False)
    table.to_csv(out / "metriky.csv", index=False, float_format="%.4f")
    overview(table).to_csv(out / "prehlad.csv", index=False)
    plot_forest(table, sizes, out / "podskupiny.png")
    counts = table["stav"].value_counts().to_dict()
    report = _clean(
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "runs": {"autoencoder": AE_RUN, "classifier": CLF_RUN},
            "source": {
                "val": "zmrazená pipeline #19 na validation (CPU, po jednom zázname)",
                "test": "uložené predikcie z #22 (docs/vysledky/test/predikcie_test.csv)",
            },
            "n_records": {s: len(f) for s, f in frames.items()},
            "n_bootstrap": args.bootstrap,
            "min_cases": MIN_CASES,
            "metric_states": counts,
            "note": "Opisná analýza; modely ani prahy sa nemenili (overené proti zmrazeniu #22).",
        }
    )
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    view = overview(table).set_index(["dimenzia", "skupina"])
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(view.to_string())
    print(f"\nStavy metrík: {counts}\nVýstupy: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
