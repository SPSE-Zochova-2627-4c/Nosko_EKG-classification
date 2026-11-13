"""Prieskum datasetu PTB-XL a zastúpenia labelov (issue #5).

Spustenie:  uv run ekg-explore

Všetky čísla počíta z dát a manifestov (mapovanie labelov v1.0, rozdelenie v1.0, kvalita v1.0),
takže report je reprodukovateľný. Výstupy:
    outputs/prieskum/*.csv, *.json     tabuľky
    outputs/prieskum/*.png             grafy (kopírujú sa do docs/obrazky/prieskum_*.png)
"""

from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from ekg import contract, paths

LABELS = list(contract.LABELS)
GROUPS = ["NORM", *LABELS]  # NORM = kód NORM v anotáciách (akákoľvek likelihood)
SPLITS = ["train", "val", "test"]
ANONYMIZED_AGE = 300


# --- tabuľka záznamov ---------------------------------------------------------


def build_table(db: pd.DataFrame, manifest: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Jeden riadok na záznam: split, labely, NORM, vek, pohlavie, kvalita."""
    t = manifest[["patient_id", "split", *LABELS, "usable_classifier", "strict_norm"]].copy()
    t["NORM"] = db.loc[t.index, "scp_codes"].apply(lambda d: int("NORM" in d))
    age = db.loc[t.index, "age"]
    t["age_over_89"] = age == ANONYMIZED_AGE
    t["age"] = age.where(~t["age_over_89"])
    t["sex"] = db.loc[t.index, "sex"].map({0: "muž", 1: "žena"})
    t["validated_by_human"] = db.loc[t.index, "validated_by_human"].astype(bool)
    t["quality"] = quality.loc[t.index, "status"]
    t["n_labels"] = t[LABELS].sum(axis=1)
    return t


# --- výpočty -----------------------------------------------------------------


def basic_counts(t: pd.DataFrame, db: pd.DataFrame) -> dict:
    per_patient = t.groupby("patient_id").size()
    dates = pd.to_datetime(db["recording_date"])
    return {
        "záznamy": len(t),
        "pacienti": int(t["patient_id"].nunique()),
        "záznamov na pacienta (max)": int(per_patient.max()),
        "pacienti s 1 záznamom": int((per_patient == 1).sum()),
        # pozor: popis PTB-XL uvádza 10/1989–6/1996, recording_date má širší rozsah (docs)
        "recording_date (rozsah v CSV)": f"{dates.min().date()} – {dates.max().date()}",
        "typy prístrojov": int(db["device"].nunique()),
        "overené človekom (validated_by_human)": int(t["validated_by_human"].sum()),
        "mapovanie labelov": t.attrs.get("mapping_version", "1.0"),
    }


def missing_values(t: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "vek chýba": int(t["age"].isna().sum() - t["age_over_89"].sum()),
        "vek > 89 (PTB-XL uvádza 300)": int(t["age_over_89"].sum()),
        "pohlavie chýba": int(t["sex"].isna().sum()),
        "výška chýba": int(db["height"].isna().sum()),
        "hmotnosť chýba": int(db["weight"].isna().sum()),
        "bez diagnostického kódu aj AFIB (vynechané z Modelu B)": int(
            (~t["usable_classifier"]).sum()
        ),
    }
    out = pd.DataFrame({"záznamy": rows})
    out["podiel"] = out["záznamy"] / len(t)
    return out


def label_counts(t: pd.DataFrame) -> pd.DataFrame:
    """Počet a podiel pozitívnych záznamov po množinách (menovateľ = všetky záznamy množiny)."""
    rows = {}
    for g in [*GROUPS, "strict_norm"]:
        col = t[g].astype(int)
        row = {}
        for s in [*SPLITS, "spolu"]:
            sub = col if s == "spolu" else col[t["split"] == s]
            row[s] = int(sub.sum())
            row[f"{s} %"] = 100 * sub.mean()
        rows["prísny NORM" if g == "strict_norm" else g] = row
    table = pd.DataFrame(rows).T
    counts = [*SPLITS, "spolu"]
    table[counts] = table[counts].astype(int)
    return table


def labels_per_record(t: pd.DataFrame) -> pd.Series:
    """Koľko z 5 labelov má záznam (0 = NORM alebo bez labelu)."""
    return t["n_labels"].clip(upper=4).value_counts().sort_index().rename(index={4: "4–5"})


def cooccurrence(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Počty súbehov a podmienená pravdepodobnosť P(stĺpec | riadok)."""
    m = t[GROUPS].astype(int)
    counts = m.T @ m
    conditional = counts.div(np.diag(counts), axis=0)
    return counts, conditional


def top_combinations(t: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Najčastejšie kombinácie labelov (vrátane NORM a „bez labelu“)."""

    def name(row):
        present = [g for g in GROUPS if row[g]]
        return " + ".join(present) if present else "bez labelu"

    combo = t[GROUPS].apply(name, axis=1)
    out = combo.value_counts().head(n).to_frame("záznamy")
    out["podiel %"] = 100 * out["záznamy"] / len(t)
    return out


def demographics_by_label(t: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for g in [*GROUPS, "všetky"]:
        sub = t if g == "všetky" else t[t[g] == 1]
        rows[g] = {
            "záznamy": len(sub),
            "vek medián": sub["age"].median(),
            "vek > 89 %": 100 * sub["age_over_89"].mean(),
            "ženy %": 100 * (sub["sex"] == "žena").mean(),
            "kvalita warning %": 100 * (sub["quality"] == "warning").mean(),
        }
    return pd.DataFrame(rows).T


def pair_overlaps(counts: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Dvojice labelov zoradené podľa počtu spoločných záznamov."""
    pairs = [(a, b, int(counts.loc[a, b])) for a, b in combinations(GROUPS, 2)]
    return sorted(pairs, key=lambda p: -p[2])


# --- grafy -------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_labels(lc: pd.DataFrame, path) -> None:
    plt = _plt()
    names = [*GROUPS, "prísny NORM"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(
        names, lc.loc[names, "spolu"], color=["tab:green"] + ["tab:red"] * 5 + ["tab:olive"]
    )
    for b, pct in zip(bars, lc.loc[names, "spolu %"], strict=True):
        ax.annotate(
            f"{pct:.1f} %",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        ylabel="počet záznamov",
        title=f"Zastúpenie labelov v PTB-XL (n = {int(lc.attrs['n'])} záznamov)",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cooccurrence(conditional: pd.DataFrame, counts: pd.DataFrame, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(conditional.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(GROUPS)), GROUPS)
    ax.set_yticks(range(len(GROUPS)), GROUPS)
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            v = conditional.iat[i, j]
            ax.text(
                j,
                i,
                f"{100 * v:.0f} %\n({counts.iat[i, j]})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v > 0.6 else "black",
            )
    ax.set(
        xlabel="… má aj tento label",
        ylabel="záznam s labelom …",
        title="Súbehy labelov: P(stĺpec | riadok)",
    )
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_labels_per_record(lpr: pd.Series, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in lpr.index], lpr.to_numpy(), color="tab:purple")
    for i, v in enumerate(lpr.to_numpy()):
        ax.annotate(f"{v}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set(
        xlabel="počet labelov z (MI, STTC, CD, HYP, AFIB)",
        ylabel="záznamy",
        title="Koľko porúch má jeden záznam",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_age_sex(t: pd.DataFrame, path) -> None:
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(0, 95, 5)
    for sex, color in [("muž", "tab:blue"), ("žena", "tab:pink")]:
        a.hist(t.loc[t["sex"] == sex, "age"].dropna(), bins=bins, alpha=0.6, label=sex, color=color)
    a.set(
        xlabel="vek [roky] (bez 293 záznamov s vekom > 89)",
        ylabel="záznamy",
        title="Vek a pohlavie",
    )
    a.legend()
    a.grid(alpha=0.3)
    data = [t.loc[t[g] == 1, "age"].dropna() for g in GROUPS]
    b.boxplot(data, tick_labels=GROUPS, showfliers=False)
    b.set(ylabel="vek [roky]", title="Vek podľa labelu")
    b.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- hlavný beh --------------------------------------------------------------


def main() -> int:
    from ekg.data import labels, ptbxl, splits
    from ekg.quality import QUALITY_VERSION

    db = ptbxl.load_database()
    manifest = splits.load_manifest()
    quality = pd.read_csv(
        paths.PROJECT_ROOT / "manifests" / f"quality_v{QUALITY_VERSION}.csv", index_col="ecg_id"
    )
    t = build_table(db, manifest, quality)
    t.attrs["mapping_version"] = labels.MAPPING_VERSION

    out = paths.OUTPUT_DIR / "prieskum"
    out.mkdir(parents=True, exist_ok=True)
    basic = basic_counts(t, db)
    miss = missing_values(t, db)
    lc = label_counts(t)
    lc.attrs["n"] = len(t)
    lpr = labels_per_record(t)
    counts, cond = cooccurrence(t)
    combos = top_combinations(t)
    demo = demographics_by_label(t)
    quality_split = pd.crosstab(t["quality"], t["split"])[SPLITS]

    (out / "zakladne_pocty.json").write_text(json.dumps(basic, ensure_ascii=False, indent=2) + "\n")
    for name, table in [
        ("chybajuce_hodnoty", miss),
        ("labely", lc),
        ("labely_na_zaznam", lpr.to_frame("záznamy")),
        ("subehy_pocty", counts),
        ("subehy_podmienene", cond),
        ("kombinacie", combos),
        ("demografia_podla_labelu", demo),
        ("kvalita_podla_mnoziny", quality_split),
    ]:
        table.to_csv(out / f"{name}.csv")
    plot_labels(lc, out / "labely.png")
    plot_cooccurrence(cond, counts, out / "subehy.png")
    plot_labels_per_record(lpr, out / "labely_na_zaznam.png")
    plot_age_sex(t, out / "vek_pohlavie.png")

    fmt = {"float_format": lambda v: f"{v:.1f}"}
    print("Základné počty:")
    for k, v in basic.items():
        print(f"  {k}: {v}")
    print("\nChýbajúce hodnoty:\n" + miss.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nLabely (menovateľ = všetky záznamy množiny):\n" + lc.to_string(**fmt))
    print("\nPočet labelov na záznam:\n" + lpr.to_string())
    print(
        "\nNajčastejšie súbehy dvojíc:",
        ", ".join(f"{a}+{b} {n}" for a, b, n in pair_overlaps(counts)[:6]),
    )
    print("\nNajčastejšie kombinácie:\n" + combos.to_string(**fmt))
    print("\nDemografia a kvalita podľa labelu:\n" + demo.to_string(**fmt))
    print("\nKvalita podľa množiny:\n" + quality_split.to_string())
    print(f"\nVýstupy: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
