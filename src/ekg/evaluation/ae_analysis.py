"""Vyhodnotenie detekcie anomálií a analýza chýb Modelu A na VALIDATION (issue #14).

Spustenie:  uv run ekg-ae-analysis [--run ae_v1]

Používa uložený checkpoint a prah (#13) – overí, že prah patrí k checkpointu (SHA256).
Výstupy v outputs/<run>/analyza/:
    metriky.json                 matica zámen, senzitivita, špecificita, precision, F1, AUROC, AUPRC
    detekcia_podla_labelu.csv    koľko záznamov s daným labelom je označených ako anomália
    vynechane_skupiny.csv        ako sa správa model na záznamoch mimo referenčného cieľa
    chyba_po_zvodoch.csv/.png    priemerná chyba po zvodoch pre NORM a každý label
    falosne_poplachy.csv         všetky FP s metadátami
    prehliadnute.csv             všetky FN s labelmi
    priklad_*.png                EKG + rekonštrukcia pre TP, FP, FN, TN
Test množina sa nepoužíva.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ekg import contract, paths
from ekg.anomaly_threshold import confusion, reference_target

LABELS = list(contract.LABELS)
LEADS = list(contract.LEADS)
ERROR_COLS = [f"error_{lead}" for lead in LEADS]


# --- výpočty (čisté funkcie, testované) -------------------------------------


def add_predictions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Doplní stĺpce target (0/1/NaN), pred (0/1) a outcome (TP/FP/TN/FN/vynechaný)."""
    out = frame.copy()
    out["target"] = reference_target(out)
    out["pred"] = (out["error"] >= threshold).astype(int)
    outcome = np.select(
        [
            (out["target"] == 1) & (out["pred"] == 1),
            (out["target"] == 0) & (out["pred"] == 1),
            (out["target"] == 0) & (out["pred"] == 0),
            (out["target"] == 1) & (out["pred"] == 0),
        ],
        ["TP", "FP", "TN", "FN"],
        default="vynechaný",
    )
    out["outcome"] = outcome
    return out


def summary_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    ref = frame[frame["target"].notna()]
    y, s = ref["target"].to_numpy(int), ref["error"].to_numpy()
    m = confusion(y, s, threshold)
    m.update(
        {
            "AUROC": float(roc_auc_score(y, s)),
            "AUPRC": float(average_precision_score(y, s)),
            "n_strict_norm": int((y == 0).sum()),
            "n_abnormal": int((y == 1).sum()),
            "n_excluded": int(frame["target"].isna().sum()),
            "threshold": threshold,
        }
    )
    return {k: (int(v) if isinstance(v, int | np.integer) else float(v)) for k, v in m.items()}


def detection_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Podiel záznamov označených ako anomália pre každý label.

    'všetky' = záznamy, ktoré label majú (môžu mať aj iné),
    'iba tento label' = záznamy s jediným labelom – čistejší pohľad na konkrétnu poruchu.
    """
    n_labels = frame[LABELS].sum(axis=1)
    rows = {}
    for label in LABELS:
        has = frame[label] == 1
        only = has & (n_labels == 1)
        rows[label] = {
            "záznamy": int(has.sum()),
            "zachytené": float(frame.loc[has, "pred"].mean()),
            "medián chyby": float(frame.loc[has, "error"].median()),
            "iba tento label – záznamy": int(only.sum()),
            "iba tento label – zachytené": float(frame.loc[only, "pred"].mean())
            if only.any()
            else float("nan"),
        }
    multi = n_labels >= 2
    rows["2 a viac labelov"] = {
        "záznamy": int(multi.sum()),
        "zachytené": float(frame.loc[multi, "pred"].mean()),
        "medián chyby": float(frame.loc[multi, "error"].median()),
        "iba tento label – záznamy": 0,
        "iba tento label – zachytené": float("nan"),
    }
    return pd.DataFrame(rows).T


def excluded_groups(frame: pd.DataFrame, reasons: pd.Series) -> pd.DataFrame:
    """Záznamy mimo referenčného cieľa podľa dôvodu, prečo nie sú prísny NORM."""
    excl = frame[frame["target"].isna()]
    reason = reasons.reindex(excl.index).fillna("?")
    grouped = excl.groupby(reason)
    return pd.DataFrame(
        {
            "záznamy": grouped.size(),
            "označené ako anomália": grouped["pred"].mean(),
            "medián chyby": grouped["error"].median(),
        }
    ).sort_values("záznamy", ascending=False)


def lead_error_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Medián chyby po zvodoch pre prísny NORM a pre každý label (riadky) × zvody (stĺpce)."""
    groups = {"prísny NORM": frame["strict_norm"].astype(bool)}
    groups.update({label: frame[label] == 1 for label in LABELS})
    table = pd.DataFrame(
        {name: frame.loc[mask, ERROR_COLS].median() for name, mask in groups.items()}
    ).T
    table.columns = LEADS
    return table


MI_CODES = ("IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "IPLMI", "IPMI", "PMI")
MI_SUBTYPES = {"inferiórny MI (IMI)": "IMI", "anteroseptálny MI (ASMI)": "ASMI"}


def mi_localization(frame: pd.DataFrame, scp_codes: pd.Series, top: int = 4) -> pd.DataFrame:
    """Pre záznamy s jediným typom MI: zvody s najväčšou chybou (× medián prísneho NORM)."""
    norm = frame.loc[frame["strict_norm"].astype(bool), ERROR_COLS].median()
    # záznam bez SCP kódov (NaN po reindexe) = prázdny slovník
    codes = scp_codes.reindex(frame.index).apply(lambda d: d if isinstance(d, dict) else {})
    rows = {}
    for name, code in MI_SUBTYPES.items():
        only = codes.apply(lambda d, c=code: c in d and not any(m in d for m in MI_CODES if m != c))
        ratio = (frame.loc[only, ERROR_COLS].median() / norm).set_axis(LEADS)
        best = ratio.sort_values(ascending=False).head(top)
        rows[name] = {
            "záznamy": int(only.sum()),
            "zvody s najväčšou chybou": ", ".join(f"{k} {v:.1f}×" for k, v in best.items()),
        }
    return pd.DataFrame(rows).T


# --- grafy -------------------------------------------------------------------


def plot_lead_profile(profile: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio = profile / profile.loc["prísny NORM"]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(ratio.to_numpy(), cmap="Reds", aspect="auto", vmin=1)
    ax.set_xticks(range(len(LEADS)), LEADS)
    ax.set_yticks(range(len(ratio)), ratio.index)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            ax.text(j, i, f"{ratio.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(
        "Medián rekonštrukčnej chyby po zvodoch – koľkokrát viac ako prísny NORM (validation)"
    )
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_example(ecg_id, x, recon, err_time, info: str, path) -> None:
    """12 zvodov: originál vs. rekonštrukcia (mV) + chyba v čase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(contract.N_SAMPLES) / contract.SAMPLING_RATE
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(7, 2, height_ratios=[1] * 6 + [0.8])
    for i, lead in enumerate(LEADS):
        ax = fig.add_subplot(grid[i % 6, i // 6])
        ax.plot(t, x[i], color="black", lw=0.7, label="originál")
        ax.plot(t, recon[i], color="tab:red", lw=0.7, alpha=0.8, label="rekonštrukcia")
        ax.set_ylabel(lead, rotation=0, labelpad=15, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    ax = fig.add_subplot(grid[6, :])
    ax.fill_between(t, err_time, color="tab:red", alpha=0.5)
    ax.set(xlabel="čas [s]", ylabel="chyba")
    ax.grid(alpha=0.2)
    fig.suptitle(f"ecg_id {ecg_id} – {info}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --- hlavný beh --------------------------------------------------------------


def pick_examples(frame: pd.DataFrame) -> dict[str, int]:
    """Typické príklady: TP s najvyššou chybou, FP s najvyššou chybou,
    FN s najnižšou chybou, TN s mediánovou chybou."""
    ex = {}
    for outcome, how in [("TP", "max"), ("FP", "max"), ("FN", "min"), ("TN", "median")]:
        sub = frame[frame["outcome"] == outcome]
        if sub.empty:
            continue
        if how == "median":
            ex[outcome] = int((sub["error"] - sub["error"].median()).abs().idxmin())
        else:
            ex[outcome] = int(getattr(sub["error"], f"idx{how}")())
    return ex


def main(argv: list[str] | None = None) -> int:
    import torch

    from ekg.anomaly_threshold import load_val_errors
    from ekg.data import labels, ptbxl
    from ekg.data.cache import SignalCache
    from ekg.models.autoencoder import ConvAutoencoder
    from ekg.models.recon_error import error_over_time
    from ekg.preprocessing import Preprocessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ae_v1")
    args = parser.parse_args(argv)

    run_dir = paths.CHECKPOINT_DIR / args.run
    th_data = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    if th_data["checkpoint_sha256"] != digest:
        print("CHYBA: threshold.json nepatrí k tomuto checkpointu.")
        return 1
    threshold = th_data["threshold"]

    frame, _ = load_val_errors(args.run, th_data["aggregation"])
    frame = add_predictions(frame, threshold)
    db = ptbxl.load_database()
    frame[["age", "sex"]] = db.loc[frame.index, ["age", "sex"]]
    reasons = labels.norm_exclusion_reason(db, ptbxl.load_scp_statements())

    out = paths.OUTPUT_DIR / args.run / "analyza"
    out.mkdir(parents=True, exist_ok=True)

    metrics = summary_metrics(frame, threshold)
    (out / "metriky.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    det = detection_by_label(frame)
    det.to_csv(out / "detekcia_podla_labelu.csv")
    excl = excluded_groups(frame, reasons)
    excl.to_csv(out / "vynechane_skupiny.csv")
    profile = lead_error_profile(frame)
    profile.to_csv(out / "chyba_po_zvodoch.csv")
    plot_lead_profile(profile, out / "chyba_po_zvodoch.png")
    mi_loc = mi_localization(frame, db["scp_codes"])
    mi_loc.to_csv(out / "lokalizacia_mi.csv")

    cols = ["error", "age", "sex", *LABELS, *ERROR_COLS]
    fp = frame[frame["outcome"] == "FP"].sort_values("error", ascending=False)
    fn = frame[frame["outcome"] == "FN"].sort_values("error")
    fp[cols].assign(najhorsi_zvod=fp[ERROR_COLS].idxmax(axis=1).str[6:]).to_csv(
        out / "falosne_poplachy.csv"
    )
    fn[cols].to_csv(out / "prehliadnute.csv")

    # príklady EKG s rekonštrukciou (v mV – inverzia normalizácie)
    model, _ = ConvAutoencoder.load(run_dir / "best.pt")
    prep, cache = Preprocessor.load(), SignalCache()
    mean = np.asarray(prep.mean)[:, None]
    std = np.asarray(prep.std)[:, None]
    for outcome, ecg_id in pick_examples(frame).items():
        x = torch.from_numpy(prep.transform(cache.get([ecg_id])))
        with torch.no_grad():
            recon = model(x)
        err_t = error_over_time(x, recon)[0].numpy()
        x_mv, r_mv = x[0].numpy() * std + mean, recon[0].numpy() * std + mean
        row = frame.loc[ecg_id]
        present = [label for label in LABELS if row[label] == 1] or ["prísny NORM"]
        info = (
            f"{outcome}: {', '.join(present)}, chyba {row['error']:.4f} "
            f"({'≥' if row['pred'] else '<'} prah {threshold:.4f})"
        )
        plot_example(ecg_id, x_mv, r_mv, err_t, info, out / f"priklad_{outcome}.png")

    # --- výpis ---
    print(f"Model {args.run}, prah {threshold:.5f} ({th_data['rule']}), validation\n")
    print(
        f"Referenčný cieľ: {metrics['n_strict_norm']} prísny NORM, "
        f"{metrics['n_abnormal']} s patológiou, {metrics['n_excluded']} vynechaných"
    )
    print(
        f"Matica zámen:  TP {metrics['TP']}  FP {metrics['FP']}  "
        f"TN {metrics['TN']}  FN {metrics['FN']}\n"
        f"senzitivita {metrics['senzitivita']:.3f}  špecificita {metrics['špecificita']:.3f}  "
        f"precision {metrics['precision']:.3f}  F1 {metrics['F1']:.3f}  "
        f"AUROC {metrics['AUROC']:.3f}  AUPRC {metrics['AUPRC']:.3f}\n"
    )
    print("Detekcia podľa labelu:")
    print(det.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nVynechané záznamy podľa dôvodu (nie sú v metrikách):")
    print(excl.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nMedián chyby po zvodoch (× prísny NORM):")
    print((profile / profile.loc["prísny NORM"]).to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nLokalizácia MI (iba jeden typ MI v zázname):")
    print(mi_loc.to_string())
    tn = frame[frame["outcome"] == "TN"]
    worst = fp[ERROR_COLS].idxmax(axis=1).str[6:].value_counts().head(3).to_dict()
    print(
        f"\nFalošné poplachy ({len(fp)}): vek medián {fp['age'].median():.0f} "
        f"(TN {tn['age'].median():.0f}), ženy {(fp['sex'] == 1).mean():.0%} "
        f"(TN {(tn['sex'] == 1).mean():.0%}), najčastejší najhorší zvod: {worst}"
    )
    print(f"Prehliadnuté ({len(fn)}): labely {fn[LABELS].sum().astype(int).to_dict()}")
    print(f"\nVýstupy: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
