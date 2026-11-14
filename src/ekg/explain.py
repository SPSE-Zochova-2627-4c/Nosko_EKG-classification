"""Na čo sa modely pozerajú: mapy dôležitosti (Model B) a chyba v čase (Model A) – issue #27.

Mapy ukazujú, ktoré časti VSTUPU ovplyvnili výstup modelu. Nie sú to lekárske vysvetlenia
a nehovoria, či je rozhodnutie správne.

Model B (klasifikátor), pre každý label zvlášť:
    Integrated Gradients (Sundararajan et al. 2017)
        atribúcia každej vzorky každého zvodu (12 × 1000). Integruje gradient logitu labelu po
        priamke od základnej línie (nulový signál po normalizácii = priemer zvodu z train, teda
        „rovná čiara“) k záznamu. Kladná hodnota = táto časť zvyšuje logit labelu.
        Kontrola úplnosti: súčet atribúcií = logit(záznam) − logit(základná línia); odchýlka
        sa vracia v `ig_delta` (relatívne k |rozdielu logitov|).
    Grad-CAM 1D (Selvaraju et al. 2017)
        posledná konvolučná vrstva (256 kanálov × 63 časových bodov ≈ 160 ms na bod): váha kanála
        = priemerný gradient logitu, mapa = ReLU(Σ váha · aktivácia), roztiahnutá na 1000 vzoriek
        a normovaná na [0, 1]. Iba ČAS – zvody sa v sieti miešajú už v prvej vrstve.
Model A (autoencoder):
    štvorcová chyba rekonštrukcie (x − x̂)² v každej vzorke každého zvodu v normalizovanom
    priestore – priemer cez všetko je presne anomaly score (#12).

Zobrazenie: farebné pozadie za signálom každého zvodu (vyhladené kĺzavým priemerom 0,1 s,
normované 99. percentilom cez všetky zvody, takže sa zvody dajú porovnať), nad zvodmi pás
s Grad-CAM (Model B) alebo chybou v čase (Model A).

Spustenie:  uv run ekg-explain --ecg-id 127 [--label STTC] [-o priečinok]
            uv run ekg-explain examples/00017_lr
            uv run ekg-explain --check     kontrola máp na validation (lokalizácia MI, porovnanie
                                           s náhodným modelom) → docs/vysledky/vysvetlenie/
Test množina je zamknutá rovnako ako v ekg-predict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ekg import contract

LABELS = list(contract.LABELS)
LEADS = list(contract.LEADS)
IG_STEPS = 256  # pri 64 – 128 krokoch odchýlka úplnosti 11 – 26 %, pri 256 do 4 % (#27)
SMOOTH_SAMPLES = 10  # 0,1 s pri 100 Hz
MODEL_A = "Model A"
INFERIOR = ("II", "III", "AVF")
ANTERIOR = ("V1", "V2", "V3", "V4")


@dataclass
class Explanation:
    record_id: str
    signal_mv: np.ndarray  # (12, 1000) po filtrovaní, v mV – na zobrazenie
    probabilities: dict[str, float]
    attributions: dict[str, np.ndarray]  # label → (12, 1000), Integrated Gradients (so znamienkom)
    gradcam: dict[str, np.ndarray]  # label → (1000,) v [0, 1]
    ig_delta: dict[str, float]  # relatívna odchýlka úplnosti IG
    recon_error: np.ndarray  # (12, 1000) štvorcová chyba autoencodera
    anomaly_score: float


# --- metódy ---------------------------------------------------------------------------


def _upsample(cam: np.ndarray, n: int = contract.N_SAMPLES) -> np.ndarray:
    """Mapu dĺžky L (stredy úsekov) lineárne roztiahne na n vzoriek."""
    centers = (np.arange(len(cam)) + 0.5) * n / len(cam)
    return np.interp(np.arange(n) + 0.5, centers, cam)


def gradcam(model, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM pre všetky labely: x (1, 12, 1000) normalizovaný → (5, 1000) v [0, 1]."""
    model.eval()
    with torch.enable_grad():
        features = model.features(x)
        pooled = torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)
        logits = model.head(pooled)[0]
        maps = []
        for j in range(len(LABELS)):
            (grad,) = torch.autograd.grad(logits[j], features, retain_graph=True)
            weights = grad.mean(dim=2, keepdim=True)  # (1, C, 1)
            cam = torch.relu((weights * features).sum(dim=1))[0].detach().numpy()
            top = cam.max()
            maps.append(_upsample(cam / top if top > 0 else cam))
    return np.stack(maps)


def integrated_gradients(
    model, x: torch.Tensor, steps: int = IG_STEPS
) -> tuple[np.ndarray, np.ndarray]:
    """Integrated Gradients pre všetky labely so základnou líniou 0 (priemer zvodu).

    x (1, 12, 1000) normalizovaný → (atribúcie (5, 12, 1000), relatívna odchýlka úplnosti (5,)).
    Integrál sa počíta stredovým pravidlom v `steps` bodoch (jedna dávka).
    """
    model.eval()
    baseline = torch.zeros_like(x)
    alphas = ((torch.arange(steps, dtype=x.dtype) + 0.5) / steps)[:, None, None]
    with torch.enable_grad():
        path = (baseline + alphas * (x - baseline)).requires_grad_(True)  # (S, 12, 1000)
        logits = model(path)
        attrs = []
        for j in range(len(LABELS)):
            (grad,) = torch.autograd.grad(logits[:, j].sum(), path, retain_graph=True)
            attrs.append(((x - baseline)[0] * grad.mean(dim=0)).detach().numpy())
    attrs = np.stack(attrs)
    with torch.no_grad():
        diff = (model(x) - model(baseline))[0].numpy()
    delta = np.abs(attrs.sum(axis=(1, 2)) - diff) / np.maximum(np.abs(diff), 1e-6)
    return attrs, delta


def explain(pipe, signal: np.ndarray, record_id: str = "", steps: int = IG_STEPS) -> Explanation:
    """Mapy pre jeden záznam (12, 1000) v mV so zmrazenými modelmi pipeline (#19)."""
    from ekg.models.recon_error import squared_error

    x_mv = contract.validate_signal(signal)
    xn = torch.from_numpy(pipe.preprocessor.transform(x_mv[None]))
    threads = torch.get_num_threads()
    torch.set_num_threads(max(1, min(threads, 4)))  # dávka IG; pri 16 vláknach je CPU pomalšie
    try:
        clf = pipe.classifier.cpu()
        attrs, delta = integrated_gradients(clf, xn, steps)
        cams = gradcam(clf, xn)
        with torch.no_grad():
            probs = torch.sigmoid(clf(xn))[0].numpy()
            error = squared_error(xn, pipe.autoencoder.cpu()(xn))[0].numpy()
    finally:
        torch.set_num_threads(threads)
    mean = np.asarray(pipe.preprocessor.mean, dtype=np.float32)[:, None]
    std = np.asarray(pipe.preprocessor.std, dtype=np.float32)[:, None]
    return Explanation(
        record_id=record_id,
        signal_mv=xn.numpy()[0] * std + mean,
        probabilities={n: float(p) for n, p in zip(LABELS, probs, strict=True)},
        attributions={n: attrs[j] for j, n in enumerate(LABELS)},
        gradcam={n: cams[j] for j, n in enumerate(LABELS)},
        ig_delta={n: float(delta[j]) for j, n in enumerate(LABELS)},
        recon_error=error,
        anomaly_score=float(error.mean()),
    )


# --- súhrny ---------------------------------------------------------------------------


def smooth(values: np.ndarray, window: int = SMOOTH_SAMPLES) -> np.ndarray:
    """Kĺzavý priemer po poslednej osi (rovnaká dĺžka, okraje bez posunu)."""
    if window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    pad = window // 2
    padded = np.pad(flat, ((0, 0), (pad, window - 1 - pad)), mode="edge")
    out = np.stack([np.convolve(row, kernel, mode="valid") for row in padded])
    return out.reshape(values.shape)


def lead_shares(attribution: np.ndarray) -> dict[str, float]:
    """Podiel kladnej atribúcie v každom zvode (súčet = 1; rovnomerne = 1/12 = 0,083)."""
    positive = np.clip(attribution, 0, None).sum(axis=1)
    total = positive.sum()
    shares = positive / total if total > 0 else np.full(len(LEADS), 1 / len(LEADS))
    return {lead: float(s) for lead, s in zip(LEADS, shares, strict=True)}


def heat(values: np.ndarray, positive_only: bool = True) -> np.ndarray:
    """Na zobrazenie: kladná časť, vyhladená, normovaná 99. percentilom celého záznamu → [0, 1]."""
    v = smooth(np.clip(values, 0, None) if positive_only else np.abs(values))
    top = np.percentile(v, 99)
    return np.clip(v / top, 0, 1) if top > 0 else np.zeros_like(v)


def time_profile(attribution: np.ndarray) -> np.ndarray:
    """Kladná atribúcia sčítaná cez zvody, vyhladená a normovaná maximom → (1000,) v [0, 1]."""
    v = smooth(np.clip(attribution, 0, None).sum(axis=0))
    top = v.max()
    return v / top if top > 0 else v


def view(expl: Explanation, target: str) -> dict:
    """Čo sa kreslí pre `target` (label alebo MODEL_A): pozadie zvodov, pás nad nimi, texty."""
    if target == MODEL_A:
        return {
            "heat": heat(expl.recon_error),
            "strip": smooth(expl.recon_error.mean(axis=0)),
            "strip2": None,
            "strip_label": "chyba v čase (priemer cez zvody, vyhladená)",
            "title": (
                f"Model A – rekonštrukčná chyba v čase (anomaly score {expl.anomaly_score:.4f})"
            ),
            "legend": "chyba (x − x̂)², vyhladená, relatívne k 99. percentilu",
            "cmap": "Oranges",
        }
    if target not in LABELS:
        raise ValueError(f"Neznámy cieľ {target!r}: {[MODEL_A, *LABELS]}")
    return {
        "heat": heat(expl.attributions[target]),
        "strip": expl.gradcam[target],
        "strip2": time_profile(expl.attributions[target]),
        "strip_label": "Grad-CAM (výplň) a Integrated Gradients sčítané cez zvody (čiara), 0–1",
        "title": (
            f"Model B – {target} (P = {expl.probabilities[target]:.3f}): "
            "Integrated Gradients po zvodoch, Grad-CAM v čase"
        ),
        "legend": "kladná atribúcia (zvyšuje P), vyhladená, relatívne k 99. percentilu",
        "cmap": "Reds",
    }


def top_leads(attribution: np.ndarray, n: int = 3) -> str:
    """Napr. 'V1 28 %, V6 9 %, aVR 9 %' – zvody s najväčším podielom kladnej atribúcie."""
    from ekg.visualization import lead_label

    shares = sorted(lead_shares(attribution).items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{lead_label(k)} {v:.0%}".replace("%", " %") for k, v in shares)


def default_target(result: contract.PredictionResult) -> str:
    """Čo ukázať najprv: najpravdepodobnejší pozitívny label, inak Model A (mapa labelu
    s pravdepodobnosťou blízko 0 by nič nevysvetľovala)."""
    positive = {n: r.probability for n, r in result.labels.items() if r.positive}
    return max(positive, key=positive.get) if positive else MODEL_A


# --- obrázok --------------------------------------------------------------------------


def plot_explanation(expl: Explanation, target: str):
    """12 zvodov s farebným pozadím (mapa) a pás nad nimi – matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ekg.visualization import LAYOUT, lead_label, time_axis

    v = view(expl, target)
    t = time_axis()
    signal = expl.signal_mv
    lim = float(np.percentile(np.abs(signal), 99.5)) * 1.25 or 1.0
    fig = plt.figure(figsize=(16, 11.5))
    grid = fig.add_gridspec(
        8,
        2,
        height_ratios=[0.9, 0.05] + [1] * 6,
        hspace=0.45,
        wspace=0.12,
        left=0.05,
        right=0.93,
        top=0.95,
        bottom=0.07,
    )

    strip_ax = fig.add_subplot(grid[0, :])
    strip_ax.fill_between(t, v["strip"], color="#7f2704" if target == MODEL_A else "#67000d",
                          alpha=0.35, lw=0)  # fmt: skip
    strip_ax.plot(t, v["strip"], color="#222222", lw=0.6 if v["strip2"] is not None else 0.9)
    if v["strip2"] is not None:
        strip_ax.plot(t, v["strip2"], color="#111111", lw=1.2)
    strip_ax.set_xlim(0, contract.DURATION_S)
    strip_ax.set_ylim(0, max(float(np.max(v["strip"])) * 1.1, 1e-9))
    strip_ax.set_xticks(range(contract.DURATION_S + 1))
    strip_ax.tick_params(labelsize=7)
    strip_ax.set_title(v["strip_label"], fontsize=9, loc="left", pad=2)
    strip_ax.grid(alpha=0.25)

    image = None
    for col, leads in enumerate(LAYOUT):
        for row, lead in enumerate(leads):
            ax = fig.add_subplot(grid[row + 2, col])
            i = LEADS.index(lead)
            image = ax.imshow(
                v["heat"][i][None, :],
                extent=(0, contract.DURATION_S, -lim, lim),
                aspect="auto",
                cmap=v["cmap"],
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.plot(t, signal[i], color="#111111", lw=0.8)
            ax.set_xlim(0, contract.DURATION_S)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(range(contract.DURATION_S + 1))
            ax.tick_params(labelsize=7)
            ax.set_ylabel("mV", fontsize=7)
            share = ""
            if target != MODEL_A:
                share = f"   podiel {lead_shares(expl.attributions[target])[lead]:.0%}"
            ax.set_title(f"{lead_label(lead)}{share}", fontsize=9, loc="left", pad=2)
            if row == len(leads) - 1:
                ax.set_xlabel("čas (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)

    bar = fig.colorbar(image, cax=fig.add_axes((0.945, 0.07, 0.008, 0.72)))
    bar.set_label(v["legend"], fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Záznam {expl.record_id} – {v['title']}", fontsize=12, fontweight="bold",
                 x=0.01, ha="left", y=0.995)  # fmt: skip
    fig.text(
        0.01,
        0.005,
        "Mapa ukazuje, ktoré časti vstupu ovplyvnili výstup modelu – nie lekárske vysvetlenie "
        f"a nehovorí, či je výsledok správny. ⚠ {contract.DISCLAIMER}",
        fontsize=8.5,
    )
    return fig


# --- kontrola máp na validation -------------------------------------------------------


def region_share(attribution: np.ndarray, leads: tuple[str, ...]) -> float:
    shares = lead_shares(attribution)
    return float(sum(shares[lead] for lead in leads))


def mi_localization_check(pipe, signals, ids, scp_codes) -> dict:
    """Podiel kladnej MI atribúcie v dolných (II, III, aVF) a predných (V1–V4) zvodoch pre
    správne rozpoznané záznamy s iba inferiórnym (IMI) alebo iba anteroseptálnym (ASMI) MI."""
    from ekg.evaluation.ae_analysis import MI_CODES

    out = {}
    for name, code in (("IMI", "IMI"), ("ASMI", "ASMI")):
        rows = []
        for x, ecg_id in zip(signals, ids, strict=True):
            codes = scp_codes.get(ecg_id) or {}
            if code not in codes or any(m in codes for m in MI_CODES if m != code):
                continue
            e = explain(pipe, x, str(ecg_id))
            if e.probabilities["MI"] < pipe.label_thresholds[LABELS.index("MI")]:
                continue  # iba záznamy, kde model MI naozaj hlási
            a = e.attributions["MI"]
            rows.append((ecg_id, region_share(a, INFERIOR), region_share(a, ANTERIOR)))
        inferior = [r[1] for r in rows]
        anterior = [r[2] for r in rows]
        out[name] = {
            "záznamov": len(rows),
            "podiel dolné zvody (medián)": float(np.median(inferior)) if rows else None,
            "podiel predné zvody (medián)": float(np.median(anterior)) if rows else None,
            "ecg_id": [int(r[0]) for r in rows],
        }
    return out


def randomization_check(pipe, signals, seed: int = 0) -> dict:
    """Sanity check (Adebayo et al. 2018): mapy natrénovaného modelu sa musia líšiť od máp
    rovnakej siete s náhodnými váhami. Spearmanova korelácia |IG| (vyhladené) a Grad-CAM."""
    from scipy.stats import spearmanr

    from ekg.models.classifier import ResNet1dClassifier

    torch.manual_seed(seed)
    random_model = ResNet1dClassifier(pipe.classifier.config).eval()
    ig, cam = [], []
    for x in signals:
        xn = torch.from_numpy(pipe.preprocessor.transform(contract.validate_signal(x)[None]))
        a1, _ = integrated_gradients(pipe.classifier, xn)
        a2, _ = integrated_gradients(random_model, xn)
        c1, c2 = gradcam(pipe.classifier, xn), gradcam(random_model, xn)
        for j in range(len(LABELS)):
            ig.append(spearmanr(smooth(np.abs(a1[j])).ravel(), smooth(np.abs(a2[j])).ravel())[0])
            cam.append(spearmanr(c1[j], c2[j])[0])
    return {
        "záznamov": len(signals),
        "IG Spearman medián": float(np.nanmedian(ig)),
        "Grad-CAM Spearman medián": float(np.nanmedian(cam)),
    }


def run_check(out_dir: Path, n_random: int = 30) -> dict:
    """Kontroly máp iba na VALIDATION (test množina sa nepoužíva)."""
    from ekg.data import ptbxl, splits
    from ekg.data.cache import SignalCache
    from ekg.inference import Pipeline

    pipe = Pipeline.load()
    ids = splits.split_ids("val", subset="classifier")
    db = ptbxl.load_database()
    signals = SignalCache().get(ids)
    codes = db.loc[ids, "scp_codes"].to_dict()
    result = {
        "mi_lokalizacia": mi_localization_check(pipe, signals, list(ids), codes),
        "nahodny_model": randomization_check(pipe, signals[:n_random]),
        "ig_steps": IG_STEPS,
        "split": "val",
    }
    deltas = [max(explain(pipe, x).ig_delta.values()) for x in signals[:n_random]]
    result["ig_uplnost_max_odchylka_median"] = float(np.median(deltas))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kontrola.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


# --- CLI ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from ekg import inference, paths

    parser = argparse.ArgumentParser(description="Mapy dôležitosti a chyba v čase (#27).")
    parser.add_argument("record", nargs="?", help="WFDB záznam (cesta bez prípony)")
    parser.add_argument("--ecg-id", type=int, help="záznam PTB-XL (nie z test množiny)")
    parser.add_argument("--label", choices=[MODEL_A, *LABELS], action="append",
                        help="iba vybrané (predvolene Model A + všetkých 5 labelov)")  # fmt: skip
    parser.add_argument("-o", "--output", type=Path, help="priečinok (predvolene outputs/)")
    parser.add_argument("--check", action="store_true", help="kontrola máp na validation")
    args = parser.parse_args(argv)

    if args.check:
        out = paths.PROJECT_ROOT / "docs" / "vysledky" / "vysvetlenie"
        print(json.dumps(run_check(out), indent=2, ensure_ascii=False))
        return 0
    if (args.record is None) == (args.ecg_id is None):
        parser.error("zadaj práve jeden WFDB záznam alebo --ecg-id (alebo --check)")
    try:
        pipe = inference.Pipeline.load()
        if args.ecg_id is None and inference.ptbxl_id(args.record) is not None:
            args.ecg_id = inference.ptbxl_id(args.record)
        if args.ecg_id is not None:
            [(rid, signal, _meta, _ann)] = list(inference.ptbxl_records([args.ecg_id]))
        else:
            rid, signal = Path(args.record).name, inference.read_wfdb(args.record)
        expl = explain(pipe, signal, rid)
    except (inference.PipelineError, contract.InvalidInputError, OSError, KeyError) as exc:
        print(f"Chyba: {exc}", file=sys.stderr)
        return 1

    import matplotlib.pyplot as plt

    out = args.output or paths.OUTPUT_DIR / "vysvetlenie"
    out.mkdir(parents=True, exist_ok=True)
    for target in args.label or [MODEL_A, *LABELS]:
        fig = plot_explanation(expl, target)
        name = "model_a" if target == MODEL_A else target
        path = out / f"zaznam_{rid}_{name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        extra = ""
        if target != MODEL_A:
            extra = f"  P = {expl.probabilities[target]:.3f}, zvody: "
            extra += top_leads(expl.attributions[target])
        print(f"{path}{extra}")
    print(f"Úplnosť IG (max. relatívna odchýlka): {max(expl.ig_delta.values()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
