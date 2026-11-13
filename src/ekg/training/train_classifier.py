"""Tréning Modelu B – multi-label klasifikátora MI, STTC, CD, HYP, AFIB (issue #16).

Spustenie:
    uv run ekg-train-clf [--config configs/classifier.yaml] [--resume]
    uv run ekg-train-clf --pos-weight none --run-name clf_v1_bez_vah     # porovnávací beh
    uv run ekg-clf-compare clf_v1 clf_v1_bez_vah                         # tabuľka behov

Nevyváženosť labelov: pos_weight = N_neg / N_pos pre každý label, spočítané IBA z train
množiny. BCEWithLogitsLoss potom prehliadnutý pozitívny príklad zriedkavého labelu (AFIB)
trestá viac – akoby bolo pozitívnych príkladov rovnako ako negatívnych.

Výber modelu a early stopping: podľa val macro AUROC (priemer AUROC cez 5 labelov).
Val loss sa na to nehodí – beh s pos_weight a beh bez neho optimalizujú inú loss,
ich straty sa teda nedajú porovnať. AUROC nezávisí od prahu (prahy rieši #17).
Test množina sa tu nepoužíva vôbec (#22).

Výstupy (mimo gitu):
    checkpoints/<run_name>/best.pt     najlepší model podľa val macro AUROC
    checkpoints/<run_name>/last.pt     stav po poslednej epoche – na pokračovanie (--resume)
    outputs/<run_name>/metrics.csv     loss, AUROC a AUPRC po labeloch v každej epoche
    outputs/<run_name>/report.json     train početnosti, pos_weight, najlepšia epocha a metriky
    outputs/<run_name>/config.yaml     použitá konfigurácia vrátane prepínačov z príkazového riadku
    outputs/<run_name>/krivky.png      tréningové krivky
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ekg import contract, paths
from ekg.models.classifier import ClassifierConfig, ResNet1dClassifier, loss_fn
from ekg.training.common import (
    CsvLog,
    EarlyStopping,
    epoch_generator,
    load_yaml,
    pick_device,
    set_seed,
)

DEFAULT_CONFIG = paths.CONFIG_DIR / "classifier.yaml"
LABELS = list(contract.LABELS)
FIELDS = (
    ["epoch", "train_loss", "val_bce", "val_macro_auroc", "val_macro_auprc"]
    + [f"auroc_{label}" for label in LABELS]
    + [f"auprc_{label}" for label in LABELS]
    + ["lr", "seconds", "best"]
)
VAL_BATCH = 256


# --- nevyváženosť ---------------------------------------------------------------


def dataset_targets(ds) -> np.ndarray:
    """Matica cieľov (N, 5); EKGDataset ju má hotovú, inak sa poskladá zo záznamov."""
    targets = getattr(ds, "targets", None)
    if targets is not None:
        return np.asarray(targets, dtype=np.float32)
    return np.stack([np.asarray(ds[i]["target"], dtype=np.float32) for i in range(len(ds))])


def label_counts(targets: np.ndarray) -> dict[str, dict]:
    """Počet pozitívnych a negatívnych príkladov a podiel pozitívnych pre každý label."""
    n = len(targets)
    pos = targets.sum(axis=0)
    return {
        label: {"pos": int(p), "neg": int(n - p), "share": round(float(p) / n, 4)}
        for label, p in zip(LABELS, pos, strict=True)
    }


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    """pos_weight = N_neg / N_pos pre každý label (tvar (5,))."""
    pos = targets.sum(axis=0)
    if np.any(pos == 0):
        missing = [label for label, p in zip(LABELS, pos, strict=True) if p == 0]
        raise ValueError(f"Label bez pozitívnych príkladov v train množine: {missing}")
    neg = len(targets) - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


# --- vyhodnotenie ---------------------------------------------------------------


def classification_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """AUROC a AUPRC po labeloch + makro priemery (prah netreba)."""
    out: dict[str, float] = {}
    for j, label in enumerate(LABELS):
        out[f"auroc_{label}"] = float(roc_auc_score(targets[:, j], probs[:, j]))
        out[f"auprc_{label}"] = float(average_precision_score(targets[:, j], probs[:, j]))
    out["val_macro_auroc"] = float(np.mean([out[f"auroc_{label}"] for label in LABELS]))
    out["val_macro_auprc"] = float(np.mean([out[f"auprc_{label}"] for label in LABELS]))
    return out


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pravdepodobnosti a ciele pre celú množinu (poradie podľa loadera)."""
    model.eval()
    probs, targets = [], []
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(batch["target"].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def bce(probs: np.ndarray, targets: np.ndarray) -> float:
    """Nevážená binárna krížová entropia – porovnateľná medzi behmi s váhami a bez nich."""
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))


def train_one_epoch(model, loader, device, optimizer, criterion, amp: bool) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


# --- tréning ----------------------------------------------------------------------


def _check_no_test(cfg: dict) -> None:
    for key in ("train", "val"):
        if cfg["data"][key].split("/")[0] == "test":
            raise ValueError("Test množina sa pri tréningu nepoužíva (je zamknutá pre #22).")


def train(
    cfg: dict,
    train_ds,
    val_ds,
    ckpt_dir: Path,
    out_dir: Path,
    resume: bool = False,
    device: torch.device | None = None,
    stop_after_epoch: int | None = None,  # iba pre test prerušenia
    verbose: bool = True,
) -> dict:
    _check_no_test(cfg)
    device = device or pick_device()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    set_seed(seed)
    tr = cfg["training"]
    amp = tr.get("amp", "none") == "bf16" and device.type == "cuda"

    train_targets = dataset_targets(train_ds)
    counts = label_counts(train_targets)
    mode = cfg["imbalance"]["pos_weight"]
    if mode == "train":
        pos_weight = compute_pos_weight(train_targets)
    elif mode == "none":
        pos_weight = None
    else:
        raise ValueError(f"imbalance.pos_weight musí byť 'train' alebo 'none', nie {mode!r}")
    criterion = loss_fn(None if pos_weight is None else pos_weight.to(device))

    model = ResNet1dClassifier(ClassifierConfig(**cfg["model"])).to(device)
    opt = torch.optim.Adam(model.parameters(), **cfg["optimizer"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", **cfg["scheduler"])
    stopper = EarlyStopping(tr["early_stopping_patience"], tr["min_delta"])  # minimalizuje −AUROC
    start_epoch = 1

    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"{last_path} neexistuje – nie je z čoho pokračovať.")
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        stopper.load_state(state["early_stopping"])
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        if verbose:
            best = -stopper.best
            print(f"Pokračujem od epochy {start_epoch} (najlepší val macro AUROC {best:.4f}).")
    log = CsvLog(out_dir / "metrics.csv", FIELDS, append=resume)

    pin = device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, pin_memory=pin)
    t_start = time.perf_counter()
    for epoch in range(start_epoch, tr["max_epochs"] + 1):
        if stopper.should_stop:
            break
        t0 = time.perf_counter()
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            generator=epoch_generator(seed, epoch),
            pin_memory=pin,
        )
        train_loss = train_one_epoch(model, train_loader, device, opt, criterion, amp)
        probs, targets = predict(model, val_loader, device, amp)
        m = classification_metrics(targets, probs)
        val_auroc = m["val_macro_auroc"]
        sched.step(val_auroc)
        is_best = stopper.step(-val_auroc, epoch)
        if is_best:
            model.save(
                best_path,
                epoch=epoch,
                val_macro_auroc=val_auroc,
                run_name=cfg["run_name"],
                seed=seed,
                pos_weight=None if pos_weight is None else pos_weight.tolist(),
                amp="bf16" if amp else "none",
            )
        seconds = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_bce": f"{bce(probs, targets):.6f}",
            **{k: f"{v:.5f}" for k, v in m.items()},
            "lr": opt.param_groups[0]["lr"],
            "seconds": f"{seconds:.1f}",
            "best": int(is_best),
        }
        log.write(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "early_stopping": stopper.state(),
                "epoch": epoch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            last_path,
        )
        if verbose:
            per_label = " ".join(f"{label} {m[f'auroc_{label}']:.3f}" for label in LABELS)
            print(
                f"epocha {epoch:>3}  train {train_loss:.4f}  val BCE {row['val_bce']}  "
                f"AUROC {val_auroc:.4f} ({per_label})  {seconds:.0f} s{' *' if is_best else ''}",
                flush=True,
            )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {"interrupted": True, "epoch": epoch}

    rows = log.read()
    best_row = next(r for r in rows if int(r["epoch"]) == stopper.best_epoch)
    report = {
        "run_name": cfg["run_name"],
        "seed": seed,
        "device": str(device),
        "amp": "bf16" if amp else "none",
        "torch": str(torch.__version__),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_label_counts": counts,
        "pos_weight_mode": mode,
        "pos_weight": None
        if pos_weight is None
        else {label: round(float(w), 4) for label, w in zip(LABELS, pos_weight, strict=True)},
        "best_epoch": stopper.best_epoch,
        "epochs_run": len(rows),
        "stopped_early": stopper.should_stop,
        "train_minutes_this_session": round((time.perf_counter() - t_start) / 60, 2),
        "best_val": {k: float(best_row[k]) for k in FIELDS if k.startswith(("val_", "au"))},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return {
        "best_epoch": stopper.best_epoch,
        "best_val_macro_auroc": -stopper.best,
        "epochs_run": len(rows),
        "stopped_early": stopper.should_stop,
        "report": report,
    }


def plot_curves(metrics_csv: Path, out_png: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    m = pd.read_csv(metrics_csv)
    best = m.loc[m["val_macro_auroc"].idxmax()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(m["epoch"], m["train_loss"], label="train loss (ako sa trénovalo)")
    ax1.plot(m["epoch"], m["val_bce"], label="validation BCE (bez váh)")
    ax1.set(xlabel="epocha", ylabel="binárna krížová entropia", title="Loss")
    for label in LABELS:
        ax2.plot(m["epoch"], m[f"auroc_{label}"], lw=1, label=label)
    ax2.plot(m["epoch"], m["val_macro_auroc"], color="black", lw=2, label="macro")
    ax2.set(xlabel="epocha", ylabel="AUROC (validation)", title="AUROC po labeloch")
    for ax in (ax1, ax2):
        ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax2.annotate(
        f"najlepšia epocha {int(best['epoch'])}\nmacro AUROC {best['val_macro_auroc']:.3f}",
        (best["epoch"], best["val_macro_auroc"]),
        xytext=(10, -40),
        textcoords="offset points",
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Prepínače z príkazového riadku majú prednosť pred konfiguráciou."""
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.pos_weight:
        cfg["imbalance"]["pos_weight"] = args.pos_weight
    if args.amp:
        cfg["training"]["amp"] = args.amp
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    return cfg


def main(argv: list[str] | None = None) -> int:
    from ekg.data.dataset import make_dataset

    parser = argparse.ArgumentParser(description="Tréning multi-label klasifikátora (Model B).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="pokračovať z last.pt")
    parser.add_argument("--run-name", help="prepíše run_name z konfigurácie")
    parser.add_argument("--pos-weight", choices=["train", "none"], help="prepíše imbalance")
    parser.add_argument("--amp", choices=["bf16", "none"], help="prepíše training.amp")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)

    cfg = apply_overrides(load_yaml(args.config), args)
    _check_no_test(cfg)
    name = cfg["run_name"]
    ckpt_dir, out_dir = paths.CHECKPOINT_DIR / name, paths.OUTPUT_DIR / name
    if not args.resume and (ckpt_dir / "last.pt").exists():
        print(f"Beh {name!r} už existuje. Pokračuj cez --resume alebo zvoľ iný --run-name.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def dataset(spec: str):
        split, subset = spec.split("/")
        return make_dataset(split, subset)

    device = pick_device()
    print(f"Beh {name}: zariadenie {device}, amp {cfg['training'].get('amp', 'none')}")
    print("Načítavam dáta…")
    train_ds, val_ds = dataset(cfg["data"]["train"]), dataset(cfg["data"]["val"])
    print(f"train {len(train_ds)} záznamov, val {len(val_ds)} záznamov")
    counts = label_counts(dataset_targets(train_ds))
    print("pozitívne v train: " + ", ".join(f"{k} {v['pos']}" for k, v in counts.items()))
    if cfg["imbalance"]["pos_weight"] == "train":
        w = compute_pos_weight(dataset_targets(train_ds))
        print("pos_weight: " + ", ".join(f"{k} {v:.2f}" for k, v in zip(LABELS, w, strict=True)))
    print()

    try:
        summary = train(cfg, train_ds, val_ds, ckpt_dir, out_dir, resume=args.resume, device=device)
    except KeyboardInterrupt:
        print(f"\nPrerušené. Pokračuj príkazom: uv run ekg-train-clf --run-name {name} --resume")
        return 130
    plot_curves(out_dir / "metrics.csv", out_dir / "krivky.png", title=name)
    print(
        f"\nHotovo: najlepšia epocha {summary['best_epoch']}, "
        f"val macro AUROC {summary['best_val_macro_auroc']:.4f}"
        f"{' (early stopping)' if summary['stopped_early'] else ''}"
    )
    print(f"Model: {ckpt_dir / 'best.pt'}\nReport: {out_dir / 'report.json'}")
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    """Tabuľka najlepších val metrík pre viac behov (z outputs/<run>/report.json)."""
    parser = argparse.ArgumentParser(description="Porovnanie behov klasifikátora.")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)

    cols = ["val_macro_auroc", "val_macro_auprc"] + [f"auroc_{label}" for label in LABELS]
    header = ["beh", "pos_weight", "amp", "epocha"] + [c.replace("val_", "") for c in cols]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for run in args.runs:
        path = paths.OUTPUT_DIR / run / "report.json"
        if not path.exists():
            print(f"{run}: chýba {path}")
            continue
        r = json.loads(path.read_text())
        values = [f"{r['best_val'][c]:.3f}" for c in cols]
        print(" | ".join([run, r["pos_weight_mode"], r["amp"], str(r["best_epoch"])] + values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
