"""Tréning autoencodera na prísnom NORM (issue #11).

Spustenie:  uv run ekg-train-ae [--config configs/autoencoder.yaml] [--resume]

Výstupy (mimo gitu):
    checkpoints/<run_name>/best.pt     najlepší model podľa val loss (formát ekg.models.checkpoint)
    checkpoints/<run_name>/last.pt     stav po poslednej epoche – na pokračovanie (--resume)
    outputs/<run_name>/metrics.csv     loss po epochách
    outputs/<run_name>/config.yaml     kópia konfigurácie
    outputs/<run_name>/krivky.png      tréningové krivky

Prerušenie (Ctrl+C, vypnutie počítača) nevadí: last.pt sa ukladá po každej epoche
a --resume pokračuje od nasledujúcej epochy s rovnakým stavom optimizera, plánovača,
early stoppingu aj poradím miešania – výsledok je rovnaký ako bez prerušenia.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ekg import paths
from ekg.models.autoencoder import AutoencoderConfig, ConvAutoencoder
from ekg.training.common import (
    CsvLog,
    EarlyStopping,
    epoch_generator,
    load_yaml,
    pick_device,
    set_seed,
)

DEFAULT_CONFIG = paths.CONFIG_DIR / "autoencoder.yaml"
FIELDS = ["epoch", "train_loss", "val_loss", "lr", "seconds", "best"]


def run_one_epoch(model, loader, device, optimizer=None) -> float:
    """Priemerná MSE cez záznamy; s optimizerom trénuje, bez neho iba vyhodnotí."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["signal"].to(device)
            loss = nn.functional.mse_loss(model(x), x)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


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
    device = device or pick_device()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    set_seed(seed)

    model = ConvAutoencoder(AutoencoderConfig(**cfg["model"])).to(device)
    opt = torch.optim.Adam(model.parameters(), **cfg["optimizer"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", **cfg["scheduler"])
    tr = cfg["training"]
    stopper = EarlyStopping(tr["early_stopping_patience"], tr["min_delta"])
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
        if verbose:
            print(f"Pokračujem od epochy {start_epoch} (najlepšia val loss {stopper.best:.5f}).")
    log = CsvLog(out_dir / "metrics.csv", FIELDS, append=resume)

    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
    for epoch in range(start_epoch, tr["max_epochs"] + 1):
        if stopper.should_stop:
            break
        t0 = time.perf_counter()
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            generator=epoch_generator(seed, epoch),
        )
        train_loss = run_one_epoch(model, train_loader, device, opt)
        val_loss = run_one_epoch(model, val_loader, device)
        sched.step(val_loss)
        is_best = stopper.step(val_loss, epoch)
        if is_best:
            model.save(
                best_path, epoch=epoch, val_loss=val_loss, run_name=cfg["run_name"], seed=seed
            )
        seconds = time.perf_counter() - t0
        log.write(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "lr": opt.param_groups[0]["lr"],
                "seconds": f"{seconds:.1f}",
                "best": int(is_best),
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "early_stopping": stopper.state(),
                "epoch": epoch,
                "torch_rng": torch.get_rng_state(),
            },
            last_path,
        )
        if verbose:
            mark = " *" if is_best else ""
            print(
                f"epocha {epoch:>3}  train {train_loss:.5f}  val {val_loss:.5f}  "
                f"lr {opt.param_groups[0]['lr']:.1e}  {seconds:.0f} s{mark}",
                flush=True,
            )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {"interrupted": True, "epoch": epoch}

    return {
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best,
        "epochs_run": len(log.read()),
        "stopped_early": stopper.should_stop,
    }


def plot_curves(metrics_csv: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    m = pd.read_csv(metrics_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(m["epoch"], m["train_loss"], label="train (prísny NORM)")
    ax.plot(m["epoch"], m["val_loss"], label="validation (prísny NORM)")
    best = m.loc[m["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
    ax.annotate(
        f"najlepšia epocha {int(best['epoch'])}\nval MSE {best['val_loss']:.4f}",
        (best["epoch"], best["val_loss"]),
        xytext=(10, 30),
        textcoords="offset points",
    )
    ax.set(
        xlabel="epocha",
        ylabel="MSE (rekonštrukčná chyba)",
        yscale="log",
        title="Tréning autoencodera",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from ekg.data.dataset import make_dataset

    parser = argparse.ArgumentParser(description="Tréning autoencodera na prísnom NORM.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="pokračovať z last.pt")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    name = cfg["run_name"]
    ckpt_dir, out_dir = paths.CHECKPOINT_DIR / name, paths.OUTPUT_DIR / name
    if not args.resume and (ckpt_dir / "last.pt").exists():
        print(
            f"Beh {name!r} už existuje. Pokračuj cez --resume alebo zmeň run_name v konfigurácii."
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    def dataset(spec: str):
        split, subset = spec.split("/")
        return make_dataset(split, subset)

    device = pick_device()
    print(f"Načítavam dáta… (zariadenie: {device})")
    train_ds, val_ds = dataset(cfg["data"]["train"]), dataset(cfg["data"]["val"])
    print(f"train {len(train_ds)} záznamov, val {len(val_ds)} záznamov\n")

    try:
        summary = train(cfg, train_ds, val_ds, ckpt_dir, out_dir, resume=args.resume, device=device)
    except KeyboardInterrupt:
        print("\nPrerušené. Pokračuj príkazom: uv run ekg-train-ae --resume")
        return 130
    plot_curves(out_dir / "metrics.csv", out_dir / "krivky.png")
    print(
        f"\nHotovo: najlepšia epocha {summary['best_epoch']}, "
        f"val MSE {summary['best_val_loss']:.5f}"
        f"{' (early stopping)' if summary['stopped_early'] else ''}"
    )
    print(f"Model: {ckpt_dir / 'best.pt'}\nKrivky: {out_dir / 'krivky.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
