"""Porovnanie rýchlosti tréningu na CPU a GPU (issue #28). Nepotrebuje dataset.

Spustenie:  uv run ekg-gpu-bench [--steps 30] [--batch-size 64] [--threads N]

Pre oba modely projektu (Model A – autoencoder, Model B – ResNet klasifikátor) urobí
niekoľko tréningových krokov (forward + backward + Adam) na náhodnom vstupe tvaru
(batch, 12, 1000) – najprv na CPU, potom na GPU, ak je dostupné. Prvé kroky sa nemerajú
(zahrievanie: GPU pri prvom behu kompiluje/vyberá kernely). Na GPU meria float32 aj
zmiešanú presnosť bfloat16 – na RX 9070 XT je bf16 asi 5× rýchlejšia, lebo MIOpen pre
viaceré float32 1D konvolúcie volí pomalé záložné algoritmy (docs/gpu_prostredie.md).
Vypíše čas na krok, počet záznamov za sekundu a zrýchlenie GPU oproti CPU, plus verzie.

Náhodný vstup stačí – čas kroku závisí od tvaru dát a architektúry, nie od hodnôt.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
from torch import nn

from ekg import contract
from ekg.check_env import pick_device


@dataclass
class BenchResult:
    model: str
    device: str
    ms_per_step: float
    samples_per_s: float
    final_loss: float


MODELS = ("autoencoder", "classifier")


def _build(model_name: str) -> tuple[nn.Module, nn.Module]:
    """Model s predvolenou konfiguráciou a jeho loss."""
    if model_name == "classifier":
        from ekg.models.classifier import ResNet1dClassifier, loss_fn

        return ResNet1dClassifier(), loss_fn()
    if model_name == "autoencoder":
        from ekg.models.autoencoder import ConvAutoencoder

        return ConvAutoencoder(), nn.MSELoss()
    raise ValueError(f"Neznámy model: {model_name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(
    model_name: str,
    device: torch.device,
    *,
    steps: int = 30,
    warmup: int = 5,
    batch_size: int = 64,
    seed: int = 0,
    amp: bool = False,
) -> BenchResult:
    """Odmeria priemerný čas jedného tréningového kroku.

    amp=True: forward a loss bežia v automatickej zmiešanej presnosti bfloat16
    (torch.autocast), váhy a krok optimalizátora ostávajú vo float32. Iba na GPU.
    """
    if amp and device.type != "cuda":
        raise ValueError("Zmiešaná presnosť (amp) sa meria iba na GPU.")
    torch.manual_seed(seed)
    model, criterion = _build(model_name)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(batch_size, contract.N_LEADS, contract.N_SAMPLES, device=device)
    if model_name == "classifier":
        y = torch.randint(0, 2, (batch_size, contract.N_LABELS), device=device).float()
    else:
        y = x  # autoencoder rekonštruuje vstup

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        loss = step()
    _sync(device)
    elapsed = time.perf_counter() - start

    loss_value = loss.item()
    if not math.isfinite(loss_value):
        raise RuntimeError(f"{model_name} na {device}: loss nie je konečné číslo ({loss_value}).")
    return BenchResult(
        model=model_name,
        device=device.type + (" bf16" if amp else ""),
        ms_per_step=elapsed / steps * 1000,
        samples_per_s=steps * batch_size / elapsed,
        final_loss=loss_value,
    )


def environment_info() -> dict[str, str]:
    """Verzie, ktoré sa zapisujú do dokumentácie (OS, jadro, PyTorch, ROCm/CUDA, GPU)."""
    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "wsl": "áno" if "microsoft" in platform.release().lower() else "nie",
        "torch": torch.__version__,
        "hip (ROCm)": str(getattr(torch.version, "hip", None)),
        "cuda": str(torch.version.cuda),
        "cpu vlákna": str(torch.get_num_threads()),
    }
    _, device_name = pick_device()
    info["zariadenie"] = device_name
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rýchlosť tréningu CPU vs. GPU.")
    parser.add_argument("--steps", type=int, default=30, help="merané kroky (predvolene 30)")
    parser.add_argument("--warmup", type=int, default=5, help="nemerané úvodné kroky")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None, help="počet CPU vlákien PyTorch")
    parser.add_argument("--skip-cpu", action="store_true", help="merať iba GPU")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    print("Prostredie:")
    for key, value in environment_info().items():
        print(f"  {key:<11} {value}")
    print(f"  CPU jadrá   {os.cpu_count()}")

    # (zariadenie, amp): CPU float32, GPU float32 a GPU bfloat16
    runs = [] if args.skip_cpu else [(torch.device("cpu"), False)]
    gpu, _ = pick_device()
    if gpu.type == "cuda":
        runs.append((gpu, False))
        if torch.cuda.is_bf16_supported():  # napr. NVIDIA T4 v Colabe bf16 nemá
            runs.append((gpu, True))
        else:
            print("\nGPU nepodporuje bfloat16 – meria sa iba float32.")
    else:
        print("\nGPU nie je dostupné – meria sa iba CPU.")
    if not runs:
        print("Nie je čo merať (--skip-cpu bez GPU).")
        return 1

    print(
        f"\nTréningový krok (forward + backward + Adam), dávka {args.batch_size}, "
        f"{args.steps} meraných krokov po {args.warmup} zahrievacích:\n"
    )
    print(f"  {'model':<12} {'zariadenie':<10} {'ms/krok':>9} {'záznamy/s':>10} {'zrýchlenie':>11}")
    for model_name in MODELS:
        cpu_ms = None
        for device, amp in runs:
            r = bench(
                model_name,
                device,
                steps=args.steps,
                warmup=args.warmup,
                batch_size=args.batch_size,
                amp=amp,
            )
            if device.type == "cpu":
                cpu_ms = r.ms_per_step
            speedup = f"{cpu_ms / r.ms_per_step:.1f}×" if cpu_ms and device.type != "cpu" else ""
            print(
                f"  {r.model:<12} {r.device:<10} {r.ms_per_step:>9.1f} "
                f"{r.samples_per_s:>10.0f} {speedup:>11}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
