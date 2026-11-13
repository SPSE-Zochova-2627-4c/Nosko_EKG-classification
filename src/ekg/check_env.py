"""Overenie vývojového prostredia (nepotrebuje dataset).

Spustenie:  uv run ekg-check

Skontroluje verzie knižníc, dostupné zariadenie (CPU / CUDA / ROCm / MPS),
zápis do pracovných adresárov a urobí jeden tréningový krok malej
1D konvolučnej siete na náhodnom vstupe tvaru (batch, 12, 1000).
"""

import importlib
import platform
import sys
import tempfile

from ekg import paths

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "wfdb", "matplotlib", "yaml", "tqdm", "torch"]


def pick_device():
    import torch

    if torch.cuda.is_available():  # platí pre NVIDIA CUDA aj AMD ROCm
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return torch.device("cuda"), f"{backend} – {torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"
    return torch.device("cpu"), "CPU"


def check_imports() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"  [OK]   {name:<11} {getattr(module, '__version__', '?')}")
        except Exception as exc:  # zámerne široké – chceme vypísať akúkoľvek chybu importu
            print(f"  [CHYBA] {name:<11} {exc}")
            ok = False
    return ok


def check_dirs() -> bool:
    ok = True
    for label, directory in [
        ("dáta", paths.DATA_DIR),
        ("checkpointy", paths.CHECKPOINT_DIR),
        ("výstupy", paths.OUTPUT_DIR),
    ]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
            print(f"  [OK]   {label:<11} {directory}")
        except OSError as exc:
            print(f"  [CHYBA] {label:<11} {directory}: {exc}")
            ok = False
    return ok


def check_training_step() -> bool:
    import torch
    from torch import nn

    torch.manual_seed(0)
    device, device_name = pick_device()
    print(f"  zariadenie: {device_name}")

    model = nn.Sequential(
        nn.Conv1d(12, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(16, 5),  # 5 labelov: MI, STTC, CD, HYP, AFIB
    ).to(device)
    x = torch.randn(4, 12, 1000, device=device)
    y = torch.randint(0, 2, (4, 5), device=device).float()

    loss = nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    ok = torch.isfinite(loss).item() and grads_ok
    print(f"  [{'OK' if ok else 'CHYBA'}]   vstup {tuple(x.shape)} -> loss {loss.item():.4f}")
    return bool(ok)


def main() -> int:
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
    print(f"Projekt: {paths.PROJECT_ROOT}\n")

    print("Knižnice:")
    results = [check_imports()]
    print("\nAdresáre:")
    results.append(check_dirs())
    if results[0]:
        print("\nTréningový krok:")
        results.append(check_training_step())

    if all(results):
        print("\nProstredie je pripravené.")
        return 0
    print("\nNiektoré kontroly zlyhali – pozri [CHYBA] vyššie.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
