"""Model B – 1D ResNet, multi-label klasifikátor MI, STTC, CD, HYP, AFIB (issue #15).

Vstup (B, 12, 1000) po preprocessingu v1.0, výstup (B, 5) LOGITY v poradí contract.LABELS.

Multi-label, nie multi-class: záznam môže mať viac porúch naraz (napr. MI + CD), preto
- každý label má vlastný NEZÁVISLÝ výstup a sigmoid: P(label) = σ(logit) ∈ [0, 1],
- tréning s BCEWithLogitsLoss (binárna krížová entropia pre každý label zvlášť),
- NIE softmax – ten by vynútil súčet pravdepodobností = 1, teda práve jednu triedu.
Model vracia logity (nie pravdepodobnosti), lebo BCEWithLogitsLoss počíta sigmoid
numericky stabilne vo vnútri; pravdepodobnosti dáva predict_proba().

Architektúra (predvolená konfigurácia):

    stem      Conv1d k=15 stride 2 (64 kanálov) + BN + ReLU + MaxPool stride 2   → (B,  64, 250)
    stage 1   2× reziduálny blok, 64 kanálov                                      → (B,  64, 250)
    stage 2   2× reziduálny blok, 128 kanálov, prvý so stride 2                   → (B, 128, 125)
    stage 3   2× reziduálny blok, 256 kanálov, prvý so stride 2                   → (B, 256,  63)
    hlava     [priemer ‖ maximum] cez čas → Dropout → Linear(512 → 5)

Reziduálny blok:  y = ReLU( BN(Conv(ReLU(BN(Conv(x))))) + skratka(x) )
Skratka (identita alebo Conv 1×1, keď sa mení počet kanálov či dĺžka) umožní
gradientu prejsť cez veľa vrstiev – preto sa dajú trénovať hlbšie siete.
Priemer aj maximum cez čas: priemer zachytí trvalé zmeny (napr. HYP),
maximum krátke lokálne udalosti, ktoré sa objavia hocikde v 10 s (napr. extrasystola).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ekg import contract
from ekg.models import checkpoint

MODEL_TYPE = "resnet1d_classifier"


@dataclass
class ClassifierConfig:
    in_channels: int = contract.N_LEADS
    n_labels: int = contract.N_LABELS
    stem_channels: int = 64
    stem_kernel: int = 15
    channels: tuple[int, ...] = (64, 128, 256)  # šírka stage-ov
    blocks: tuple[int, ...] = (2, 2, 2)  # počet blokov v každom stage
    kernel_size: int = 7
    dropout: float = 0.3

    def __post_init__(self):
        self.channels, self.blocks = tuple(self.channels), tuple(self.blocks)
        if len(self.channels) != len(self.blocks):
            raise ValueError("channels a blocks musia mať rovnakú dĺžku.")
        if self.kernel_size % 2 == 0 or self.stem_kernel % 2 == 0:
            raise ValueError("Veľkosti kernelov musia byť nepárne.")
        if self.n_labels != contract.N_LABELS:
            raise ValueError(f"Kontrakt má {contract.N_LABELS} labelov.")


class ResidualBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1dClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig | None = None):
        super().__init__()
        self.config = c = config or ClassifierConfig()
        self.stem = nn.Sequential(
            nn.Conv1d(
                c.in_channels,
                c.stem_channels,
                c.stem_kernel,
                stride=2,
                padding=c.stem_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(c.stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers: list[nn.Module] = []
        prev = c.stem_channels
        for i, (ch, n) in enumerate(zip(c.channels, c.blocks, strict=True)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(ResidualBlock1d(prev, ch, c.kernel_size, stride))
                prev = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(c.dropout), nn.Linear(2 * prev, c.n_labels))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Mapa čŕt pred hlavou (B, C, L) – využije Grad-CAM (#27)."""
        return self.body(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1)
        return self.head(pooled)  # logity (B, 5)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Pravdepodobnosti (B, 5) v poradí contract.LABELS – v režime eval()."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(x))
        finally:
            self.train(was_training)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- checkpoint ---------------------------------------------------------

    def save(self, path: Path, **extra) -> None:
        checkpoint.save(path, MODEL_TYPE, asdict(self.config), self, **extra)

    @classmethod
    def load(cls, path: Path) -> tuple[ResNet1dClassifier, dict]:
        ckpt = checkpoint.load(path, MODEL_TYPE)
        model = cls(ClassifierConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt


def loss_fn(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """BCEWithLogitsLoss; pos_weight (5,) zvýši váhu zriedkavých labelov (počíta sa v #16)."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def main() -> int:
    """Prehľad architektúry a skúšobný priechod skutočnou dávkou Modelu B."""
    import time

    from ekg.data.dataset import make_loader

    torch.manual_seed(0)
    model = ResNet1dClassifier()
    print(f"ResNet1dClassifier: {model.n_parameters():,} parametrov".replace(",", " "))
    model.eval()
    x = torch.zeros(1, contract.N_LEADS, contract.N_SAMPLES)
    with torch.no_grad():
        x = model.stem(x)
        print(f"  stem               → {tuple(x.shape)}")
        for i, block in enumerate(model.body):
            x = block(x)
            print(f"  reziduálny blok {i + 1}  → {tuple(x.shape)}")
    print(f"  hlava              → (1, {contract.N_LABELS})  {list(contract.LABELS)}")

    batch = next(iter(make_loader("train", "classifier", batch_size=32, seed=0)))
    model.train()
    start = time.perf_counter()
    logits = model(batch["signal"])
    loss = loss_fn()(logits, batch["target"])
    loss.backward()
    elapsed = time.perf_counter() - start
    proba = model.predict_proba(batch["signal"])
    print(
        f"\nskutočná dávka {tuple(batch['signal'].shape)} → logity {tuple(logits.shape)}, "
        f"BCE pred tréningom {loss.item():.3f} (pri všetkých logitoch 0 by bola ln 2 = 0.693)"
    )
    print(f"forward+backward {elapsed * 1000:.0f} ms (CPU)")
    first = ", ".join(
        f"{k} {v:.2f}" for k, v in zip(contract.LABELS, proba[0].tolist(), strict=True)
    )
    print(f"pravdepodobnosti prvého záznamu (netrénovaný model): {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
