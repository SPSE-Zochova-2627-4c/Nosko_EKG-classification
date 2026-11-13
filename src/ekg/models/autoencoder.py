"""Model A – 1D konvolučný autoencoder pre 12-zvodové EKG (issue #10).

Vstup aj výstup majú tvar (B, 12, 1000) – signál po preprocessingu v1.0.

Architektúra (predvolená konfigurácia):

    vstup        (B,  12, 1000)
    encoder      3 bloky, každý: Conv1d so stride 2 (zmenší dĺžku na polovicu)
                 + BatchNorm + ReLU + Conv1d + BatchNorm + ReLU
                 (B,  32, 500) → (B,  64, 250) → (B, 128, 125)
    latent       Conv1d 1×1 → (B, 16, 125)        ← úzke hrdlo: 2 000 čísel namiesto 12 000
    decoder      zrkadlovo: ConvTranspose1d (zdvojnásobí dĺžku) + BatchNorm + ReLU + Conv1d …
                 (B, 128, 125) → (B,  64, 250) → (B,  32, 500) → (B,  32, 1000)
    výstup       Conv1d → (B, 12, 1000), bez aktivácie (normalizovaný signál môže byť záporný)

Model je plne konvolučný: každý bod latentnej reprezentácie opisuje krátky úsek
(~8 vzoriek = 80 ms) všetkých 12 zvodov. Hrdlo (latent_channels) určuje, koľko
informácie sa „zmestí“ – príliš široké hrdlo by sa naučilo kopírovať vstup a dobre
by zrekonštruovalo aj anomálie; príliš úzke by nezvládlo ani normálne EKG.
Šírka hrdla sa ladí pri tréningu (#11) podľa validation.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ekg import contract
from ekg.models import checkpoint

MODEL_TYPE = "conv_autoencoder"


@dataclass
class AutoencoderConfig:
    in_channels: int = contract.N_LEADS
    channels: tuple[int, ...] = (32, 64, 128)  # šírka encoder blokov; každý blok dĺžku /2
    latent_channels: int = 16
    kernel_size: int = 7
    dropout: float = 0.0

    def __post_init__(self):
        self.channels = tuple(self.channels)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size musí byť nepárny (symetrický padding).")
        factor = 2 ** len(self.channels)
        if contract.N_SAMPLES % factor:
            raise ValueError(f"{contract.N_SAMPLES} vzoriek sa nedá {len(self.channels)}× deliť 2.")

    @property
    def latent_length(self) -> int:
        return contract.N_SAMPLES // 2 ** len(self.channels)

    @property
    def compression(self) -> float:
        """Koľkokrát je latentná reprezentácia menšia ako vstup."""
        latent = self.latent_channels * self.latent_length
        return contract.N_LEADS * contract.N_SAMPLES / latent


def _conv_bn_relu(c_in: int, c_out: int, k: int, stride: int = 1) -> list[nn.Module]:
    return [
        nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    ]


class ConvAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | None = None):
        super().__init__()
        self.config = config or AutoencoderConfig()
        c, k = self.config, self.config.kernel_size

        enc: list[nn.Module] = []
        prev = c.in_channels
        for ch in c.channels:
            enc += _conv_bn_relu(prev, ch, k, stride=2)  # dĺžka / 2
            enc += _conv_bn_relu(ch, ch, k)
            if c.dropout:
                enc.append(nn.Dropout(c.dropout))
            prev = ch
        enc.append(nn.Conv1d(prev, c.latent_channels, kernel_size=1))  # hrdlo
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = [*_conv_bn_relu(c.latent_channels, prev, 1)]
        for ch in reversed((c.channels[0], *c.channels[:-1])):
            dec += [
                nn.ConvTranspose1d(prev, ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),  # dĺžka × 2
                *_conv_bn_relu(ch, ch, k),
            ]
            prev = ch
        dec.append(nn.Conv1d(prev, c.in_channels, k, padding=k // 2))  # lineárny výstup
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- checkpoint ---------------------------------------------------------

    def save(self, path: Path, **extra) -> None:
        checkpoint.save(path, MODEL_TYPE, asdict(self.config), self, **extra)

    @classmethod
    def load(cls, path: Path) -> tuple[ConvAutoencoder, dict]:
        """Načíta model v režime eval() a vráti ho spolu s celým checkpointom."""
        ckpt = checkpoint.load(path, MODEL_TYPE)
        model = cls(AutoencoderConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval(), ckpt


def main() -> int:
    """Prehľad architektúry a skúšobný priechod na skutočnej dávke prísneho NORM."""
    import time

    from ekg.data.dataset import make_loader

    torch.manual_seed(0)
    model = ConvAutoencoder()
    cfg = model.config
    print(f"ConvAutoencoder: {model.n_parameters():,} parametrov".replace(",", " "))
    print(
        f"latent: ({cfg.latent_channels}, {cfg.latent_length}) = "
        f"{cfg.latent_channels * cfg.latent_length} čísel, kompresia {cfg.compression:.1f}×\n"
    )
    model.eval()
    x = torch.zeros(1, contract.N_LEADS, contract.N_SAMPLES)
    with torch.no_grad():
        for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
            print(name)
            for layer in part:
                x = layer(x)
                if isinstance(layer, nn.Conv1d | nn.ConvTranspose1d):
                    print(f"  {type(layer).__name__:<16} → {tuple(x.shape)}")

    batch = next(iter(make_loader("train", "strict_norm", batch_size=32, seed=0)))["signal"]
    model.train()
    start = time.perf_counter()
    recon = model(batch)
    loss = nn.functional.mse_loss(recon, batch)
    loss.backward()
    elapsed = time.perf_counter() - start
    print(
        f"\nskutočná dávka prísneho NORM {tuple(batch.shape)} → {tuple(recon.shape)}, "
        f"MSE pred tréningom {loss.item():.3f}, forward+backward {elapsed * 1000:.0f} ms (CPU)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
