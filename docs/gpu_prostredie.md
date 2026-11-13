# Tréningové prostredie s GPU (issue #28)

Kód: [`src/ekg/gpu_benchmark.py`](../src/ekg/gpu_benchmark.py) ·
Konfigurácia: [`pyproject.toml`](../pyproject.toml) (`[tool.uv.sources]`) ·
Záloha: [`notebooks/colab_trening.ipynb`](../notebooks/colab_trening.ipynb)

```bash
uv run ekg-check       # má vypísať „zariadenie: ROCm – AMD Radeon RX 9070 XT“
uv run ekg-gpu-bench   # rýchlosť tréningu CPU vs. GPU (float32 aj bfloat16)
```

## Kde sa čo robí

| Počítač | PyTorch | Použitie |
|---|---|---|
| Intel Mac (16 GB RAM) | 2.2.2, iba CPU | písanie kódu, testy, malé pokusy |
| **PC – AMD RX 9070 XT, WSL2 Ubuntu** | **2.14.0 + ROCm 10.1 (iba gfx1201)** | **trénovanie** |
| Google Colab / Kaggle | predinštalovaný s CUDA | záloha, keď PC nie je k dispozícii |

## Zapísané verzie (PC, 13. 11. 2026)

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8 jadier, 16 vlákien) |
| RAM | 32 GB (vo WSL viditeľných 30 GB) |
| GPU | AMD Radeon RX 9070 XT, 16 GB, architektúra RDNA 4 (**gfx1201**) |
| OS | Windows 11 Home, zostava 26300 |
| ovládač GPU | AMD Adrenalin, verzia ovládača 32.0.32015.7029 (18. 10. 2026) |
| WSL | 2.6.1, jadro 6.6.87.2-microsoft-standard-WSL2 |
| Linux | Ubuntu 24.04.5 LTS (vo WSL2) |
| Python | 3.11.18 (nainštalovaný cez uv) |
| PyTorch | 2.14.0+rocm10.1.0 (`torch.version.hip` = 7.16.26385) |
| index | `https://stable.repo.amd.com/rocm/whl-next/` |

## Prečo WSL2 a balíky od AMD

- **Linux namiesto Windows:** PyTorch s ROCm je na Linuxe najdlhšie a najlepšie podporovaný.
  Pre Windows existujú iba samostatné balíky od AMD, ktoré sa horšie zamykajú v `uv.lock`.
- **WSL2 namiesto dual-bootu:** netreba preinštalovať počítač. Od ROCm 7.13 je podpora WSL
  (ROCDXG – prístup ku GPU cez Windows ovládač a zariadenie `/dev/dxg`) priamo v runtime ROCm,
  takže **do Ubuntu netreba inštalovať ROCm ani ovládač**. Všetko potrebné je v pip balíkoch.
- **Index AMD namiesto indexu PyTorch:** oficiálny `download.pytorch.org/whl/rocm10.0` má torch,
  ktorý vyžaduje `rocm[device-all]` – knižnice pre **všetky** GPU od AMD (aj serverové), teda
  mnoho GB navyše. AMD index ponúka `torch[device-gfx1201]` iba pre našu kartu: spolu ~1,6 GB
  na stiahnutie (torch 198 MB, rocm-sdk-core 399 MB, rocm-sdk-libraries 248 MB,
  rocm-sdk-device-gfx1201 327 MB, amd-torch-device-gfx1201 50 MB, triton 407 MB).

## Inštalácia na novom PC (od nuly)

Vo Windows (PowerShell):

```powershell
wsl --install -d Ubuntu-24.04     # pri prvom spustení vytvoriť používateľa a heslo
```

V Ubuntu:

```bash
# systémové balíky – bez libgomp1 sa torch nenačíta, bez libc6-dev padá BatchNorm na GPU
sudo apt-get update && sudo apt-get install -y git curl libgomp1 libc6-dev

# GitHub CLI (oficiálny repozitár GitHubu)
sudo mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update && sudo apt-get install -y gh

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# repozitár – v linuxovom súborovom systéme (~), NIE v /mnt/c (tam je WSL výrazne pomalšie)
gh auth login -h github.com -p https -w -s project,read:org,workflow
cd ~ && gh repo clone SPSE-Zochova-2627-4c/Nosko_EKG-classification
cd Nosko_EKG-classification
git config user.name "Matúš Nosko"
git config user.email "81552089+prysthedev@users.noreply.github.com"

uv sync              # na Linuxe sám vyberie torch s ROCm z indexu AMD
uv run ekg-check     # „zariadenie: ROCm – AMD Radeon RX 9070 XT“
uv run pytest -q
```

## Ako to funguje v `pyproject.toml`

```toml
dependencies = [
    "torch==2.2.2; sys_platform == 'darwin' and platform_machine == 'x86_64'",   # Intel Mac
    "torch[device-gfx1201]>=2.14; sys_platform == 'linux'",                      # PC (WSL2)
    "torch>=2.2.2; sys_platform == 'win32' or (sys_platform == 'darwin' and platform_machine != 'x86_64')",
    "triton; sys_platform == 'linux'", "rocm; sys_platform == 'linux'", ...   # závislosti ROCm buildu
]

[[tool.uv.index]]
name = "amd-rocm"
url = "https://stable.repo.amd.com/rocm/whl-next/"
explicit = true          # index sa použije IBA pre balíky z [tool.uv.sources]

[tool.uv.sources]
torch = { index = "amd-rocm", marker = "sys_platform == 'linux'" }
triton = { index = "amd-rocm" }
...
```

- `explicit = true`: index AMD obsahuje aj `numpy`, `pillow`, `jinja2` a ďalšie. Bez tohto nastavenia
  by ich uv bral od AMD (aj pre Mac). Takto idú z AMD iba balíky ROCm, všetko ostatné z PyPI.
- Balíky ROCm (`triton`, `rocm`, `rocm-sdk-*`, `amd-torch-device-*`) sú uvedené ako **priame**
  závislosti iba pre Linux, lebo uv uplatňuje `[tool.uv.sources]` len na priame závislosti
  projektu. Ako nepriame závislosti torchu by ich hľadal na PyPI a zamknutie zlyhá.
- `uv.lock` po zmene neobsahuje žiadne balíky NVIDIA CUDA (predtým 12 balíkov `nvidia-*-cu12`,
  ktoré by sa na Linuxe sťahovali zbytočne – rádovo GB, presnú veľkosť som nemeral).

## Rýchlosť tréningu: CPU vs. GPU

`uv run ekg-gpu-bench --steps 100 --warmup 20` – jeden tréningový krok = forward + backward + Adam,
dávka 64 záznamov (12 × 1000), náhodné dáta, skutočné modely projektu:

| model | CPU (Ryzen 7 7800X3D) | GPU float32 | GPU bfloat16 |
|---|---|---|---|
| autoencoder (Model A, 321 tis. param.) | 69,4 ms | 43,5 ms (1,6×) | **10,5 ms (6,6×)** |
| ResNet klasifikátor (Model B, 2,18 mil. param.) | 117,2 ms | 41,9 ms (2,8×) | **9,1 ms (12,9×)** |

V zátvorke je zrýchlenie oproti CPU. Časy na CPU kolíšu podľa záťaže počítača (v inom behu
klasifikátor 170 ms/krok), GPU časy boli v opakovaných behoch stabilné (±10 %).

**Prečo je float32 na GPU taký pomalý?** Knižnica MIOpen (AMD obdoba cuDNN) pre viaceré 1D
konvolúcie nášho tvaru (hlavne so `stride 2` a pri spätnom šírení) vyberá pomalé záložné algoritmy
(`GemmFwdRest`, `GemmBwdRest`, `ConvDirectNaive*`) – overené cez `MIOPEN_LOG_LEVEL=6`. Na jeden krok
klasifikátora sa tak spúšťa ~1 500 malých kernelov a čas rastie lineárne s dávkou (autoencoder: dávka 256
trvala ~4,5× dlhšie ako dávka 64, počet záznamov za sekundu sa teda nezlepšil – 1 221 vs. 1 386). Úplné vyhľadávanie (`MIOPEN_FIND_MODE=NORMAL`,
`torch.backends.cudnn.benchmark = True`) nepomohlo, vypnutie MIOpen (`cudnn.enabled = False`)
bolo ešte 2× pomalšie.

**Riešenie: zmiešaná presnosť bfloat16** (`torch.autocast`). Výpočty v sieti bežia v 16-bitovej
presnosti, pre ktorú má RDNA 4 rýchle maticové jednotky a MIOpen rýchle algoritmy, váhy a krok
optimalizátora ostávajú v 32-bitovej presnosti. bfloat16 má rovnaký rozsah ako float32, takže
na rozdiel od float16 netreba `GradScaler` (float16 bol rovnako rýchly: 8,3 ms/krok).
→ **Odporúčanie pre #16:** trénovať klasifikátor s `torch.autocast("cuda", dtype=torch.bfloat16)`
a overiť, že výsledky (val loss, AUROC) sú rovnaké ako vo float32.

## Problémy počas inštalácie (a riešenia)

| problém | príčina | riešenie |
|---|---|---|
| prvé `uv sync` sťahovalo ~0,5 MB/s viac ako 10 min | `uv.lock` mal pre Linux torch 2.2.2 z PyPI s 12 balíkmi NVIDIA CUDA (rádovo GB), na AMD nepoužiteľné | zastavené, torch pre Linux z indexu AMD |
| test s indexom PyTorch sťahoval ďalšie GB | `torch+rocm10.0` z PyPI indexu PyTorch vyžaduje `rocm[device-all]` (všetky GPU) | index AMD s `torch[device-gfx1201]` |
| `ImportError: libgomp.so.1` | čisté Ubuntu nemá knižnicu OpenMP | `sudo apt install libgomp1` |
| `miopenStatusUnknownError` pri BatchNorm na GPU | MIOpen kompiluje kernel až pri prvom použití a potrebuje systémové hlavičky C (`stdint.h`), ktoré čisté Ubuntu nemá | `sudo apt install libc6-dev` |
| `uv lock`: „no version of triton==3.8.0+…rocm10.1.0“ | `explicit` index sa uplatní iba na priame závislosti | balíky ROCm pridané ako priame závislosti pre Linux |
| float32 na GPU iba 1,6–2,8× rýchlejší ako CPU | MIOpen volí pomalé algoritmy pre naše 1D konvolúcie | bfloat16 autocast (6,6–12,9×) |

Neúspešný predpoklad: myslel som si, že `uv lock` zlyhal iba pre Python 3.12 (tak to hlásil prvý
pokus). Po obmedzení na 3.11 zlyhal rovnako – skutočná príčina bola v `explicit` indexe (tabuľka vyššie).

Ďalšie obmedzenia WSL: profiler PyTorch nezaznamenáva časy na GPU (iba na CPU) a nástroje
ROCm profiler/debugger nie sú podporované. `pin_memory=True` v `DataLoader` funguje bez varovaní.

## Synchronizácia Mac ↔ PC

- **Kód – git:** všetko ide cez GitHub. Pred prácou `git pull` (a `uv sync`, ak sa zmenil
  `uv.lock`), po práci commit + push vo vetve issue. Na každom počítači je vlastný klon repozitára.
- **Dáta – nesynchronizujú sa, sťahujú sa:** `uv run ekg-download` (overí SHA256) →
  `uv run ekg-preprocess` → `uv run ekg-check-data`. Preprocessing je deterministický a normalizácia
  sa počíta z train množiny podľa manifestu v gite (`manifests/`), takže výsledok je na oboch
  počítačoch rovnaký. Dáta sú mimo gitu (`data/` v `.gitignore`).
- **Natrénované modely – GitHub Releases:** checkpointy nejdú do gitu. Každý natrénovaný model sa
  nahrá ako release `model-<run_name>` (súbory `best.pt` a prahy), napr.
  `gh release download model-ae_v1 --repo SPSE-Zochova-2627-4c/Nosko_EKG-classification -D checkpoints/ae_v1`.
- **Testy na oboch počítačoch:** testy, ktoré potrebujú GPU, dataset alebo checkpoint, sa bez nich
  preskočia (`skip`), takže `uv run pytest` prejde na Macu aj na PC.

## Záloha: Google Colab / Kaggle

[`notebooks/colab_trening.ipynb`](../notebooks/colab_trening.ipynb) naklonuje repozitár,
nainštaluje projekt **bez závislostí** (`pip install --no-deps -e .` – použije sa predinštalovaný
PyTorch s CUDA, nie ROCm z `uv.lock`), stiahne dáta, spustí `ekg-check`, `ekg-gpu-bench`
a tréning. Checkpointy sa dajú ukladať na Google Drive cez `EKG_CHECKPOINT_DIR`.
