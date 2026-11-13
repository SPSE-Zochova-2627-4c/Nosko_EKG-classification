# Detekcia anomálií a klasifikácia EKG

Maturitný projekt – Matúš Nosko, IV.C, SPŠE Zochova 9 Bratislava (odbor umelá inteligencia).

Systém spracuje 10-sekundový 12-zvodový EKG záznam (PTB-XL, 100 Hz) dvoma modelmi:

- **Model A – 1D konvolučný autoencoder** trénovaný iba na normálnych záznamoch;
  vysoká rekonštrukčná chyba znamená anomáliu.
- **Model B – multi-label klasifikátor** (1D ResNet / InceptionTime) s piatimi nezávislými
  sigmoid výstupmi: MI, STTC, CD, HYP, AFIB.

> ⚠️ Ide o výskumný a vzdelávací nástroj, **nie o lekársku diagnostiku**.

Plán práce je v [GitHub Project boarde](https://github.com/orgs/SPSE-Zochova-2627-4c/projects/18)
a v [issues](https://github.com/SPSE-Zochova-2627-4c/Nosko_EKG-classification/issues).

## Inštalácia

Projekt používa [uv](https://docs.astral.sh/uv/) – ten nainštaluje správnu verziu Pythonu (3.11)
aj všetky knižnice podľa `uv.lock`, takže prostredie je na každom počítači rovnaké.

1. Nainštaluj uv (macOS / Linux):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Na Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.
   Potom otvor nový terminál.

2. Naklonuj repozitár a nainštaluj závislosti:

   ```bash
   git clone https://github.com/SPSE-Zochova-2627-4c/Nosko_EKG-classification.git
   cd Nosko_EKG-classification
   uv sync
   ```

3. Over prostredie (nepotrebuje dataset):

   ```bash
   uv run ekg-check
   ```

   Na konci má vypísať `Prostredie je pripravené.` Rovnakú kontrolu spúšťajú aj testy:

   ```bash
   uv run pytest
   ```

### Poznámka k PyTorch

| Počítač | PyTorch | Použitie |
|---|---|---|
| Intel Mac | 2.2.2, iba CPU (novšie verzie pre Intel Mac neexistujú) | písanie kódu, testy, malé pokusy |
| PC s AMD RX 9070 XT (Linux, ROCm) | novšia verzia s ROCm – issue #28 | trénovanie |
| Google Colab / Kaggle | predinštalovaná verzia s CUDA | záloha na trénovanie |

Kód sa preto píše tak, aby fungoval s PyTorch 2.2 aj novším.

## Štruktúra

```
├── src/ekg/        # zdrojový kód (Python balík `ekg`)
├── tests/          # testy (pytest)
├── configs/        # konfigurácie experimentov
├── notebooks/      # Jupyter notebooky na prieskum dát
├── docs/           # dokumentácia a denník rozhodnutí
├── data/           # dataset – mimo gitu, pozri data/README.md
├── checkpoints/    # natrénované modely – mimo gitu
└── outputs/        # reporty, grafy, predikcie – mimo gitu
```

Umiestnenie dát, checkpointov a výstupov sa dá zmeniť premennými prostredia
`EKG_DATA_DIR`, `EKG_CHECKPOINT_DIR` a `EKG_OUTPUT_DIR`.

## Dataset

[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) – 21 799 záznamov od 18 869 pacientov.
Sťahovanie pripravuje issue #3.

## Vývoj

```bash
uv run ruff check .      # kontrola kódu
uv run ruff format .     # formátovanie
uv add <balík>           # pridanie novej knižnice
```

Rozhodnutia a ich dôvody (aj neúspešné pokusy) sa zapisujú do [docs/dennik.md](docs/dennik.md).
