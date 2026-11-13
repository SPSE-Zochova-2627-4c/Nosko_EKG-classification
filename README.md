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
├── manifests/      # rozdelenie train/val/test (v gite)
├── data/           # dataset – mimo gitu, pozri data/README.md
├── checkpoints/    # natrénované modely – mimo gitu
└── outputs/        # reporty, grafy, predikcie – mimo gitu
```

Umiestnenie dát, checkpointov a výstupov sa dá zmeniť premennými prostredia
`EKG_DATA_DIR`, `EKG_CHECKPOINT_DIR` a `EKG_OUTPUT_DIR`.

## Dataset

[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) – 21 799 záznamov od 18 869 pacientov,
licencia CC BY 4.0.

```bash
uv run ekg-download       # stiahne ZIP (1,84 GB), rozbalí 100 Hz časť (~0,6 GB), overí SHA256
uv run ekg-verify-data    # načíta všetky záznamy a overí tvar (12, 1000)
```

Dáta skončia v `data/raw/ptb-xl/`. Podrobnosti sú v [data/README.md](data/README.md).

Mapovanie labelov a rozdelenie dát:

```bash
uv run ekg-labels         # počty labelov a prísneho NORM (docs/mapovanie_labelov.md)
uv run ekg-splits --check # overí manifest manifests/splits_v1.0.csv (docs/rozdelenie_dat.md)
uv run ekg-preprocess     # cache signálov + normalizácia z train (docs/preprocessing.md)
uv run ekg-check-data     # kontrolný priechod celou dátovou pipeline (docs/data_loader.md)
uv run ekg-ae-info        # architektúra autoencodera (docs/model_a_autoencoder.md)
uv run ekg-clf-info       # architektúra klasifikátora (docs/model_b_klasifikator.md)
uv run ekg-train-ae       # tréning autoencodera, --resume po prerušení (docs/trening_autoencodera.md)
uv run ekg-ae-report      # chyba autoencodera na validation, AUROC NORM vs. patológia
uv run ekg-calibrate-ae   # prah anomálie na validation (docs/prah_anomalie.md)
```

V kóde:

```python
from ekg.data import ptbxl

db = ptbxl.load_database()  # anotácie, index = ecg_id
x = ptbxl.load_signal(db.loc[1, "filename_lr"])  # np.ndarray (12, 1000), float32
```

Citácia: Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022).
PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet.
https://doi.org/10.13026/kfzx-aw45

## Vývoj

```bash
uv run ruff check .      # kontrola kódu
uv run ruff format .     # formátovanie
uv add <balík>           # pridanie novej knižnice
```

Rozhodnutia a ich dôvody (aj neúspešné pokusy) sa zapisujú do [docs/dennik.md](docs/dennik.md).
