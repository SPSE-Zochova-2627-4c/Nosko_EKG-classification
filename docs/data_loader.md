# Dátový loader

Kód: [`src/ekg/data/dataset.py`](../src/ekg/data/dataset.py) · Kontrola: `uv run ekg-check-data`
([`check_pipeline.py`](../src/ekg/data/check_pipeline.py))

## Celá dátová pipeline

```
PTB-XL (WFDB, 100 Hz)                        #3  ekg-download
  └─ cache signálov (N, 12, 1000)            #8  ekg-preprocess
       └─ manifest: split + labely + NORM    #4, #6  ekg-labels, ekg-splits
            └─ preprocessing v1.0 (iba train parametre)   #8
                 └─ EKGDataset / DataLoader  #9  → dávky pre Model A a Model B
```

## Použitie

```python
from ekg.data.dataset import make_loader

train_b = make_loader("train", "classifier", batch_size=64, seed=42)  # Model B, miešané
val_b = make_loader("val", "classifier")  # bez miešania
train_a = make_loader("train", "strict_norm", batch_size=64)  # Model A – iba prísny NORM
val_all = make_loader("val", "all")  # vyhodnotenie autoencodera

for batch in train_b:
    batch["signal"]  # (B, 12, 1000) float32, po preprocessingu
    batch["target"]  # (B, 5) float32 – MI, STTC, CD, HYP, AFIB
    batch["ecg_id"]  # (B,) int64 – na dohľadanie záznamu
```

| Podmnožina | train | val | test |
|---|---|---|---|
| `classifier` (Model B) | 17 135 | 2 150 | 2 166 |
| `strict_norm` (Model A) | 3 905 | 454 | 464 |

## Rozhodnutia

- **Preload do pamäte:** celá podmnožina sa pri vytvorení raz spracuje a drží v RAM (train Modelu B
  ≈ 0,8 GB). Tréning potom nečaká na disk ani na filter. Pri malej pamäti sa dá vypnúť (`preload=False`).
- **Reprodukovateľnosť:** poradie miešania určuje `seed` (vlastný `torch.Generator`), workery dostanú
  odvodený seed. Validation a test sa **nikdy nemiešajú**: `shuffle=True` pre ne vyhodí chybu.
- **Test množina** je zamknutá rovnako ako v `split_ids` (`allow_test=True`).
- Posledná neúplná dávka sa nezahadzuje, takže v epoche sa vyhodnotí každý záznam.

## Kontrolný priechod (`uv run ekg-check-data`)

Prejde všetky dávky v train, val aj test (`classifier` a `strict_norm`) a overí:

| Kontrola | Čo by odhalila |
|---|---|
| tvar `(B, 12, 1000)`, float32, konečné hodnoty | chybu preprocessingu |
| ciele iba 0/1 a zhodné s manifestom pre dané `ecg_id` | pomiešané labely |
| signál z loadera = záznam načítaný priamo z WFDB + preprocessing (25 náhodných z každej podmnožiny) | cache s pomiešaným poradím záznamov |
| každý záznam práve raz za epochu | duplicitné alebo chýbajúce záznamy |
| prísny NORM má všetky ciele 0 | chybu výberu NORM |
| žiadny pacient vo viacerých množinách | únik medzi množinami |
| rovnaký seed → rovnaké poradie, validation vždy rovnako | nereprodukovateľný tréning |

Test množina sa tu otvára **iba na technickú kontrolu** (tvary a väzby). Nevyhodnocuje sa na nej žiadny model.

Výsledok (Intel Mac, ~26 s): všetky kontroly prešli. Testy `tests/test_dataset.py` navyše dokazujú,
že kontrola naozaj odhalí pomiešanú cache aj nesprávne labely.
