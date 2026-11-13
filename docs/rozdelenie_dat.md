# Rozdelenie dát na train / validation / test (verzia 1.0)

Kód: [`src/ekg/data/splits.py`](../src/ekg/data/splits.py) · Manifest:
[`manifests/splits_v1.0.csv`](../manifests/splits_v1.0.csv) · Report: `uv run ekg-splits`

## Pravidlo

Používa sa odporúčané rozdelenie autorov PTB-XL podľa stĺpca `strat_fold`:

| Množina | Foldy | Na čo |
|---|---|---|
| train | 1–8 | trénovanie modelov, normalizačné parametre (#8), váhy tried (#16) |
| val | 9 | early stopping, výber prahov (#13, #17), porovnanie variantov |
| test | 10 | **iba** záverečné vyhodnotenie (#22) |

**Prečo `strat_fold` a nie vlastné náhodné rozdelenie:**
- Foldy sú rozdelené **podľa pacientov**. Všetky záznamy jedného pacienta sú v jednom folde, takže
  model nemôže „spoznať“ pacienta z train a dostať tak nadhodnotený výsledok.
- Foldy sú **stratifikované** podľa labelov, veku a pohlavia, preto majú množiny podobné zastúpenie tried.
- Foldy 9 a 10 majú anotácie **overené človekom**, čo dáva spoľahlivejšie vyhodnotenie.
- Rovnaké rozdelenie používa benchmark Strodthoff et al. (2020), takže výsledky sa dajú porovnať.

## Výsledok

| | záznamy | pacienti | Model B | prísny NORM |
|---|---|---|---|---|
| train | 17 418 | 15 023 | 17 135 | 3 905 |
| val | 2 183 | 1 942 | 2 150 | 454 |
| test | 2 198 | 1 904 | 2 166 | 464 |

**Prekrývanie pacientov medzi množinami: 0.** Overuje to `check_no_leak()` pri každom vytvorení
aj načítaní manifestu.

Pozitívne labely v Modeli B (podiel záznamov v množine):

| Label | train | val | test |
|---|---|---|---|
| MI | 4 379 (25,6 %) | 540 (25,1 %) | 550 (25,4 %) |
| STTC | 4 186 (24,4 %) | 528 (24,6 %) | 521 (24,1 %) |
| CD | 3 907 (22,8 %) | 495 (23,0 %) | 496 (22,9 %) |
| HYP | 2 119 (12,4 %) | 268 (12,5 %) | 262 (12,1 %) |
| AFIB | 1 211 (7,1 %) | 151 (7,0 %) | 152 (7,0 %) |

Podiely sa medzi množinami líšia najviac o 0,5 percentuálneho bodu, stratifikácia teda funguje.

## Manifest

`manifests/splits_v1.0.csv` (v gite, 1 riadok = 1 záznam):

| Stĺpec | Význam |
|---|---|
| `ecg_id`, `patient_id`, `strat_fold`, `filename_lr` | z PTB-XL |
| `split` | `train` / `val` / `test` |
| `MI`, `STTC`, `CD`, `HYP`, `AFIB` | ciele Modelu B (0/1), mapovanie v1.0 (#4) |
| `usable_classifier` | záznam patrí do Modelu B |
| `strict_norm` | záznam patrí do tréningu/kalibrácie autoencodera |
| `mapping_version` | verzia mapovania labelov |

Manifest je v gite, aby každý tréning a vyhodnotenie používali presne tie isté záznamy, aj na
inom počítači. `uv run ekg-splits --check` (a test) overí, že manifest stále zodpovedá dátam a kódu.

V kóde:

```python
from ekg.data.splits import split_ids

train_ids = split_ids("train", subset="classifier")  # Model B
norm_ids = split_ids("train", subset="strict_norm")  # Model A
val_ids = split_ids("val")
```

## Zámok test množiny

`split_ids("test")` vyhodí `TestSetLockedError`. Test množinu dostaneš iba cez
`split_ids("test", allow_test=True)` a to sa má objaviť **iba** v záverečnom vyhodnotení (#22).
Ide o ochranu pred nechceným „ladením na test dátach“: každé rozhodnutie (architektúra, prahy,
early stopping) sa robí na validation.
