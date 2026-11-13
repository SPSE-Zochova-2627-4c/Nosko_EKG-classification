# Mapovanie anotácií PTB-XL na labely (verzia 1.0)

Kód: [`src/ekg/data/labels.py`](../src/ekg/data/labels.py) · Tabuľka všetkých 71 SCP kódov:
[`mapovanie_labelov_v1.0.csv`](mapovanie_labelov_v1.0.csv) · Prehľad počtov: `uv run ekg-labels`

## Ako sú anotácie v PTB-XL uložené

Každý záznam má v `ptbxl_database.csv` stĺpec `scp_codes`, napr. `{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}`
(SCP kód → *likelihood*, istota kardiológa v %). `scp_statements.csv` opisuje 71 kódov troch typov:

| Typ | Počet | Príklady | Likelihood |
|---|---|---|---|
| diagnostický | 44 | NORM, IMI, NDT, CLBBB, LVH | väčšinou vyplnená (medián 100) |
| form (tvar) | 19 | PVC, STD_, INVT, LVOLT | väčšinou 0 |
| rytmický | 12 | SR, **AFIB**, AFLT, STACH | takmer vždy 0 |

Diagnostické kódy majú `diagnostic_class` = jedna z 5 supertried: NORM, MI, STTC, CD, HYP.

## Labely pre Model B

| Label | Pravidlo | Kódy |
|---|---|---|
| MI | ktorýkoľvek diagnostický kód s `diagnostic_class = MI` | 14 (IMI, ASMI, …) |
| STTC | … `diagnostic_class = STTC` | 13 (NDT, NST_, ISC_, …) |
| CD | … `diagnostic_class = CD` | 11 (LAFB, IRBBB, CLBBB, …) |
| HYP | … `diagnostic_class = HYP` | 5 (LVH, RVH, …) |
| AFIB | rytmický kód `AFIB` | 1 |

- **Likelihood sa nepoužíva** (kód sa počíta aj s likelihood 0 alebo 15). Rovnako postupuje
  benchmark Strodthoff et al. (2020), takže výsledky sa dajú porovnať. Prísnejší prah
  (napr. ≥ 50) je možný neskorší experiment.
- **AFIB nie je supertrieda**, ale rytmický kód. Likelihood má takmer vždy 0
  (1 466 z 1 514 záznamov), preto by akýkoľvek prah AFIB zničil.
- **Flutter (AFLT, 73 záznamov) sa do AFIB nezaratúva**: je to iná arytmia (pravidelná).
- NORM nie je výstup Modelu B. NORM záznam má všetkých 5 labelov = 0.
- **Vynecháva sa 348 záznamov**, ktoré nemajú žiadny diagnostický kód ani AFIB, pretože nevieme,
  či sú normálne.
- 37 záznamov má súčasne NORM aj AFIB (NORM opisuje tvar, AFIB rytmus), tie majú AFIB = 1.

### Počty pozitívnych záznamov (Model B: 21 451 záznamov)

| Label | train (fold 1–8) | val (9) | test (10) | spolu |
|---|---|---|---|---|
| MI | 4 379 | 540 | 550 | 5 469 |
| STTC | 4 186 | 528 | 521 | 5 235 |
| CD | 3 907 | 495 | 496 | 4 898 |
| HYP | 2 119 | 268 | 262 | 2 649 |
| AFIB | 1 211 | 151 | 152 | 1 514 |

HYP a AFIB sú najzriedkavejšie, čo treba riešiť pri tréningu (#16).

### Vplyv prahu likelihood (zvažované varianty)

| Prah | MI | STTC | CD | HYP | záznamov bez labelu |
|---|---|---|---|---|---|
| **≥ 0 (zvolené)** | **5 469** | **5 235** | **4 898** | **2 649** | **348** |
| ≥ 50 | 4 134 | 5 078 | 4 891 | 2 258 | 1 311 |
| = 100 | 2 975 | 4 565 | 4 778 | 1 468 | 4 320 |

## Prísny NORM pre autoencoder (Model A)

Autoencoder sa má naučiť iba „čisté“ normálne EKG. Záznam je **prísny NORM**, ak spĺňa všetko:

1. NORM s likelihood 100,
2. žiadny iný diagnostický kód,
3. z rytmických kódov iba SR (vylúči napr. AFIB, sínusovú tachykardiu, bradykardiu),
4. bez `static_noise`, `burst_noise`, `electrodes_problems`,
5. bez `baseline_drift`, `extra_beats`, `pacemaker`,
6. bez form kódov (PVC, PAC, ST depresia, inverzia T vlny, nízka voltáž …): sú to odchýlky,
   ktoré by autoencoder nemal považovať za normu.

`validated_by_human` sa nevyžaduje: validation a test (foldy 9 a 10) sú v PTB-XL overené
človekom celé a v train by sme stratili ďalších ~19 % záznamov.

| Prvý dôvod vylúčenia | train | val | test |
|---|---|---|---|
| nie je NORM s likelihood 100 | 11 675 | 1 475 | 1 477 |
| iný diagnostický kód | 136 | 14 | 18 |
| rytmus iný ako SR | 299 | 34 | 38 |
| šum alebo problém s elektródami | 820 | 106 | 93 |
| baseline drift, extra beats alebo kardiostimulátor | 404 | 53 | 71 |
| form kód | 179 | 47 | 37 |
| **prísny NORM** | **3 905** | **454** | **464** |

Pôvodný odhad „~9 500 NORM“ zodpovedá všetkým záznamom s kódom NORM. Po prísnom výbere zostáva
v train približne 4 tisíc, čo na 1D konvolučný autoencoder stačí.

## Overené príklady (testy `tests/test_labels.py`)

| Príklad | MI | STTC | CD | HYP | AFIB | Prísny NORM |
|---|---|---|---|---|---|---|
| `NORM 100, SR` | 0 | 0 | 0 | 0 | 0 | áno |
| `NORM 100, SR` + static_noise | 0 | 0 | 0 | 0 | 0 | nie (šum) |
| `NORM 100, IMI 15, SR` | 1 | 0 | 0 | 0 | 0 | nie (iný diag. kód) |
| `AFIB 0, NDT 100` | 0 | 1 | 0 | 0 | 1 | nie |
| `IMI 100, LVH 50, CLBBB 100` | 1 | 0 | 1 | 1 | 0 | nie |
| `AFLT 0` | 0 | 0 | 0 | 0 | 0 | – (vynechaný z Modelu B) |
| `NORM 100, AFIB 0` | 0 | 0 | 0 | 0 | 1 | nie (rytmus) |
| `NORM 100, SR, PVC` | 0 | 0 | 0 | 0 | 0 | nie (form kód) |

Testy navyše na skutočnom PTB-XL overujú presné počty v tabuľkách vyššie a to, že CSV tabuľka
zodpovedá kódu.

## Verzovanie

Akákoľvek zmena pravidiel = nová verzia (`MAPPING_VERSION`) a nový súbor
`mapovanie_labelov_v<verzia>.csv`. Verzia sa ukladá k modelom a výsledkom, aby bolo jasné,
s akými labelmi vznikli.
