# Kontrola kvality EKG záznamov (verzia 1.0)

Kód: [`src/ekg/quality.py`](../src/ekg/quality.py) · Výsledok pre PTB-XL:
[`manifests/quality_v1.0.csv`](../manifests/quality_v1.0.csv) · Spustenie: `uv run ekg-quality`

## Dve úrovne kontroly

| Úroveň | Čo kontroluje | Výsledok |
|---|---|---|
| **technická platnosť** (kontrakt #2) | tvar (12, 1000), 100 Hz, číselné a konečné hodnoty | neplatný vstup → `InvalidInputError`, **záznam sa nespracuje** |
| **kvalita signálu** (tento modul) | plochý zvod, extrémna amplitúda, saturácia, anotácie šumu | `ok` alebo `warning` so zoznamom problémov, **záznam sa nikdy neodmietne** |

**Prečo kvalita záznam neodmieta:** aj zašumené EKG môže obsahovať diagnosticky užitočnú informáciu.
Rozhodnutie, či výsledku veriť, patrí človeku, ktorý vo výstupe vidí `quality.status = "warning"`
a konkrétne problémy. Na tréning autoencodera sa zašumené záznamy aj tak nepoužívajú, lebo ich vylučuje
výber prísneho NORM (#4).

## Pravidlá

### Na signáli (pre akýkoľvek vstup, signál v mV pred preprocessingom)

| Pravidlo | Podmienka | Čo znamená |
|---|---|---|
| plochý zvod | rozkmit (max − min) < **0,05 mV** | odpojená elektróda alebo chýbajúci zvod |
| extrémna amplitúda | \|x\| > **10 mV** | artefakt (pohyb, elektróda). 99,99. percentil všetkých zvodov PTB-XL je 9,6 mV |
| saturácia | ≥ **10 po sebe idúcich vzoriek (100 ms)** presne na maxime \|x\| > 1 mV | orezaný signál (prebudený zosilňovač) |

### Z anotácií PTB-XL (iba pre záznamy z datasetu)

`static_noise` (statický šum), `burst_noise` (nárazový šum), `electrodes_problems` (problém s elektródami),
`baseline_drift` (drift základnej línie). Text sa vyčistí (napr. `" , I-V1,  "` → `I-V1`,
nemecké `alles` → „všetky zvody“) a pridá sa k problému.

Príklad výstupu: `{"status": "warning", "issues": ["plochý zvod: V5", "problém s elektródami (V4, V5)"]}`

## Výsledky pre PTB-XL

| | train | val | test | spolu |
|---|---|---|---|---|
| **ok** | 13 472 | 1 639 | 1 670 | 16 781 (77 %) |
| **warning** | 3 946 | 544 | 528 | 5 018 (23 %) |
| **vyradené** | 0 | 0 | 0 | **0** |

| Problém | záznamy |
|---|---|
| plochý zvod | 1 |
| extrémna amplitúda > 10 mV | 25 |
| saturácia | 0 |
| statický šum (anotácia) | 3 260 |
| nárazový šum (anotácia) | 612 |
| problém s elektródami (anotácia) | 30 |
| drift základnej línie (anotácia) | 1 598 |

### Overenie pravidiel na signáli pomocou anotácií

Pravidlá na signáli nevidia anotácie PTB-XL, a predsa:
- **Jediný plochý zvod** v celom datasete (`ecg_id 12722`, V5) má v PTB-XL poznámku „problém s elektródami V4, V5“.
- Zo **25 záznamov s amplitúdou > 10 mV má 17 anotáciu šumu** (najmä nárazový šum, ktorý sa prejavuje
  krátkymi vysokými výkyvmi).
- Celkovo **18 z 26** záznamov označených pravidlami na signáli (69 %) potvrdzuje aj anotácia.
  Zvyšných 8 sú pravdepodobne artefakty, ktoré anotátor nezaznamenal (rovnaký jav ako falošný poplach v #14).

### Neúspešný prvý pokus (saturácia)

Prvá verzia pravidla saturácie hľadala **≥ 5 vzoriek (50 ms)** na maxime a označila 3 záznamy. Kontrola
ukázala, že ide o **falošné poplachy**:
- `ecg_id 12592` (V5): signál klesá k minimu a stúpa plynulo. Pri rozlíšení PTB-XL 0,001 mV je dno pomalej
  vlny jednoducho 5 vzoriek rovnaké.
- `ecg_id 2974` (V2): „plató“ je na úplnom začiatku záznamu, nie na ostrom vrchole.

Pravidlo sa preto sprísnilo na **≥ 10 vzoriek (100 ms)**. Taký dlhý úplne plochý úsek pri amplitúde nad 1 mV
je fyziologicky nemožný. V PTB-XL sa saturácia nevyskytuje. Pravidlo zostáva pre iné prístroje
(napr. externá validácia #31) a je otestované na umelých dátach.

## Použitie

- **Výstup systému:** pole `quality` v kontrakte (#2) plní `quality.assess(signal, anotácie)` v inference pipeline (#19).
- **Analýza:** `manifests/quality_v1.0.csv` (status, príznaky po typoch, dôvody) využije analýza
  výsledkov podľa kvality signálu (#26).
- Každé označenie má v CSV aj **dôvod**. Test overuje, že žiadny záznam nie je označený bez dôvodu
  a žiadny nie je vyradený.
