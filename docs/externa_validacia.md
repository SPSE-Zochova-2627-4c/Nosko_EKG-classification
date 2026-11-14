# Externá validácia na datasete Georgia (issue #31)

Ako funguje **zmrazený systém** (#22) na dátach z inej krajiny, nemocnice a prístroja? Nič sa
nedotrénovalo ani neladilo – rovnaké modely, preprocessing a prahy ako pri teste na PTB-XL.

## Dataset

| | |
|---|---|
| názov | The Georgia 12-lead ECG Challenge (G12EC) Database – súčasť PhysioNet/CinC Challenge 2021 |
| verzia | Challenge 2021 **v1.0.3**, DOI [10.13026/34va-7q14](https://doi.org/10.13026/34va-7q14) |
| licencia | **CC BY 4.0** (rovnako ako PTB-XL) |
| pôvod | Emory University, Atlanta (USA) – PTB-XL je z Nemecka |
| záznamy | 10 344, 12 zvodov, **500 Hz**, 5 – 10 s, diagnózy ako kódy SNOMED CT |
| stiahnuté | 1,3 GB do `data/raw/georgia/`, všetkých 20 699 súborov overených SHA256 voči oficiálnemu `SHA256SUMS.txt` |

**Prečo Georgia:** z troch datasetov v zadaní (Chapman, CPSC 2018, Georgia) má jediná zároveň
10 s záznamy a diagnózy pre AFIB, poruchy vedenia, hypertrofiu aj ST-T zmeny. Chapman nemá
ischémiu podľa lokalizácie ani zväčšenie predsiení a LVH má iba v 15 záznamoch, CPSC má
záznamy 6 – 144 s. **Infarkt však Georgia takmer nemá
(7 záznamov)** – MI sa preto hodnotiť nedá (rozhodol Matúš pri výbere datasetu).

Stiahnutie (paralelne, s overením):

```bash
cd data/raw/georgia
U=https://physionet.org/files/challenge-2021/1.0.3
curl -sSfLO $U/SHA256SUMS.txt && curl -sSfLO $U/LICENSE.txt
grep " training/georgia/" SHA256SUMS.txt | sed 's| training/georgia/| |' > georgia.sha256
awk '{print $2}' georgia.sha256 | xargs -P 12 -I{} sh -c \
  'mkdir -p "$(dirname {})" && curl -sSf --retry 5 -o {} '$U'/training/georgia/{}'
sha256sum -c --quiet georgia.sha256 && echo OK
uv run ekg-external          # ~21 min, väčšinu zaberá bootstrap (1000× pre 6 metrík)
```

## Mapovanie diagnóz na labely

Kód Georgia sa zaratá do labelu, iba ak zodpovedá **diagnostickému kódu PTB-XL v tej istej
supertriede** ([mapovanie_labelov.md](mapovanie_labelov.md)). Celá tabuľka s počtami:
[`mapovanie.csv`](vysledky/externa/mapovanie.csv).

| label | pozitívne kódy Georgia (záznamov) | nejasné – label sa nehodnotí |
|---|---|---|
| AFIB | AF 570, AFAFL 2 (flutter **nie**, ako v PTB-XL) | – |
| MI | MI 7 | Q vlny (QAb 464) – v PTB-XL iba tvarový kód |
| STTC | nešpecifické ST-T 1 883, dlhé QT 1 391, ischémia laterálna 903 / dolná 451 / predná 281 | abnormálna T 2 306, inverzia T 812, abnormálny ST 992, ST depresia / elevácia |
| CD | 1AVB 769, RBBB 542 + 28, IRBBB 407, LBBB 231, NSIVCB 203, LAFB 180, BBB 116, … | predĺžené PR |
| HYP | LVH 1 232, LAE 870, RVH 86, LAA 72, VH 71, … | vysoké napätie ĽK (LVHV) |

- **Nejasný** = v PTB-XL je to iba „tvarový“ nález (form statement), nie diagnóza. Záznam s takým
  nálezom a bez jasného pozitívneho kódu sa pre ten label nepočíta ani ako pozitívny, ani ako
  negatívny (STTC: 780 záznamov, MI: 459).
- **Model B** hodnotí záznamy, pri ktorých vieme všetky labely: aspoň jeden pozitívny label, alebo
  iba bežný rytmus (sínusový, bradykardia, tachykardia, sínusová arytmia) a nejasné nálezy.
  Záznamy iba s nenamapovanými kódmi (napr. odchýlka osi, extrasystoly) sa vynechajú – rovnako ako
  PTB-XL vynecháva záznamy bez diagnostického kódu.
- **Model A:** NORM = záznam **iba so sínusovým rytmom** (Georgia nemá kód „normálne EKG“),
  patológia = aspoň jeden pozitívny label. Nie je to presne prísny NORM z PTB-XL (ten vyžaduje
  „normálne EKG“ so 100 % istotou a bez šumu).
- **Nezhody systémov:** Georgia kóduje odchýlku osi (LAD 937 záznamov) zvlášť, PTB-XL ju
  nepovažuje za diagnózu; Georgia rozlišuje ischémiu podľa lokalizácie, ale nemá „starý infarkt“.

## Príprava signálu

- 500 Hz → 100 Hz polyfázovým filtrom s antialiasingom (`scipy.signal.resample_poly`), prvých 10 s
  (5000 → 1000 vzoriek). Potom uložený train preprocessing (#5) – nič sa nepočíta nanovo.
- Test overuje, že prevzorkovanie zachová 5 Hz sínus a odstráni 120 Hz (nad Nyquistom 50 Hz).
- **Vynechaných 58 záznamov:** 52 kratších ako 10 s (doplnenie by vytvorilo umelý plochý úsek)
  a 6, ktoré odmietol kontrakt (#2) pre NaN v signáli. **Vyhodnotených 10 286.**
- Ako bol z 500 Hz vyrobený PTB-XL 100 Hz, nie je zdokumentované – iný filter pri prevzorkovaní je
  možný malý zdroj rozdielu.

## Výsledky

Bootstrap 1000× **po záznamoch** – Georgia nemá identifikátor pacienta, takže ak má jeden pacient
viac záznamov, intervaly sú o niečo užšie, než by mali byť.
Podrobne: [`porovnanie_ptbxl.csv`](vysledky/externa/porovnanie_ptbxl.csv),
[`metriky_po_labeloch.csv`](vysledky/externa/metriky_po_labeloch.csv),
[`report.json`](vysledky/externa/report.json).

### Model A (autoencoder) – 1 734 NORM vs. 6 509 patológií

| | PTB-XL test | Georgia |
|---|---|---|
| AUROC | 0,796 (0,771 – 0,820) | **0,797** (0,785 – 0,808) |
| senzitivita | 0,586 | 0,694 |
| špecificita | 0,858 | **0,765** |

### Model B (klasifikátor) – 9 474 záznamov

| label | pozitívnych | AUROC PTB-XL | AUROC Georgia (95 % CI) | F1 PTB-XL → Georgia | recall | precision |
|---|---|---|---|---|---|---|
| STTC | 4 053 | 0,933 | **0,830** (0,821 – 0,838) | 0,765 → 0,689 | 0,764 → 0,611 | 0,765 → 0,789 |
| CD | 2 227 | 0,917 | **0,896** (0,889 – 0,904) | 0,767 → 0,681 | 0,752 → 0,800 | 0,784 → 0,593 |
| HYP | 2 196 | 0,903 | **0,812** (0,802 – 0,823) | 0,606 → 0,567 | 0,737 → 0,527 | 0,515 → 0,612 |
| AFIB | 568 | 0,981 | **0,927** (0,911 – 0,942) | 0,870 → 0,676 | 0,855 → 0,683 | 0,884 → 0,669 |
| MI | **7** | 0,910 | 0,728 (0,517 – 0,866) | nehodnotiteľné | 6 / 7 | 0,002 |

Macro AUROC štyroch hodnotiteľných labelov: PTB-XL test 0,934 → Georgia **0,866**.

## Prečo je výkon nižší

**1. Iný signál (posun domény).** Porovnanie normálnych záznamov – prísny NORM PTB-XL (validation,
454) a Georgia iba so sínusovým rytmom (1 734), [`posun_domeny.csv`](vysledky/externa/posun_domeny.csv):

- hrudné zvody V2 – V6 majú v Georgia o **14 – 25 % menší rozkmit** (V4: 2,08 → 1,56 mV), dolné
  zvody III a aVF o 7 – 14 % väčší,
- rekonštrukčná chyba autoencodera je na normálnych záznamoch v mediáne o **29 % vyššia**
  (0,0195 → 0,0251), najviac v II, aVF a V6 (≈ 1,5×),
- preto Model A na „normálnych“ záznamoch Georgia hlási anomáliu v **23 %** prípadov (PTB-XL 9 %).
  AUROC sa nezmenilo (0,797) – poradie záznamov ostalo rovnako dobré, ale **celá škála sa posunula**,
  takže prah z PTB-XL dáva viac poplachov (špecificita 0,858 → 0,765) a vyššiu senzitivitu.
- Vek to nevysvetľuje: medián veku NORM záznamov Georgia je 56 rokov, trénovací prísny NORM 53 (#26).

**2. Prahy sa neprenášajú.** AUROC (nezávislé od prahu) kleslo menej ako F1. Najviac pri AFIB:
AUROC 0,927 je stále vysoké, ale prah 0,98 z PTB-XL zachytí iba 68 % fibrilácií. Pravdepodobnosti
modelu sú na nových dátach inak rozložené (kalibrácia). Prahy som podľa pravidiel #22 neladil.

**3. Iné kódovanie diagnóz.** [`pozitivita_podla_kodu.csv`](vysledky/externa/pozitivita_podla_kodu.csv)
ukazuje, ako často Model B hlási label pri každom kóde Georgia:
- **MI hlási pri 16 % záznamov iba so sínusovým rytmom** a pri 35 – 68 % záznamov s inou
  poruchou (najviac laterálna ischémia 68 %, odchýlka osi 60 %, RBBB 59 %). Na PTB-XL teste bola
  špecificita MI 0,867, na Georgia 0,643. Časť týchto „falošných“ MI môžu byť infarkty, ktoré
  Georgia kóduje ako ischémiu alebo Q vlny (Georgia nemá kód „starý infarkt“) – overiť sa to nedá.
- **CD pri odchýlke osi doľava 82 %:** v PTB-XL je ľavý predný hemiblok (LAFB, CD) častou príčinou
  odchýlky osi; Georgia ich kóduje oddelene (LAD 937, LAFB iba 180). Precision CD 0,784 → 0,593.
- **AFIB pri flutteri 37 %** (186 záznamov) – rovnaká zámena ako na PTB-XL, ale flutter je v Georgia
  častejší (1,8 % vs. 0,3 % záznamov).
- **STTC:** Georgia zaratáva do ST-T aj dlhé QT (1 391 záznamov); model ho označí v 62 %.

**4. HYP – hypotéza:** recall HYP klesol najviac (0,737 → 0,527). Kritériá LVH sú napäťové a mapy
v #27 ukázali, že model sa pri LVH pozerá hlavne na V5/V6 a V1 – práve hrudné zvody majú v Georgia
menší rozkmit. Súvis je pravdepodobný, ale nie dokázaný.

## Záver

- Systém **zovšeobecňuje čiastočne**: poradie záznamov (AUROC) sa pri CD (0,896) a AFIB (0,927)
  zachovalo dobre, pri STTC (0,830) a HYP (0,812) horšie. Autoencoder má rovnaké AUROC ako na PTB-XL.
- Najväčší problém sú **prahy a kalibrácia** na novom prístroji a **posun amplitúd** – v praxi by
  bolo treba prahy prekalibrovať na validačných dátach novej nemocnice (to je iná úloha, nie
  vyhodnotenie zmrazeného systému).
- Časť poklesu je spôsobená **rozdielnym kódovaním diagnóz** (MI, os, QT) – nie je to čisto chyba
  modelu. Bez spoločného prečítania záznamov kardiológom sa nedá oddeliť.
- **MI sa na Georgia overiť nedalo** (7 záznamov).

## Kód a overenie

- `src/ekg/evaluation/external.py` – mapovanie, prevzorkovanie, vyhodnotenie, rozbor podľa kódov,
  posun domény, CLI `ekg-external`. Nový súbor v zmrazenom kóde – zapísaný v
  [`zmeny_po_teste.json`](vysledky/test/zmeny_po_teste.json); z PTB-XL číta iba validation.
- `tests/test_external.py` (16 testov):
  - hlavička, mapovanie (pozitívne, nejasné, flutter, nenamapované kódy, referencia Modelu A),
  - prevzorkovanie (zachová frekvenciu, odfiltruje nad Nyquistom, krátky záznam),
  - vyhodnotenie (vynechá nejasné a nepoužiteľné, AUROC = sklearn),
  - rozbor podľa kódov,
  - test množina PTB-XL sa nepoužíva,
  - uložený report (počty sedia, vynechané = krátke + neplatné, licencia a DOI).
- Predikcie všetkých 10 286 záznamov (schéma 1.2): `outputs/externa/predikcie_georgia.csv` (mimo gitu).
