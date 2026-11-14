# Spoločná inference oboch modelov (issue #19)

Kód: [`src/ekg/inference.py`](../src/ekg/inference.py) · Výstup: [kontrakt](kontrakt.md) (schéma 1.2) ·
Príklady skutočných výstupov: [`vysledky/inference/`](vysledky/inference/)

```bash
uv run ekg-predict --ecg-id 10                       # záznam PTB-XL (nie z test množiny)
uv run ekg-predict cesta/k/zaznamu                   # vlastný WFDB záznam (.hea + .dat, bez prípony)
uv run ekg-predict --ecg-id 10 --ecg-id 127 --json vysledky.json --csv vysledky.csv
```

```python
from ekg.inference import Pipeline, read_wfdb

pipe = Pipeline.load()  # ae_v1 + clf_v1, CPU
result = pipe.predict(read_wfdb("cesta/k/zaznamu"), record_id="moj-zaznam")
result.anomaly.anomaly_score, result.labels["MI"].probability
print(result.to_json())
```

Toto je časť systému, ktorú požaduje maturitné zadanie: **jeden 12-zvodový záznam → anomaly score
z autoencodera a pravdepodobnosť každej z 5 porúch z klasifikátora**. Zobrazenie EKG a výsledkov
pribudne v #20.

## Postup pre jeden záznam

| krok | čo sa deje | odkiaľ |
|---|---|---|
| 1. overenie vstupu | tvar (12, 1000), 100 Hz, konečné čísla; WFDB záznam sa transponuje a zvody zoradia podľa `sig_name` | kontrakt #2 |
| 2. kontrola kvality | plochý zvod, extrémna amplitúda, saturácia (+ anotácie šumu pri PTB-XL) – **iba varovanie**, záznam sa neodmietne | #7 |
| 3. preprocessing | filter + normalizácia s parametrami **z train množiny** (`manifests/preprocessing_v1.0.json`) | #8 |
| 4. Model A | rekonštrukčná chyba po zvodoch → **anomaly score** = priemer cez zvody; anomália, ak score ≥ prah 0,0384 | #11–#13 |
| 5. Model B | 5 pravdepodobností (sigmoid), label pozitívny, ak P ≥ prah labelu | #15–#17 |
| 6. súlad modelov | pravidlo nižšie | #19 |

**Model B beží vždy** – aj keď Model A anomáliu nehlási (žiadny gating). Výstup má preto vždy všetkých
5 pravdepodobností; o tom, ako výsledky spojiť, rozhoduje iba pravidlo súladu, nie poradie modelov.

## Čo pipeline pri načítaní overí

Pipeline sa **nespustí** (chyba `PipelineError` s vysvetlením), ak:

- chýba checkpoint alebo súbor s prahmi – chyba odkáže na GitHub Release `model-<run>`,
- prah anomálie alebo prahy labelov **nepatria k checkpointu** (iný SHA256 – napr. po pretrénovaní
  modelu bez novej kalibrácie),
- prahy labelov majú iné poradie labelov alebo inú verziu formátu (#17),
- model bol natrénovaný s inou verziou preprocessingu, alebo modely boli natrénované s rôznymi verziami
  preprocessingu, mapovania labelov či rozdelenia dát,
- checkpoint má iný kontrakt (zvody, labely, dĺžku signálu – kontroluje už `checkpoint.load`).

Neplatný vstup vyhodí `InvalidInputError` s dôvodom (napr. „Záznam je vo formáte WFDB (1000, 12) – použi
`from_wfdb()`“) a nič sa nevypočíta. Záznamy z PTB-XL **test množiny** CLI odmietne (`TestSetLockedError`)
– aj keď sú zadané cestou k súboru; test množina patrí iba #22.

## Verzie modelov vo výstupe

```json
"model_versions": {
  "autoencoder": "ae_v1 (epocha 95, sha256 b9214cde491f)",
  "classifier": "clf_v1 (epocha 20, sha256 20d915b776a3)"
}
```

Názov behu, epocha a začiatok SHA256 checkpointu – z výstupu sa dá presne určiť, ktorý model ho vytvoril
(rovnaký SHA256 je v `prah.json`, `prahy.json` a v GitHub Release).

## Pravidlo súladu modelov

| Model A | Model B | `models_agree` | `note` |
|---|---|---|---|
| anomália | aspoň 1 label | `true` | – |
| v norme | žiadny label | `true` | – |
| v norme | aspoň 1 label | **`false`** | „Model A nehlási anomáliu, ale Model B označil STTC (P = 0.91) – záznam odporúčame skontrolovať.“ |
| anomália | žiadny label | **`false`** | „Model A hlási anomáliu, ale Model B neoznačil žiadnu z porúch MI, STTC, CD, HYP, AFIB – môže ísť o inú poruchu, šum alebo falošný poplach.“ |

- Pri nesúlade sa **výsledky oboch modelov zachovajú** – žiadny model druhý neprebije, používateľ vidí oba.
- Rozhoduje iba porovnanie s prahmi. Hraničné pásmo okolo prahu som nepoužil: pridalo by ďalší parameter,
  ktorý by bolo treba nastaviť a zdôvodniť, a pravidlo by bolo ťažšie overiť.
- Kvalita signálu súlad neovplyvňuje, zobrazuje sa zvlášť.

### Ako často modely nesúhlasia (validation, 2 150 záznamov podmnožiny `classifier`)

| prípad | záznamov | z toho so skutočnou poruchou | z toho NORM |
|---|---|---|---|
| súlad | 1 525 (71 %) | 817 | 708 |
| A v norme, B porucha | **490** (23 %) | 381 | 109 |
| A anomália, B nič | 135 (6 %) | 42 | 93 |

- Nesúlad je pri **29 %** záznamov – modely sú rôzne citlivé. Autoencoder má na validation senzitivitu
  iba 0,61 (#13), klasifikátor zachytí väčšinu porúch.
- Prípad „A v norme, B porucha“ je teda najčastejšie **správny klasifikátor** (381 z 490 záznamov porucha
  naozaj má) – autoencoder ich rekonštruuje dobre, lebo sú blízko normálnemu EKG.
- Prípad „A anomália, B nič“ je najčastejšie **NORM záznam** (93 zo 135). Z nich 62 nespĺňa prísny NORM
  (#4), na ktorom bol autoencoder trénovaný – majú napr. iné nálezy mimo 5 labelov, extrasystoly alebo
  šum; varovanie kvality má 32 z 93. Zvyšných 31 sú prísne NORM záznamy, teda falošné poplachy autoencodera.
- Príznak nesúladu je preto užitočný: označí záznamy, pri ktorých sa oplatí pozrieť sa na výsledky oboch
  modelov.

## Príklady (skutočné výstupy, validation)

[`priklad_inference.txt`](vysledky/inference/priklad_inference.txt),
[`.json`](vysledky/inference/priklad_inference.json), [`.csv`](vysledky/inference/priklad_inference.csv):

| ecg_id | skutočnosť | anomaly score (prah 0,0384) | Model B | súlad |
|---|---|---|---|---|
| 10 | NORM | 0,0086 – v norme | nič | áno |
| 17 | AFIB | 0,2492 – anomália | AFIB (P 0,998), **MI (P 0,752, prah 0,746 – falošne)** | áno |
| 127 | STTC | 0,0223 – v norme | STTC (P 0,908) | **nie** – A poruchu prehliadol |
| 352 | NORM | 0,0777 – anomália | nič | **nie** – falošný poplach autoencodera (záznam so šumom) |

## Reprodukovateľnosť a rýchlosť

- Predvolene **CPU, float32, každý záznam samostatne v 1 vlákne**. Opakovaný beh dá bitovo rovnaký
  výsledok a výsledok záznamu **nezávisí od toho, s akými inými záznamami bol spracovaný** (test).
- Pri spracovaní v dávke sa poradie sčítania vo float32 mení a pravdepodobnosti sa líšili v ~1e-7 – preto
  sa záznamy počítajú po jednom.
- Zistenie počas práce: v PyTorch 2.14 (ROCm build) trvá jeden záznam (dávka 1) pri 16 vláknach CPU
  **~0,3 s na model**, v 1 vlákne autoencoder ~2 ms a klasifikátor ~14 ms. Preto 1 vlákno – je to
  rýchlejšie aj reprodukovateľné. Celá validation (2 150 záznamov) vrátane kontroly kvality: **13 s,
  ~6 ms na záznam**.
- Zhoda s predchádzajúcimi výpočtami: rekonštrukčné chyby na validation sa od kalibrácie prahu (#13)
  líšia najviac o 4·10⁻⁷ a matica zámen je rovnaká (TP 759, FP 41); pravdepodobnosti klasifikátora sa od
  #17/#18 (GPU) líšia najviac o 7·10⁻⁷ a **žiadne rozhodnutie sa nezmenilo**.

## Overenie (testy)

`tests/test_inference.py` – 15 testov na malých netrénovaných modeloch (nepotrebujú dataset ani GPU):
pravidlo súladu vo všetkých 4 prípadoch, úplný výstup podľa JSON schémy 1.2 (anomaly score = chyba,
prahy, rozhodnutia, verzie modelov), výsledok sa rovná priamemu výpočtu modelov, opakovaný beh a dávka
vs. samostatný záznam bitovo rovnaké, obnovenie počtu vlákien, Model B beží aj bez anomálie, varovanie
kvality záznam neblokuje, neplatný vstup, prahy iného checkpointu, chýbajúci checkpoint s odkazom na
Release, rozpoznanie cesty do PTB-XL, zámok test množiny, skutočné modely používajú uložené prahy.
