# Denník rozhodnutí

Záznamy o tom, čo sa rozhodlo, prečo a s akým výsledkom – podklad pre maturitnú prácu.
Zapisujú sa aj neúspešné pokusy.

---

## 2026-11-13 – Vývojové prostredie (issue #1)

**Rozhodnutie:** Python 3.11, správa prostredia cez `uv`, závislosti v `pyproject.toml` + `uv.lock`.

**Dôvody:**
- Na Macu bol iba systémový Python 3.9, ktorý už nemá podporu, a Homebrew na Intel Macu nefunguje.
  `uv` nainštaluje vlastný Python bez Homebrew a bez administrátorských práv.
- `uv.lock` zafixuje presné verzie všetkých knižníc → rovnaké prostredie na Macu, PC aj v Colabe
  (reprodukovateľnosť).
- PyTorch pre Intel Macy skončil verziou **2.2.2** (podporuje Python najviac 3.12) a tá vyžaduje
  NumPy 1.x. Python 3.11 je preto kompromis, ktorý funguje na Macu aj s novými verziami PyTorch
  s ROCm na PC.

**Dôsledky:**
- Na Macu sa netrénuje (iba CPU); Mac slúži na písanie kódu a testy, trénuje sa na PC alebo v Colabe (#28).
- Kód musí fungovať s PyTorch 2.2 aj s novšími verziami.
- Dataset, checkpointy a výstupy sú mimo gitu (`.gitignore`); cesty sa dajú zmeniť premennými prostredia.

**Overenie:** `uv run ekg-check` – na Macu (Intel, CPU) prešli všetky kontroly vrátane jedného
tréningového kroku malej 1D CNN na vstupe (4, 12, 1000); `uv run pytest` – 3 testy prešli.

---

## 2026-11-13 – Kontrakt vstupov a výstupov (issue #2)

**Rozhodnutia:**
- Vstup modelu má tvar `(B, 12, 1000)` (zvody ako kanály pre `Conv1d`); WFDB dáva `(1000, 12)`,
  takže sa transponuje vo funkcii `from_wfdb()`, ktorá zároveň zoradí zvody podľa názvov.
- Poradie zvodov podľa PTB-XL (`I … V6`), poradie labelov `MI, STTC, CD, HYP, AFIB` je definované
  na jednom mieste (`src/ekg/contract.py`), aby sa nikde v kóde nerozišlo.
- Neplatný vstup (iný tvar, iná frekvencia, NaN) sa **odmietne s chybou**, nevyrába sa z neho výsledok.
  Zlá kvalita signálu sa naopak neodmieta, iba sa označí (rozhodne #7) – aj zašumený záznam
  má zmysel vyhodnotiť, ak to vo výstupe vidno.
- Pri rozpore modelov sa zachovajú výsledky oboch a pridá sa príznak `models_agree = false`
  (spätná väzba k prezentácii, bod 4).
- Výstup má číslo verzie schémy, aby sa staré exporty nenačítali potichu zle.
- Chýbajúci vek/pohlavie = `null`, nikdy odhad.

**Overenie:** testy `tests/test_contract.py` (platné aj neplatné vstupy, poradie labelov,
pravdepodobnosti mimo [0, 1], príklad JSON z dokumentácie sa načíta a uloží bez zmeny).

---

## 2026-11-13 – Stiahnutie a načítanie PTB-XL (issue #3)

**Rozhodnutia:**
- Sťahuje sa oficiálny ZIP z PhysioNet (1,84 GB, jeden súbor, pokračovanie po prerušení), nie
  43 598 samostatných súborov – je to rýchlejšie a spoľahlivejšie.
- Rozbalí sa iba 100 Hz časť a metadáta (~0,6 GB); 500 Hz sa dá doplniť prepínačom
  `--with-500hz`, ak budú slabé ST labely (STTC, MI).
- Každý súbor sa overí proti oficiálnemu `SHA256SUMS.txt` – tak je isté, že pracujem presne
  s publikovanou verziou 1.0.3 (reprodukovateľnosť).
- Signály sa načítavajú cez `from_wfdb()` z kontraktu (#2), takže poradie zvodov a tvar sú
  zaručené na jednom mieste. Chybný záznam sa nikdy nevynechá potichu – overenie vypíše každý problém.

**Výsledky:**
- Sťahovanie: 1,84 GB za ~1,5 min, SHA256 ZIPu `2d98687a…aec29812`.
- Všetkých 43 602 súborov (43 598 záznamových + 4 metadáta) má správny SHA256.
- `ekg-verify-data`: všetkých **21 799** záznamov od **18 869** pacientov sa načítalo v tvare
  (12, 1000) pri 100 Hz, bez NaN; 71 SCP kódov, žiadny neznámy kód. Trvanie ~3,7 min na Intel Macu.
- Poradie zvodov v súboroch PTB-XL (`I, II, III, AVR, AVL, AVF, V1–V6`) presne zodpovedá kontraktu.

---

## 2026-11-13 – Mapovanie labelov a výber NORM (issue #4)

**Zistenia z dát:**
- MI, STTC, CD, HYP sú supertriedy v `diagnostic_class` (14, 13, 11 a 5 SCP kódov).
- **AFIB nie je supertrieda, ale rytmický kód**, a likelihood má takmer vždy 0 (1 466 z 1 514).
  Prah likelihood by AFIB prakticky odstránil – dôležité poučenie, čo by som bez pozretia do dát prehliadol.
- 37 záznamov má NORM aj AFIB súčasne (NORM = tvar, AFIB = rytmus).
- Foldy 9 a 10 (validation, test) sú celé overené človekom (`validated_by_human`).

**Rozhodnutia (mapovanie v1.0):**
- MI/STTC/CD/HYP: všetky diagnostické kódy bez prahu likelihood – ako benchmark Strodthoff et al. 2020,
  aby sa dali porovnať výsledky. Alternatívy (≥ 50, = 100) sú zdokumentované s počtami.
- AFIB: kód AFIB bez ohľadu na likelihood; flutter (AFLT) sa nezaratúva.
- 348 záznamov bez diagnostického kódu aj bez AFIB sa z Modelu B vynechá.
- Prísny NORM pre autoencoder: NORM 100, žiadny iný diagnostický kód, rytmus iba SR, bez šumu,
  driftu, extra beats, kardiostimulátora a form kódov → **3 905 train / 454 val / 464 test**
  (z pôvodných 9 514 záznamov s NORM). `validated_by_human` sa nevyžaduje.

**Overenie:** 21 testov – modelové príklady (čistý NORM, NORM s patológiou, AFIB s likelihood 0,
viac labelov, iba flutter, NORM so šumom/PVC) aj presné počty na skutočnom PTB-XL;
CSV mapovacia tabuľka je testom zviazaná s kódom.

---

## 2026-11-13 – Rozdelenie na train / validation / test (issue #6)

**Rozhodnutie:** oficiálne `strat_fold` PTB-XL – foldy 1–8 train, 9 validation, 10 test.
Vlastné náhodné rozdelenie som zamietol: foldy sú už rozdelené podľa pacientov, stratifikované
a foldy 9–10 sú overené človekom; navyše sú výsledky porovnateľné s benchmarkom.

**Opatrenia proti úniku informácií:**
- `check_no_leak()` pri každom vytvorení aj načítaní manifestu overí, že žiadny pacient ani záznam
  nie je vo viacerých množinách.
- Manifest `manifests/splits_v1.0.csv` je v gite → všetky experimenty používajú tie isté záznamy.
- Test množina je zamknutá (`TestSetLockedError`); odomyká sa iba explicitne v záverečnom
  vyhodnotení (#22).

**Výsledok:** 17 418 / 2 183 / 2 198 záznamov (15 023 / 1 942 / 1 904 pacientov), prekrývanie
pacientov 0. Podiely labelov sa medzi množinami líšia najviac o 0,5 p. b. (napr. AFIB 7,1 / 7,0 / 7,0 %).
Prísny NORM: 3 905 / 454 / 464.

---

## 2026-11-13 – Preprocessing a normalizácia (issue #8)

**Rozhodnutia (preprocessing v1.0):**
- Hornopriepustný Butterworth filter 0,5 Hz, 2. rád, nulová fáza – odstráni drift, nedeformuje ST segment.
- Z-score po zvodoch s mean/std z **celej train množiny** (nie z každého záznamu): normalizácia
  každého záznamu zvlášť by zmazala absolútnu amplitúdu, ktorá je dôležitá napr. pre HYP.
- Bez dolnopriepustného filtra – pri 100 Hz nie je potrebný.
- Parametre v `manifests/preprocessing_v1.0.json` (v gite) aj s informáciou, z čoho vznikli
  (split, počet záznamov, SHA256 manifestu).
- Cache surových signálov (1 GB, `data/processed/`) – načítanie všetkých záznamov z ~4 min na sekundy.

**Výsledky:**
- std po zvodoch 0,13–0,32 mV; hrudné zvody V2–V4 majú ~2× väčšiu amplitúdu ako končatinové.
- Validation po normalizácii: priemer 0,000, std 0,978 (nie presne 1 – správne, parametre sú z train).
- Príklad ecg_id 2043: drift 1,12 mV → 0,01 mV, tvar QRS/ST/T zachovaný (docs/obrazky/preprocessing_priklad.png).
- Opakovaný fit dá bitovo rovnaké parametre (deterministické).

**Na neskôr:** filter 0,5 Hz je štandardná voľba; ak budú slabé STTC/MI, dá sa porovnať aj bez filtra
(`highpass_hz=None`) alebo s 0,05 Hz.

---

## 2026-11-13 – Dátový loader (issue #9)

**Rozhodnutia:**
- `EKGDataset` vracia `signal (12, 1000)`, `target (5)` a `ecg_id`; podmnožiny `classifier` (Model B),
  `strict_norm` (Model A) a `all`.
- Preload spracovaných signálov do RAM (train ≈ 0,8 GB, Mac má 16 GB) – rýchly tréning.
- Seed cez vlastný `torch.Generator`; validation a test sa nikdy nemiešajú.

**Overenie (`ekg-check-data`, ~26 s):** všetky dávky v train/val/test majú správny tvar a konečné
hodnoty, ciele sedia s manifestom, signál z loadera je zhodný s priamym načítaním z WFDB, každý záznam
je v epoche práve raz, prísny NORM má ciele 0, prekrývanie pacientov 0, seed je reprodukovateľný.

**Zachytená chyba:** pandas 3 vracia z `to_numpy()` pole iba na čítanie a PyTorch na to upozorňuje
(zápis do takého tenzora má nedefinované správanie) – opravené kópiou; testy teraz bežia aj s
`-W error::UserWarning`.

---

## 2026-11-13 – Model A: 1D konvolučný autoencoder (issue #10)

**Architektúra:** encoder 3 bloky (Conv1d stride 2 + BN + ReLU + Conv1d), hrdlo Conv1d 1×1 na
16 kanálov × 125 bodov (kompresia 6×), decoder zrkadlovo s ConvTranspose1d, lineárny výstup.
321 436 parametrov. Šírka hrdla a ďalšie parametre sú v `AutoencoderConfig` a budú sa ladiť v #11.

**Checkpoint:** váhy + konfigurácia + kontrakt + verzie preprocessingu, mapovania a splitov;
načítanie s `weights_only=True` (bezpečné) a s odmietnutím iného kontraktu.

**Zachytená chyba:** `torch.__version__` je objekt `TorchVersion`, ktorý bezpečné načítanie
odmietne – ukladá sa ako reťazec.

**Overenie:** 11 testov (tvar, gradienty do všetkých parametrov, bitovo rovnaký výstup po
načítaní checkpointu, odmietnutie iného typu/kontraktu, schopnosť preučiť malú dávku).
Na skutočnej dávke prísneho NORM kleslo MSE za 150 krokov z 0,85 na 0,04 – model sa učí.

---

## 2026-11-13 – Rekonštrukčná chyba (issue #12)

**Rozhodnutie:** skóre anomálie = MSE cez všetkých 12 × 1000 hodnôt (priemer chýb po zvodoch).
Rovnaká veličina ako tréningová loss; druhá mocnina zvýrazní veľké lokálne odchýlky (QRS, ST).
Uchováva sa aj chyba po zvodoch (B, 12) a v čase (B, 1000).

**Otvorená otázka do #13:** priemer cez zvody rozriedi anomáliu viditeľnú len v pár zvodoch
(napr. inferiórny MI) – 1 zvod s chybou 4 dá skóre 0,33. Alternatíva „max zvod“ sa porovná na validation
bez nového výpočtu, lebo chyby po zvodoch sú uložené.

**Overenie:** 15 testov s analyticky známymi výsledkami (identita = 0, posun c → c², posun v jednom
zvode → c²/12 resp. c², sínus → 0,5, zhoda s `mse_loss`, nezávislosť od dávky, odmietnutie NaN/zlého tvaru).
Výpočet na celej validation (2 183 záznamov) trvá 3,4 s na CPU.

---

## 2026-11-13 – Model B: multi-label klasifikátor 1D ResNet (issue #15)

**Rozhodnutia:**
- 1D ResNet ako základ (v benchmarku Strodthoff 2020 medzi najlepšími; jednoduchší na vysvetlenie
  ako InceptionTime, ktorý sa porovná v #24).
- Stem Conv k=15 + MaxPool, 3 stage-e po 2 reziduálnych blokoch (64/128/256 kanálov), hlava
  [priemer ‖ maximum cez čas] → Dropout 0,3 → Linear(512 → 5). 2,18 mil. parametrov.
- 5 nezávislých logitov + sigmoid, `BCEWithLogitsLoss` (voliteľne `pos_weight` pre nevyváženosť) –
  nie softmax, lebo záznam môže mať viac porúch naraz.

**Overenie:** 13 testov (tvar, sigmoid nezávislý od ostatných labelov, naučenie záznamov
s 0/2/5 labelmi naraz, gradienty, BCE = ručný vzorec, pos_weight, bitovo rovnaký checkpoint,
nezameniteľnosť s autoencoderom). Skutočná dávka: BCE 1,10 → 0,0003 za 60 krokov.

**Poznámka:** netrénovaná sieť mala BCE 1,10, nie 0,693 – ln 2 platí iba pre logity presne 0;
náhodné váhy dávajú rozptýlenejšie logity. Pri tréningu (#16) sa dá zvážiť inicializácia biasu
podľa podielu pozitívnych príkladov.
