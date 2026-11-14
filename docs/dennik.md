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

---

## 2026-11-13 – Tréning autoencodera, beh ae_v1 (issue #11)

**Nastavenie:** 3 905 záznamov prísneho NORM (train), Adam lr 1e-3, dávka 64, ReduceLROnPlateau,
max. 100 epoch, early stopping 12, seed 42. Tréning na Intel Macu (CPU) – ukázalo sa, že GPU
pre autoencoder netreba: ~15–20 s na epochu, spolu 26,5 min.

**Odolnosť voči prerušeniu:** po každej epoche sa ukladá celý stav; `--resume` pokračuje.
Test dokazuje bitovo rovnaký výsledok ako bez prerušenia (poradie miešania závisí iba od seedu a epochy).

**Výsledky:**
- najlepšia epocha 95, val MSE 0,0255 (z 0,178 po 1. epoche); train ≈ val → bez preučenia,
- validation: medián chyby prísny NORM 0,020; MI 0,065; STTC 0,051; CD 0,083; HYP 0,085; AFIB 0,099,
- **AUROC prísny NORM vs. patológia 0,832** (priemer cez zvody) a 0,793 (max zvod).

**Zistenia:** autoencoder bez videnia patológie rozlišuje normálne a patologické EKG; najlepšie
poruchy tvaru QRS a rytmu (CD, HYP, AFIB), horšie ST/T zmeny (MI, STTC). Rozdelenia sa prekrývajú –
to je očakávaný limit a dôvod pre Model B. Priemer cez zvody > maximum (otázka z #12).

**Ďalší nápad (P2):** užšie hrdlo (napr. 8 kanálov) by mohlo zväčšiť rozdiel medzi NORM a patológiou,
lebo autoencoder by mal menej kapacity kopírovať aj nezvyčajné tvary.

---

## 2026-11-13 – Kalibrácia prahu anomálie (issue #13)

**Postup:** na validation (454 prísny NORM vs. 1 240 s ≥ 1 labelom, 489 nejednoznačných vynechaných)
som porovnal 7 pravidiel: priemer NORM, priemer + 2/3·std, 95./99. percentil NORM, Youden J, max F1.

**Zistenia:**
- prah zo slajdu 11 (priemer chyby NORM) označí **28 % normálnych EKG** ako anomáliu → potvrdená chyba,
- priemer + k·std zlyháva na zošikmenom rozdelení (senzitivita 17–23 %),
- max F1 je skreslené podielom patológie vo validation (73 %) → špecificita 34 %,
- agregácia priemerom cez zvody je aj pri prahu lepšia ako maximum (p95: 55 % vs. 43 % senzitivita).

**Rozhodnutie (spolu):** **Youden J, prah 0,03837** – senzitivita 61 %, špecificita 91 %, precision 95 %.
Alternatíva p95 (55 % / 95 %) bola bližšie k princípu „učím sa iba z NORM“ a nezávisí od zloženia
patológie; Youden dáva lepšie vyváženie oboch chýb. Toto obmedzenie treba uviesť v práci.

Prah je uložený spolu so SHA256 checkpointu (test kontroluje, že patrí k `best.pt`).

---

## 2026-11-13 – Analýza chýb autoencodera (issue #14)

**Metriky (validation, prah Youden 0,03837):** TP 759, FP 41, TN 413, FN 481 → senzitivita 0,61,
špecificita 0,91, precision 0,95, F1 0,74, AUROC 0,83, AUPRC 0,93.

**Zistenia:**
- Zachytené podľa labelu: AFIB 80 %, HYP 77 %, CD 74 %, MI 66 %, STTC 60 %; pri jedinom labeli STTC
  len 37 % a MI 49 % → autoencoder dobre vidí zmeny tvaru QRS a rytmu, zle ST/T zmeny a starý infarkt.
- HYP má najväčšiu chybu vo V4–V6 (napäťové kritériá LVH) – autoencoder „oreže“ vysoké amplitúdy.
- Lokalizácia MI podľa chyby po zvodoch súhlasí s anatómiou iba čiastočne → na lokalizáciu bude Grad-CAM (#27).
- Najväčší „falošný poplach“ (ecg_id 10535) je prísny NORM s **nezaznamenaným artefaktom elektród** vo V5/V6 →
  časť FP sú chyby anotácií, nie modelu (limitácia PTB-XL, podnet pre #7).
- FP majú častejšie najhorší hrudný zvod (V1, V4, V3) a sú častejšie u mužov (68 % vs. 53 %).
- Najhorší FN (MI, chyba 6× pod prahom) vyzerá úplne normálne – principiálny limit autoencodera.

**Zachytená chyba:** test odhalil pád pri zázname bez SCP kódov (NaN po reindexe) – ošetrené.

---

## 2026-11-13 – Kontrola kvality záznamov (issue #7)

**Rozhodnutia:**
- Dve úrovne: technická platnosť (kontrakt – neplatný vstup sa nespracuje) a kvalita signálu
  (záznam sa iba označí `ok`/`warning` s dôvodmi, nikdy sa nevyradí).
- Pravidlá na signáli: plochý zvod (rozkmit < 0,05 mV), extrémna amplitúda (> 10 mV), saturácia
  (≥ 100 ms na maxime > 1 mV); k tomu anotácie šumu z PTB-XL.

**Výsledky (PTB-XL):** 16 781 ok, 5 018 warning (väčšinou anotácie šumu/driftu), 0 vyradených.
Jediný plochý zvod (ecg_id 12722, V5) má v PTB-XL poznámku o elektródach V4, V5; z 25 záznamov
nad 10 mV má 17 anotáciu šumu; spolu 18/26 signálových nálezov potvrdzuje anotácia.

**Neúspešný pokus:** saturácia s prahom 5 vzoriek (50 ms) dala 3 falošné poplachy – dno pomalej vlny
a začiatok záznamu pri rozlíšení 1 µV; sprísnené na 10 vzoriek (100 ms) → 0 nálezov v PTB-XL.

---

## 2026-11-13 – Metadáta a export do JSON/CSV (issue #21)

**Zistenie:** PTB-XL uvádza vek **300** pre 293 pacientov starších ako 89 rokov (anonymizácia).
Taký vek nesmie ísť do výstupu, ale ani „90“ by nebol pravdivý → `age = null` a nové pole
`age_over_89 = true`. Ide o rozšírenie kontraktu → **schéma 1.1** (voliteľné pole, 1.0 sa dá stále načítať).

**Zachytený problém:** checkpoint porovnával pri načítaní celý kontrakt vrátane verzie schémy – zmena
na 1.1 by odmietla už natrénovaný `ae_v1`. Verzia výstupného formátu s váhami nesúvisí, porovnávajú sa
už len zvody, labely, frekvencia a dĺžka.

**Export:** JSON (vnorený, formálna JSON schéma s rozsahmi hodnôt) a CSV (51 stĺpcov, chýbajúce =
prázdne, presný zápis čísel). Oba sa dajú načítať späť na rovnaké objekty – overené testami aj na
hodnote 0,1 + 0,2.

---

## 2026-11-13 – Prieskum datasetu (issue #5)

**Opravy pôvodných predpokladov:** záznamy sú z **11 typov prístrojov** (nie jedného); jeden zdroj
(jedna inštitúcia, jeden výrobca) však platí.

**Neistota, ktorú som takmer prehliadol:** stĺpec `recording_date` má dátumy 1984–2001, popis datasetu
uvádza 10/1989–6/1996. Najprv som to chcel zapísať ako „opravu“, ale 41 % záznamov je mimo obdobia
z popisu – nejde o odľahlé hodnoty. Pravdepodobne posun dátumov pri anonymizácii, neoverené →
v práci uvádzam obdobie z popisu datasetu a dátumy z CSV nepoužívam.

**Zistenia:**
- 25 % záznamov má 2+ poruchy → potvrdzuje multi-label prístup.
- AFIB je takmer vždy s inou poruchou (60 % aj STTC, 40 % aj MI); 57 % HYP má aj STTC.
- NORM + CD 415 záznamov (napr. inkompletná blokáda pravého ramienka ako variant normy).
- Vek: medián NORM 54, patológie 67–71 rokov → vek je zmätočný faktor (#26, #30).
- MI a CD častejšie u mužov (38–39 % žien vs. 54 % pri NORM).
- AFIB má najviac záznamov s varovaním kvality (28,6 %).

---

## 2026-11-13 – Tréningové prostredie s GPU (issue #28)

**Rozhodnutie:** trénuje sa na PC s AMD RX 9070 XT vo **WSL2 (Ubuntu 24.04)**, PyTorch **2.14.0 + ROCm 10.1**
z indexu AMD, iba pre architektúru karty (gfx1201). Na Linuxe ho `uv sync` nainštaluje sám, Intel Mac
ostáva na torch 2.2.2 (CPU). Záloha: Colab notebook s predinštalovaným PyTorch (CUDA).

**Dôvody:**
- WSL2 namiesto dual-bootu – netreba preinštalovať počítač. Od ROCm 7.13 je prístup ku GPU vo WSL
  (ROCDXG cez Windows ovládač) priamo v runtime, takže do Ubuntu netreba inštalovať ROCm ani ovládač.
- Index AMD namiesto indexu PyTorch: torch z `download.pytorch.org/whl/rocm10.0` vyžaduje knižnice pre
  všetky GPU od AMD (`rocm[device-all]`); `torch[device-gfx1201]` od AMD je ~1,6 GB na stiahnutie.

**Neúspešné pokusy a omyly:**
- Prvé `uv sync` na PC sťahovalo torch 2.2.2 z PyPI s 12 balíkmi NVIDIA CUDA (na AMD nepoužiteľné)
  rýchlosťou ~0,5 MB/s – zastavené.
- Skúšobná inštalácia z indexu PyTorch sťahovala knižnice pre všetky GPU (po ~4 GB stále nekončila) – zastavené.
- `uv lock` hlásil chybu pre Python 3.12, tak som Linux obmedzil na 3.11 – chyba ostala. Skutočná príčina:
  `explicit` index sa v uv uplatní iba na **priame** závislosti; balíky ROCm (triton, rocm-sdk-…) sú preto
  uvedené priamo v `pyproject.toml` s markerom pre Linux. Obmedzenie na 3.11 som vrátil.
- `ekg-check` prešiel, ale tréning skutočných modelov na GPU padal (`miopenStatusUnknownError` v BatchNorm):
  MIOpen kompiluje kernely pri prvom použití a čisté Ubuntu nemá hlavičky C → `apt install libc6-dev`.
  Malá sieť v `ekg-check` BatchNorm nemá, preto chybu neodhalila → nové testy trénujú na GPU oba modely.
- Import torchu najprv padol na chýbajúcom `libgomp.so.1` → `apt install libgomp1`.

**Výsledok – rýchlosť tréningového kroku** (`ekg-gpu-bench`, dávka 64, ms/krok):

| model | CPU Ryzen 7 7800X3D | GPU float32 | GPU bfloat16 |
|---|---|---|---|
| autoencoder | 69,4 | 43,5 (1,6×) | 10,5 (6,6×) |
| klasifikátor | 117,2 | 41,9 (2,8×) | 9,1 (12,9×) |

**Zistenie:** vo float32 je GPU len 1,6–2,8× rýchlejšie ako CPU. MIOpen pre viaceré naše 1D konvolúcie
(stride 2, spätné šírenie) volí pomalé záložné algoritmy (~1 500 kernelov na krok, čas lineárne s dávkou).
Nepomohlo `MIOPEN_FIND_MODE=NORMAL` ani `cudnn.benchmark`; bez MIOpen bolo ešte 2× pomalšie.
Zmiešaná presnosť bfloat16 je 4–5× rýchlejšia ako float32 a nepotrebuje `GradScaler`.

**Dôsledky:**
- #16: trénovať s `torch.autocast("cuda", dtype=torch.bfloat16)` a overiť, že metriky sa nelíšia od float32.
- Zapísané verzie a celý postup inštalácie: `docs/gpu_prostredie.md`.

**Overenie:** `uv run ekg-check` → „zariadenie: ROCm – AMD Radeon RX 9070 XT“; `uv run ekg-gpu-bench`;
`uv run pytest -q -W error::UserWarning` na PC (testy GPU sa na Macu preskočia).

---

## 2026-11-13 – Tréning klasifikátora, beh clf_v1 (issue #16)

**Nastavenie:** `ResNet1dClassifier` na 17 135 záznamoch (train), validation 2 150, Adam lr 1e-3,
weight decay 1e-4, dávka 64, bfloat16 na GPU, max. 60 epoch, early stopping 10 epoch, seed 42.
Jeden beh na RX 9070 XT trvá ~2,5 min (~4 s na epochu) – na Macu by to bolo hodiny.

**Rozhodnutie 1 – výber modelu podľa val macro AUROC, nie val loss:** beh s `pos_weight` a beh bez neho
optimalizujú inú loss, ich straty sa nedajú porovnať. AUROC nezávisí od prahu (prahy až v #17).

**Rozhodnutie 2 – `pos_weight = neg / pos` z train** (MI 2,91, STTC 3,09, CD 3,39, HYP 7,09, AFIB 13,15).
Porovnanie s tréningom bez váh, každá varianta 3 seedy (42, 1, 2):
- macro AUROC rovnaké: 0,939 ± 0,001 (s váhami) vs. 0,939 ± 0,002 (bez váh), macro AUPRC 0,819 oboje,
- AFIB s váhami lepší vo všetkých seedoch: AUROC 0,993 vs. 0,989, AUPRC 0,939 ± 0,006 vs. 0,925 ± 0,019,
- MI, STTC, CD, HYP – rozdiely v rozsahu rozptylu medzi seedmi,
- cena: horšia kalibrácia (nevážené val BCE 0,296 vs. 0,224) – rieši #17 prahmi na validation.
→ `clf_v1` = s `pos_weight`, seed 42 (podľa konfigurácie, nie najlepší seed – to by nadhodnotilo výsledok).

**Výsledok clf_v1 (validation, najlepšia epocha 20 z 30):** macro AUROC **0,938**, macro AUPRC 0,819;
AUROC MI 0,925, STTC 0,932, CD 0,933, HYP 0,907, AFIB 0,992. Porovnateľné s benchmarkom
(Strodthoff 2020, ~0,93). HYP je najslabší (AUPRC 0,67), ako sa čakalo.

**bf16 vs. float32 (overenie odporúčania z #28):** macro AUROC 0,938 vs. 0,939, AUPRC rovnaké – rozdiel
menší ako medzi seedmi; float32 trval 7,6 min, bf16 2,5 min. Ostávame pri bf16.

**Omyly a zistenia počas práce:**
- Pri sledovaní tréningu som predpokladal, že zlepšenie v epoche 20 spôsobilo zníženie lr – nebolo to tak,
  lr klesol až v epoche 25 (overené v `metrics.csv`). Po znížení lr train loss prudko klesá, validation nie
  (začiatok preučenia) – early stopping zastavil včas.
- Čakal som, že váhy pomôžu aj HYP (12 %) – nepomohli.
- Súbory vytvorené z Windows cez `\wsl.localhost` patrili používateľovi root (inštancia WSL bola prvýkrát
  spustená ako root kvôli apt) a ruff ich nemohol preformátovať → `chown` a reštart WSL, odvtedy v poriadku.
- Na GPU nie je tréning bitovo reprodukovateľný (MIOpen) – preto porovnanie cez viac seedov. Bitovú
  zhodu po prerušení a `--resume` overuje test na CPU.

**Overenie:** 12 nových testov (`tests/test_train_classifier.py`), `uv run ekg-clf-compare` pre 7 behov,
výsledky v `docs/vysledky/clf_v1/`, dokumentácia `docs/trening_klasifikatora.md`.

---

## 2026-11-14 – Prahy pre labely na validation (issue #17)

**Kritérium určené vopred:** maximum F1 pre každý label na validation (pri zhode vyšší prah, prah
v strede medzi susednými pravdepodobnosťami). Pri menej ako 30 pozitívnych alebo 30 negatívnych
príkladoch náhradný prah 0,5. Požadovanú citlivosť som nezvolil – chýba klinické zdôvodnenie konkrétnej
hodnoty.

**Výsledok (`clf_v1`, validation 2 150 záznamov, 1 919 pacientov):** prahy MI 0,746 · STTC 0,724 ·
CD 0,792 · HYP 0,606 · AFIB 0,981. Náhradný prah nebol potrebný (najmenej pozitívnych AFIB – 151).
Macro F1: prah 0,5 → 0,708, zvolené prahy → 0,758, krížová validácia po pacientoch → 0,748.

**Zistenia:**
- Všetky prahy nad 0,5 – potvrdenie posunu pravdepodobností pri `pos_weight` z #16. AFIB (najväčšia
  váha 13,15) má prah až 0,98: falošné poplachy 66 → 15, prehliadnuté 6 → 16.
- Bootstrap intervaly prahov sú široké (HYP 0,58 – 0,85), ale F1 krivky sú v tejto oblasti ploché,
  takže presná hodnota prahu F1 veľmi nemení.
- Výber prahu na tých istých dátach nadhodnocuje F1 asi o 0,01 (krížová validácia), skutočný prírastok
  oproti 0,5 je ~0,04.
- bf16 vs. float32: rozdiel pravdepodobností max. 0,024, zmenených 28 z 10 750 rozhodnutí → prahy
  počítam vo float32 a inference (#19) má tiež bežať vo float32.

**Omyly počas práce:**
- Prvá verzia hodnotila F1 iba na celej validation; až pri pohľade na výsledky som si uvedomil, že
  prah aj F1 sú z tých istých dát → doplnená krížová validácia po pacientoch.
- Prvý test „pri zhode vyšší prah“ nemal v skutočnosti zhodu F1 (prepočítal som ho ručne) – opravený príklad.
- Pokus na zdôvodnenie hranice 30 pozitívnych (podvzorkovanie AFIB) mení aj podiel pozitívnych, preto
  ho uvádzam iba ako orientačný: pri 10 pozitívnych je najhorší výsledok F1 0,60, od 30 ~0,84, prah 0,5
  dáva 0,83.

**Overenie:** 13 testov (`tests/test_label_thresholds.py`), prahy v `docs/vysledky/clf_v1/prahy.json`
(SHA256 checkpointu), dokumentácia `docs/prahy_labelov.md`. Test množina nebola použitá.

---

## 2026-11-14 – Vyhodnotenie klasifikátora na validation (issue #18)

**Postup:** `ekg-clf-report` vyhodnotí `clf_v1` s uloženými prahmi z #17 (overí SHA256 checkpointu),
pravdepodobnosti vo float32. Pre každý label support, matica zámen, precision, recall, špecificita, F1,
AUROC, AUPRC; nedefinovaná metrika = `null` s dôvodom (pri `clf_v1` žiadna). Macro, micro a vážené súhrny,
metriky po záznamoch, analýza chýb.

**Výsledok (validation, 2 150 záznamov):** macro F1 0,758, macro AUROC 0,938, macro AUPRC 0,819;
micro F1 0,746. Všetkých 5 rozhodnutí správne má 62 % záznamov. Najlepší AFIB (F1 0,90), najslabší HYP
(F1 0,61, precision 0,51).

**Analýza chýb:**
- Falošné poplachy sú väčšinou na záznamoch s inou poruchou (137 zo 199 FP pre MI), najčastejšie zámena
  MI ↔ STTC. AFIB nemá na NORM záznamoch žiadny falošný poplach.
- **Istota anotácie:** recall MI je 0,57 pri likelihood 15 („infarkt nie je vylúčený“) a 0,90 pri 100;
  45 % prehliadnutých MI má likelihood 15. Mapovanie (#4) berie kódy bez ohľadu na likelihood – časť
  „chýb“ je neistota v samotnej anotácii.
- Podtriedy: blokády ramienok 100 %, LVH 0,83; slabé LMI 0,30, IVCD 0,51, LAO/LAE 0,49 (P vlna pri 100 Hz).
- Vek 40–59 má nižší recall (0,66) ako ≥ 80 (0,87); príčinu som neoveril. Pohlavie bez rozdielu,
  šum v anotácii mierne horší (F1 0,72 vs. 0,75).

**Omyly počas práce:**
- Tabuľky FP/FN mali najprv stĺpce v zlom poradí (stĺpec MI na konci, lebo riadok MI ho nemal) a vekové
  skupiny zoradené abecedne („40–59“ pred „< 40“) – opravené.
- V dokumentácii som najprv napísal, že „asi 40 % prehliadnutých model označí ako normálne“ a že recall
  vo veku 40–59 je nižší „pri všetkých piatich labeloch“ – po kontrole čísel je to 31–43 % a pri AFIB
  (iba 8 záznamov) to neplatí; opravené.

**Overenie:** 13 testov (`tests/test_clf_report.py`), výsledky v `docs/vysledky/clf_v1/hodnotenie/`,
dokumentácia `docs/vyhodnotenie_klasifikatora.md`. Test množina nebola použitá.

---

## 2026-11-14 – Spoločná inference oboch modelov (issue #19)

**Súlad so zadaním:** maturitné zadanie požaduje, aby výstup „zobrazil anomaly score a pravdepodobnosť
jednotlivých porúch“. Výraz „anomaly score“ sa v projekte doteraz nevyskytoval (pole sa volalo
`reconstruction_error`). **Rozhodnutie (Matúš):** nové pole `anomaly.anomaly_score` = priamo rekonštrukčná
chyba, nie normalizovaná chyba / prah – presne podľa zadania („podľa rekonštrukčnej chyby“), bez kroku
navyše. Kontrakt → schéma 1.2, staršie výstupy (1.0, 1.1) sa pri načítaní doplnia.

**Pipeline (`ekg-predict`, `Pipeline`):** overenie vstupu → kontrola kvality → preprocessing z train →
autoencoder (prah z #13) → klasifikátor (prahy z #17, beží vždy) → pravidlo súladu. Pri načítaní overí
SHA256 prahov voči checkpointom a verzie preprocessingu, mapovania a splitov.

**Pravidlo súladu:** zhoda = (A anomália a B aspoň 1 label) alebo (A v norme a B nič); bez pásma okolo prahu.
Na validation nesúlad pri 29 % záznamov – najčastejšie „A v norme, B porucha“ (490), z toho 381 so skutočnou
poruchou → autoencoder je menej citlivý (senzitivita 0,61). „A anomália, B nič“ (135) je najmä NORM (93), z toho 62 mimo prísneho NORM.
Pôvodne som napísal „NORM so šumom“ – po kontrole má varovanie kvality iba 32 z 93, opravené.

**Overenie:** chyby autoencodera sa od #13 líšia max. o 4·10⁻⁷ (rovnaká matica zámen), pravdepodobnosti
klasifikátora od #17 max. o 7·10⁻⁷, žiadne zmenené rozhodnutie. 15 testov.

**Problémy a omyly počas práce:**
- **Pomalý jeden záznam:** v PyTorch 2.14 (ROCm build) trval záznam v dávke 1 pri 16 vláknach CPU ~0,3 s
  na model (dávka 2 už ~4 ms). Prvý pokus o porovnanie dávka vs. samostatne bežal > 10 min, kým som ho
  zastavil. S 1 vláknom ~2 ms (AE) a ~14 ms (klasifikátor).
- **Výsledok závisel od dávky:** pravdepodobnosť záznamu v dávke a samostatne sa líšila v ~1e-7
  (iné poradie sčítania). → pipeline počíta každý záznam samostatne v 1 vlákne: výsledok nezávisí od dávky
  ani od počtu jadier a je to aj rýchlejšie (celá validation 13 s).
- **Diera v zámku test množiny:** `--ecg-id` test množinu odmietal, ale cesta k súboru PTB-XL
  (`records100/00000/00009_lr`) by ho obišla → záznamy PTB-XL zadané cestou sa spracujú rovnako ako `--ecg-id`.
- Pri kontrole návratového kódu cez `wsl … bash -c '…; echo $?'` vyšlo 0 namiesto 1 – `$?` rozbalil
  vonkajší shell; zo skriptu je kód správne 1.

---

## 2026-11-14 – Zobrazenie EKG, rekonštrukcie a výsledkov (issue #20)

**Rozhodnutie (Matúš): obrázok aj web aplikácia.** `ekg-show` uloží prehľad do PNG (do práce
a prezentácie), `streamlit_app.py` je interaktívne demo na maturitu – výber záznamu PTB-XL alebo
nahratie vlastného WFDB / `.npy`. Obe používajú tú istú funkciu `plot_report` a hodnoty berú priamo
z výstupu pipeline (#19).

**Čo sa zobrazuje:** 12 pomenovaných zvodov (os 0–10 s, mV), signál po filtrovaní a rekonštrukcia
autoencodera (prevedená späť do mV), chyba po zvodoch, anomaly score s prahom, 5 pravdepodobností
s prahmi, súlad modelov, kvalita a upozornenie, že nejde o diagnózu.

**Problémy a omyly počas práce:**
- **pyarrow vs. NumPy 1.x:** po pridaní Streamlitu aplikácia spadla pri `st.dataframe` – pyarrow 26
  vyžaduje NumPy 2, projekt má NumPy 1.26 (kvôli Intel Macu je v lock súbore jedna verzia pre všetky
  platformy). Overil som, že pyarrow 25 s NumPy 1.26 funguje → `pyarrow<26`. NumPy 2 na Linuxe som
  zamietol, aby sa nezmenilo prostredie natrénovaných modelov. Chybu odhalil až beh skutočnej aplikácie
  (agent API Streamlitu), nie import.
- Prvá verzia obrázka: popisy pravdepodobností boli orezané na pravom okraji, text anomaly score
  prekrýval prah a oranžová rekonštrukcia zakrývala signál → upravené rozloženie, signál čierny,
  rekonštrukcia priesvitná.
- V tmavej téme bol obrázok z matplotlibu biely blok → svetlá téma v `.streamlit/config.toml`.

**Overenie:** 21 testov vrátane testu celej aplikácie (`AppTest`) – hodnoty v aplikácii sa zhodujú
s pipeline; aplikácia spustená a preklikaná (záznamy 10, 127, zamknutý 9, nahrávanie). Dokumentácia
`docs/zobrazenie.md`.

---

## 2026-11-14 – Záverečné vyhodnotenie na test množine (issue #22)

**Postup, aby sa nedalo ladiť na teste:**
1. Napísal som `ekg-final-eval` a testy. **Skúška nanečisto na validation** (test sa nepoužil – v skripte
   mimo repozitára som namiesto test ID podsunul validation ID) dala presne čísla z #14 a #18 (rozdiel
   0,000) → kód počíta to isté ako predchádzajúce vyhodnotenia.
2. **Zmrazenie** (`zmrazenie.json`, 44 súborov: checkpointy, prahy, preprocessing, manifest, mapovanie
   labelov, `uv.lock`, celý `src/ekg`) a **commit `a96600e` pred spustením**.
3. **Jediný beh** na celej test množine (2 198 záznamov, 1 904 pacientov, prekrytie pacientov s train/val 0),
   44 s. Pred behom sa overilo, že sa od zmrazenia nič nezmenilo.

**Výsledky (test, 95 % bootstrap po pacientoch):**
- Model A: AUROC **0,796** (0,771 – 0,820), senzitivita 0,586, špecificita 0,858 (validation 0,832 / 0,612 / 0,910).
- Model B: macro AUROC **0,929** (0,922 – 0,936), macro F1 **0,745** (0,728 – 0,762), micro F1 0,735
  (validation 0,938 / 0,758 / 0,746). AFIB F1 0,870, HYP F1 0,606.
- Súlad modelov: nesúlad pri 30 % záznamov (validation 29 %).

**Zhodnotenie:**
- Klasifikátor zovšeobecňuje – macro F1 0,745 sedí s odhadom krížovej validácie z #17 (0,748), prahy
  z validation fungujú.
- Autoencoder je na teste slabší: AUROC pod intervalom z validation, falošné poplachy na prísnom NORM
  41 → 66. Validation bola pri ňom optimistická (výber epochy aj prahu). **Nič som neupravoval** – zmena
  prahu podľa testu by bola ladenie na test dátach.
- Rovnaké vzory chýb ako na validation: zámeny MI ↔ STTC, neisté anotácie MI (recall 0,58 pri
  likelihood 15 vs. 0,89 pri 100), slabé IVCD, LMI, LAO/LAE. Nové pozorovanie: ženy micro F1 0,719 vs.
  muži 0,751 (na validation bez rozdielu) – bez intervalov iba pozorovanie, do #26.

**Omyly a úpravy počas práce:**
- Test „test množinu otvára iba záverečné vyhodnotenie“ najprv hľadal text `allow_test=True` a našiel ho
  aj v docstringoch a v kontrole dát z #9 (`check_pipeline.py`), ktorá test množinu otvára iba na kontrolu
  tvarov bez modelov. Prerobil som ho na rozbor kódu (AST): hľadá skutočné volania a overí, že kontrola
  dát nenačítava žiadny model.
- Pred zmrazením som chcel istotu, že vyhodnotenie nespadne až na teste (oprava kódu po behu by
  znamenala nové zmrazenie) – preto skúška nanečisto na validation.

---

## 2026-11-14 – Dokumentácia a spustiteľné demo (issue #23)

**Čo pribudlo:** nové README (výsledky z testu, rýchly štart, ako systém funguje s odkazmi na všetky
dokumenty, reprodukcia, obmedzenia, licencie), `docs/demo.md` (scenár s očakávanými výstupmi),
4 ukážkové záznamy v `examples/` a zdroj „Ukážka“ vo web aplikácii, `LICENSE` + `NOTICE`.

**Rozhodnutia:**
- **Licencia kódu: Apache 2.0** (rozhodol Matúš). Text som skopíroval z balíka, ktorý je už v prostredí
  (rovnaký štandardný text vo viacerých balíkoch), nie z internetu.
- **Demo bez datasetu:** 4 pôvodné záznamy PTB-XL z validation (CC BY 4.0 dovoľuje šírenie s uvedením
  zdroja; SHA256 overené voči oficiálnemu `SHA256SUMS.txt`). Nový používateľ potrebuje iba kód
  a modely (~10 MB) z Releasov – repozitár je verejný, stačí `curl`.
- Pri #23 som zámerne nemenil kód v `src/ekg` ani `uv.lock`, aby zmrazenie z #22 ostalo platné.

**Overenie ako nový používateľ:** čerstvý `git clone` do prázdneho priečinka, `uv sync`, `ekg-check`,
stiahnutie modelov cez `curl`, `ekg-final-eval --check`, demo príkazy, testy – bez datasetu.

**Nájdená chyba a jej riešenie:** v čerstvom klone `ekg-predict examples/00127_lr` padol („Nenašiel som
ptbxl_database.csv“) – `ptbxl_records` načítaval databázu aj pri prázdnom zozname `--ecg-id`. Na mojom
PC s datasetom sa to nedalo všimnúť. Oprava je v zmrazenom kóde, preto som pridal **evidenciu zmien
po teste** (`zmeny_po_teste.json`): povolené iba zapísané zmeny kódu v `src/`, nikdy modely, prahy ani
dáta; `--check` ich vypíše. Overené, že výstupy pre ukážkové záznamy sú zhodné s #19.
Rozhodnutie opraviť a zdokumentovať (namiesto obchádzania cez `ekg-show`) urobil Matúš.

**Omyly:** do demo návodu som najprv napísal, že Streamlit pri prvom spustení pýta e-mail – neoveril som
to, tak som to odstránil.
