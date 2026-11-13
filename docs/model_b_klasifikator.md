# Model B – multi-label klasifikátor (1D ResNet)

Kód: [`src/ekg/models/classifier.py`](../src/ekg/models/classifier.py) · Prehľad: `uv run ekg-clf-info`

## Úloha

Pre každý záznam odhadnúť **nezávisle** pravdepodobnosť 5 porúch: **MI, STTC, CD, HYP, AFIB**
(poradie podľa kontraktu #2, labely podľa mapovania #4).

### Multi-label, nie multi-class

Jeden záznam môže mať viac porúch naraz (napr. MI + CD), preto:

| | Multi-class (softmax) | **Multi-label (sigmoid) – použité** |
|---|---|---|
| výstup | jedna trieda | ľubovoľný počet labelov |
| pravdepodobnosti | súčet = 1 | každá zvlášť v [0, 1] |
| loss | krížová entropia | **binárna krížová entropia pre každý label** (`BCEWithLogitsLoss`) |
| rozhodnutie | argmax | **vlastný prah pre každý label** ([#17](prahy_labelov.md)) |

Model vracia **logity**. Pravdepodobnosť je P = σ(logit) = 1 / (1 + e^(−logit)) a dáva ju
`predict_proba()`. `BCEWithLogitsLoss` počíta sigmoid vo vnútri numericky stabilne.

## Prečo 1D ResNet

- V benchmarku PTB-XL (Strodthoff et al. 2020) patrili konvolučné siete typu ResNet a InceptionTime
  medzi najlepšie modely (macro AUROC ~0,93 pri superclass úlohe).
- ResNet je jednoduchší na vysvetlenie a implementáciu. InceptionTime sa s ním porovná v #24.
- **Reziduálne spojenia** (skratka `y = F(x) + x`) umožnia gradientu prejsť cez veľa vrstiev,
  takže sa dá trénovať hlbšia sieť bez toho, aby sa učenie zastavilo.

## Architektúra (predvolená konfigurácia)

| Časť | Vrstvy | Výstup |
|---|---|---|
| vstup | signál po preprocessingu v1.0 | (B, 12, 1000) |
| stem | Conv1d k=15 stride 2 + BN + ReLU + MaxPool stride 2 | (B, 64, 250) |
| stage 1 | 2× reziduálny blok (Conv k=7 + BN + ReLU + Conv k=7 + BN, + skratka) | (B, 64, 250) |
| stage 2 | 2× reziduálny blok, prvý so stride 2 | (B, 128, 125) |
| stage 3 | 2× reziduálny blok, prvý so stride 2 | (B, 256, 63) |
| hlava | [priemer ‖ maximum cez čas] → Dropout 0,3 → Linear(512 → 5) | (B, 5) logity |

- **2 181 253 parametrov**, teda ~7× viac ako autoencoder, lebo úloha je zložitejšia.
- Široký stem (k=15 = 150 ms) zachytí tvar celého QRS komplexu už v prvej vrstve.
- **Priemer aj maximum cez čas:** priemer zachytí trvalé zmeny v celom zázname (napr. HYP, blokáda
  vedenia), maximum krátke udalosti, ktoré sa môžu objaviť kdekoľvek v 10 s.
- `features()` vracia mapu čŕt (B, 256, 63) pre Grad-CAM (#27). Každý bod zodpovedá ~160 ms signálu.

## Nevyváženosť labelov

`loss_fn(pos_weight)` prijme váhy pozitívnych príkladov pre každý label. Zriedkavé labely
(AFIB 7 %, HYP 12 %) tak môžu mať pri tréningu väčšiu váhu. Váhy sa počítajú z train množiny v #16
(`pos_weight = neg / pos`, AFIB 13,15) – porovnanie s tréningom bez váh je v
[trening_klasifikatora.md](trening_klasifikatora.md).

## Checkpoint

Rovnaký mechanizmus ako pri autoencoderi (`ekg.models.checkpoint`): váhy, konfigurácia, kontrakt
vrátane poradia labelov a verzie. Checkpoint autoencodera a klasifikátora sa nedajú zameniť.

## Overenie

| Test | ✓ |
|---|---|
| výstup (B, 5), pravdepodobnosti v [0, 1], súčet nie je vynútene 1 | ✓ |
| sieť sa naučí záznamy s 0, 2 aj 5 pozitívnymi labelmi naraz | ✓ |
| gradient do všetkých parametrov, bez NaN | ✓ |
| BCE zodpovedá ručnému vzorcu, `pos_weight` zvyšuje trest za prehliadnutý pozitívny label | ✓ |
| po uložení a načítaní bitovo rovnaký výstup, checkpoint inej triedy modelu sa odmietne | ✓ |

Skúška na skutočnej dávke 32 záznamov (CPU): forward + backward ~0,3–0,6 s. Pri učení na tejto
jednej dávke kleslo BCE za 60 krokov z 1,10 na 0,0003 a všetky labely dávky sedia. Model sa teda
dokáže učiť. Či dobre zovšeobecňuje, ukáže až tréning (#16) a vyhodnotenie (#18).
Odhad tréningu na CPU: ~3–5 min na epochu (536 dávok po 32), na GPU výrazne rýchlejšie (#28).
