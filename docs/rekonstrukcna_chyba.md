# Rekonštrukčná chyba (verzia 1.0)

Kód: [`src/ekg/models/recon_error.py`](../src/ekg/models/recon_error.py)

## Definícia

Pre záznam `x` (12 × 1000, po preprocessingu) a jeho rekonštrukciu `x̂` z autoencodera:

| Úroveň | Vzorec | Tvar | Použitie |
|---|---|---|---|
| chyba zvodu | e_l = (1/1000) · Σ_t (x[l,t] − x̂[l,t])² | (B, 12) | analýza chýb (#14), `error_per_lead` vo výstupe |
| **chyba záznamu (skóre anomálie)** | **e = (1/12) · Σ_l e_l** | **(B,)** | porovnanie s prahom (#13) |
| chyba v čase | (1/12) · Σ_l (x[l,t] − x̂[l,t])² | (B, 1000) | vizualizácia, kde je anomália (#20, #27) |

Skóre záznamu je obyčajné **MSE cez všetkých 12 000 hodnôt**. Jednotka je (z-score)²,
pretože signál je normalizovaný (#8).

## Prečo MSE

- Je to tá istá veličina, ktorú autoencoder pri tréningu minimalizuje (#11), takže skóre a tréning sú konzistentné.
- Druhá mocnina zvýrazní veľké lokálne odchýlky (deformovaný QRS, posunutý ST segment)
  viac ako absolútna chyba (MAE), ktorá by ich „rozmazala“ medzi drobný šum.
- Nevýhoda: je citlivejšia na artefakty (napr. jeden prudký výkyv). Preto sa uchováva aj chyba po zvodoch a v čase.

## Agregácia cez zvody: priemer vs. maximum

Predvolené je **priemer** cez 12 zvodov. Anomália viditeľná iba v niekoľkých zvodoch
(napr. inferiórny MI v II, III, aVF) sa však priemerom „rozriedi“: chyba 4,0 v jednom zvode dá skóre
4/12 ≈ 0,33. Alternatíva **maximum** (najhorší zvod) dá 4,0. Keďže sa chyby po zvodoch ukladajú,
obe agregácie sa dajú porovnať pri kalibrácii prahu na validation (#13) bez nového výpočtu.

## Výpočet pre celú množinu

```python
from ekg.data.dataset import make_loader
from ekg.models.autoencoder import ConvAutoencoder
from ekg.models.recon_error import compute_errors

model, _ = ConvAutoencoder.load("checkpoints/autoencoder.pt")
result = compute_errors(model, make_loader("val", "all"))
table = result.to_frame()  # index ecg_id; error, error_I … error_V6, MI … AFIB
```

`compute_errors` prepne model do režimu `eval()` (BatchNorm použije naučené štatistiky, nie štatistiky
dávky), takže chyba záznamu nezávisí od toho, s ktorými záznamami je v dávke. Po skončení obnoví pôvodný režim.
Na validation (2 183 záznamov) trvá výpočet ~3,4 s na CPU.

## Overenie (testy `tests/test_recon_error.py`)

| Prípad | Očakávaná hodnota | ✓ |
|---|---|---|
| identická rekonštrukcia | 0 (záznam, zvody aj čas) | ✓ |
| konštantný posun x̂ = x + 0,5 | 0,25 v každom zvode aj zázname | ✓ |
| posun 2,0 iba vo V1 | zvod V1 = 4; priemer = 4/12; maximum = 4 | ✓ |
| odchýlka 1,0 iba počas 0,5 s | 0,05; chyba v čase je 1 iba v tých 50 bodoch | ✓ |
| x = sínus, x̂ = 0 | 0,5 | ✓ |
| zhoda s `mse_loss` | rovnaké číslo | ✓ |
| skóre nezávisí od zloženia dávky | rovnaké pre dávku aj jednotlivo | ✓ |
| NaN alebo nekonečno, zlý tvar | chyba `ValueError` | ✓ |
| `compute_errors` = priamy výpočet, rovnaké pre rôzne veľkosti dávky | ✓ | ✓ |
