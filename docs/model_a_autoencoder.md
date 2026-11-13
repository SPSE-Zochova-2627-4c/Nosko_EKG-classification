# Model A – 1D konvolučný autoencoder

Kód: [`src/ekg/models/autoencoder.py`](../src/ekg/models/autoencoder.py) · Prehľad: `uv run ekg-ae-info`

## Princíp

Autoencoder sa trénuje **iba na prísnom NORM** (#4). Naučí sa skomprimovať normálne EKG do
malej latentnej reprezentácie a z nej ho zrekonštruovať. Ak dostane záznam s anomáliou, ktorú
nikdy nevidel, zrekonštruuje ho horšie: **vysoká rekonštrukčná chyba znamená anomáliu**
(výpočet chyby #12, prah #13).

## Architektúra (predvolená konfigurácia)

| Časť | Vrstvy | Výstup |
|---|---|---|
| vstup | signál po preprocessingu v1.0 | (B, 12, 1000) |
| encoder blok 1 | Conv1d k=7 stride 2 + BN + ReLU, Conv1d k=7 + BN + ReLU | (B, 32, 500) |
| encoder blok 2 | to isté | (B, 64, 250) |
| encoder blok 3 | to isté | (B, 128, 125) |
| **hrdlo (latent)** | Conv1d 1×1 | **(B, 16, 125)** |
| decoder | Conv1d 1×1 + BN + ReLU | (B, 128, 125) |
| decoder blok 1 | ConvTranspose1d k=4 stride 2 + BN + ReLU, Conv1d k=7 + BN + ReLU | (B, 64, 250) |
| decoder blok 2 | to isté | (B, 32, 500) |
| decoder blok 3 | to isté | (B, 32, 1000) |
| výstup | Conv1d k=7, **bez aktivácie** | (B, 12, 1000) |

- **321 436 parametrov**. Latentná reprezentácia má 16 × 125 = 2 000 čísel, teda **6× menej** ako vstup (12 000).
- Každý bod latentnej reprezentácie opisuje úsek ~80 ms všetkých 12 zvodov naraz, takže model
  využíva aj vzťahy medzi zvodmi.
- Výstup je lineárny, pretože normalizovaný signál má kladné aj záporné hodnoty.

### Prečo takto

- **Konvolúcie** zachytia lokálne tvary (QRS, ST, T vlna) bez ohľadu na to, kde v 10 s sú.
- **Stride 2 namiesto poolingu:** sieť sa sama naučí, ako zmenšovať, a ConvTranspose zrkadlovo zväčšuje.
  1000 = 2³ × 125, takže tri zmenšenia vyjdú presne.
- **Šírka hrdla je kompromis.** Príliš široké hrdlo by sa naučilo vstup iba kopírovať a dobre by
  zrekonštruovalo aj anomálie. Príliš úzke by nezvládlo ani normálne EKG. `latent_channels`
  (a ostatné parametre v `AutoencoderConfig`) sa ladia pri tréningu (#11) podľa validation.

## Checkpoint

`model.save(path, **extra)` / `ConvAutoencoder.load(path)` uloží a načíta:
- váhy vrátane štatistík BatchNorm a konfiguráciu modelu,
- kontrakt (poradie zvodov a labelov, 100 Hz, 1000 vzoriek): pri inom kontrakte sa checkpoint **odmietne**,
- verzie preprocessingu, mapovania labelov, rozdelenia dát a PyTorch,
- ľubovoľné doplňujúce údaje (epocha, metriky, seed).

Načítanie používa `torch.load(weights_only=True)`, teda nespúšťa žiadny kód zo súboru.

## Overenie

| Test | Výsledok |
|---|---|
| výstup má rovnaký tvar ako vstup (aj pre iné konfigurácie) | ✓ |
| gradient sa dostane do všetkých 321 436 parametrov, bez NaN | ✓ |
| po uložení a načítaní dáva model **bitovo rovnaký** výstup (vrátane BatchNorm) | ✓ |
| checkpoint iného typu alebo s iným kontraktom sa odmietne | ✓ |
| model sa dokáže „naučiť naspamäť“ malú dávku (MSE klesne pod 10 %) | ✓ |

Skúška na skutočnej dávke 32 záznamov prísneho NORM (CPU, Intel Mac): forward + backward ~0,46 s.
Pri učení na tejto jednej dávke kleslo MSE za 150 krokov z 0,85 na 0,04 (24 s). Model sa teda reálne
učí. Ide iba o kontrolu, skutočný tréning na celej train množine je #11.
