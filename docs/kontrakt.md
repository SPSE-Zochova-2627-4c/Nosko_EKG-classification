# Kontrakt vstupov a výstupov (verzia schémy 1.0)

Tento dokument určuje, čo systém prijíma a čo vracia. V kóde ho implementuje
[`src/ekg/contract.py`](../src/ekg/contract.py). Ostatné časti projektu (loader, modely,
inference aj export) berú poradie zvodov a labelov **iba odtiaľ**.

> Výstup je výsledok modelu strojového učenia na výskumné a vzdelávacie účely, **nie lekárska diagnóza**.

## 1. Vstup

| Vlastnosť | Hodnota |
|---|---|
| Počet zvodov | 12 |
| Poradie zvodov | `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6` (ako `sig_name` v PTB-XL) |
| Vzorkovacia frekvencia | 100 Hz |
| Dĺžka | 10 s = 1000 vzoriek |
| Jednotka | mV |
| Tvar jedného záznamu pre model | `(12, 1000)`, tj. (zvody, čas) |
| Tvar dávky pre model | `(B, 12, 1000)`, `float32` |

### Prečo `(12, 1000)` a nie `(1000, 12)`

WFDB (formát PTB-XL) vracia signál ako `(1000, 12)`, tj. riadok je časový okamih. `Conv1d`
v PyTorch očakáva `(batch, kanály, čas)`, takže zvody sú kanály a záznam sa
transponuje. Na prevod slúži `from_wfdb()`. Ak dostane názvy zvodov (`record.sig_name`),
zvody aj preusporiada do správneho poradia.

```python
import wfdb
from ekg.contract import from_wfdb

record = wfdb.rdrecord("data/raw/ptb-xl/records100/00000/00001_lr")
x = from_wfdb(record.p_signal, record.sig_name, record.fs)  # -> np.ndarray (12, 1000), float32
```

### Platný vstup (príklad)

NumPy pole tvaru `(12, 1000)`, reálne čísla v mV, všetky konečné, 100 Hz,
napr. záznam `00001_lr` z PTB-XL po `from_wfdb()`.

### Neplatný vstup → `InvalidInputError`

Inference sa pri neplatnom vstupe **nespustí** a chyba vysvetlí dôvod:

| Situácia | Správanie |
|---|---|
| iná frekvencia než 100 Hz | chyba (signál treba vopred prevzorkovať, napr. pri externých datasetoch v #31) |
| tvar `(1000, 12)` | chyba s nápovedou „použi `from_wfdb()`“ |
| iný tvar (iná dĺžka, iný počet zvodov) | chyba s uvedeným tvarom |
| nečíselné hodnoty | chyba |
| `NaN` alebo nekonečno | chyba s menami postihnutých zvodov |
| chýbajúci zvod podľa `sig_name` | chyba so zoznamom chýbajúcich zvodov |

Kvalita signálu (plochý zvod, šum, podozrivá amplitúda) **nie je** dôvod na odmietnutie.
Rieši ju kontrola kvality (#7) a vo výstupe sa objaví ako varovanie v poli `quality`.

## 2. Výstup

Poradie labelov je pevné: **`MI, STTC, CD, HYP, AFIB`**. Index `i` vo výstupe Modelu B `(B, 5)`
zodpovedá `LABELS[i]`. Každý výstup je **nezávislý sigmoid**, nie softmax, takže súčet
pravdepodobností nemusí byť 1 a pozitívnych môže byť viac labelov naraz.

| Pole | Typ | Popis | Kto ho plní |
|---|---|---|---|
| `schema_version` | str | verzia tejto schémy, teraz `"1.0"` | – |
| `record_id` | str | identifikátor záznamu (napr. `ecg_id` z PTB-XL alebo názov súboru) | inference |
| `model_versions` | {str: str} | verzie použitých checkpointov | #19 |
| `anomaly.is_anomaly` | bool | `reconstruction_error >= threshold` | Model A, #13 |
| `anomaly.reconstruction_error` | float | jedno skóre na záznam | #12 |
| `anomaly.threshold` | float | prah kalibrovaný na validation | #13 |
| `anomaly.error_per_lead` | {zvod: float} | chyba po zvodoch v poradí `LEADS` | #12 |
| `labels.<LABEL>.probability` | float ∈ [0, 1] | sigmoid skóre | Model B |
| `labels.<LABEL>.threshold` | float | prah daného labelu z validation | #17 |
| `labels.<LABEL>.positive` | bool | `probability >= threshold` | #17 |
| `consistency.models_agree` | bool | súlad Modelu A a B | #19 |
| `consistency.note` | str \| null | vysvetlenie pri nesúlade | #19 |
| `quality.status` | `"ok"` \| `"warning"` \| `"unknown"` | výsledok kontroly kvality (`unknown`, kým #7 nie je hotové) | #7 |
| `quality.issues` | [str] | zoznam nájdených problémov | #7 |
| `metadata.age` | int \| null | vek, ak je známy | #21 |
| `metadata.sex` | `"male"` \| `"female"` \| null | pohlavie (PTB-XL: 0 = muž, 1 = žena) | #21 |
| `metadata.source` | str \| null | zdroj záznamu | #21 |
| `disclaimer` | str | upozornenie, že nejde o diagnózu | – |

Chýbajúce metadáta sú vždy `null`, nikdy sa nedopĺňajú odhadom.

### Rozpor modelov

Modely môžu nesúhlasiť: autoencoder nehlási anomáliu, ale klasifikátor dá napríklad
P(MI) = 0.85, alebo naopak. Výstup v takom prípade **zachová výsledky oboch modelov**
a nastaví `consistency.models_agree = false` s vysvetlením v `note`. Žiadny model druhý neprebije.
Presné pravidlo (napr. či sa počíta aj hraničný prípad) doplní #19.

### Príklad výstupu

Súbor [`priklad_vystupu.json`](priklad_vystupu.json). Čísla sú **vymyslené** na ilustráciu,
nepochádzajú z modelu. Ukazuje práve prípad rozporu modelov. Test `tests/test_contract.py`
overuje, že tento príklad zodpovedá kódu.

```json
{
  "record_id": "PRIKLAD-0001",
  "model_versions": {"autoencoder": "ae-0.0.0", "classifier": "resnet1d-0.0.0"},
  "anomaly": {
    "is_anomaly": false,
    "reconstruction_error": 0.0142,
    "threshold": 0.0215,
    "error_per_lead": {"I": 0.012, "II": 0.015, "...": "...", "V6": 0.012}
  },
  "labels": {
    "MI":   {"probability": 0.85, "threshold": 0.41, "positive": true},
    "STTC": {"probability": 0.32, "threshold": 0.38, "positive": false},
    "CD":   {"probability": 0.07, "threshold": 0.35, "positive": false},
    "HYP":  {"probability": 0.11, "threshold": 0.27, "positive": false},
    "AFIB": {"probability": 0.02, "threshold": 0.30, "positive": false}
  },
  "consistency": {
    "models_agree": false,
    "note": "Model A nehlási anomáliu, ale Model B označil MI – záznam odporúčame skontrolovať."
  },
  "quality": {"status": "ok", "issues": []},
  "metadata": {"age": 63, "sex": "male", "source": "PTB-XL"},
  "schema_version": "1.0",
  "disclaimer": "Výsledok modelu strojového učenia na výskumné a vzdelávacie účely, nie lekárska diagnóza."
}
```

## 3. Verzovanie schémy

- Pridanie nového **voliteľného** poľa → `1.x` (napr. `1.1`).
- Zmena významu, premenovanie alebo odstránenie poľa, zmena poradia labelov či zvodov → `2.0`.
- `PredictionResult.from_dict()` odmietne neznámu verziu, takže starý export sa nenačíta potichu zle.
