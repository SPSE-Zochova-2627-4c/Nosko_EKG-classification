# Metadáta a export výsledkov (JSON, CSV)

Kód: [`src/ekg/export.py`](../src/ekg/export.py) · JSON schéma:
[`schema/vystup_v1.2.schema.json`](schema/vystup_v1.2.schema.json) · Príklady:
[`priklad_vystupu.json`](priklad_vystupu.json), [`priklad_exportu.csv`](priklad_exportu.csv)

```python
from ekg import export

export.to_json(results, "vysledky.json")  # results = zoznam PredictionResult (kontrakt #2)
export.to_csv(results, "vysledky.csv")
results = export.load_json("vysledky.json")  # spätné načítanie – rovnaké objekty
results = export.load_csv("vysledky.csv")
```

## Metadáta

| Pole | Zdroj (PTB-XL) | Pravidlo |
|---|---|---|
| `age` | stĺpec `age` | celé číslo 0–89. **Vek 300 = anonymizovaný vek > 89** (293 záznamov) → `age = null`, `age_over_89 = true` |
| `age_over_89` | odvodené | nové v schéme 1.1 |
| `sex` | stĺpec `sex` (0 = muž, 1 = žena) | `"male"` / `"female"` / `null` |
| `source` | – | `"PTB-XL"` alebo napr. názov súboru |
| kvalita | #7 | `quality.status` + `quality.issues` (pravidlá na signáli + anotácie šumu) |

**Chýbajúce hodnoty sa nikdy neodhadujú.** Vek 300 nie je skutočný vek, ale ani „90“ by nebol pravdivý,
preto `null` s príznakom. Výška a hmotnosť sa neexportujú (v PTB-XL chýbajú v 68 % a 57 % záznamov a model ich nepoužíva).

## JSON

Úplný, vnorený formát, presne štruktúra `PredictionResult` z [kontraktu](kontrakt.md):

```json
{
  "schema_version": "1.2",
  "created_at": "2026-11-13T12:00:00+00:00",
  "disclaimer": "Výsledok modelu strojového učenia …, nie lekárska diagnóza.",
  "results": [ { "record_id": "…", "anomaly": {…}, "labels": {…}, … } ]
}
```

Formálna [JSON schéma](schema/vystup_v1.2.schema.json) (draft 2020-12) okrem štruktúry kontroluje aj
rozsahy: pravdepodobnosti v [0, 1], chyby ≥ 0, vek 0–89, povolené hodnoty `quality.status` a `sex`
a prítomnosť všetkých 5 labelov a 12 zvodov. Testy overujú, že export schéme zodpovedá a že
neplatné hodnoty schéma odmietne.

## CSV

Plochá tabuľka pre Excel a analýzu: **1 riadok = 1 záznam**, 52 stĺpcov v tomto poradí:

| Stĺpce | Význam |
|---|---|
| `record_id`, `schema_version` | identifikátor záznamu, verzia schémy |
| `anomaly_score`, `is_anomaly`, `reconstruction_error`, `anomaly_threshold` | výsledok Modelu A (#12, #13); `anomaly_score` = rekonštrukčná chyba (od 1.2) |
| `error_I` … `error_V6` | rekonštrukčná chyba po zvodoch (12 stĺpcov) |
| `prob_MI` … `prob_AFIB` | pravdepodobnosti Modelu B (5 stĺpcov) |
| `threshold_MI` … `threshold_AFIB` | prahy labelov (#17) |
| `positive_MI` … `positive_AFIB` | rozhodnutie `prob ≥ threshold` |
| `models_agree`, `consistency_note` | súlad modelov (#19) |
| `quality_status`, `quality_issues` | kvalita (#7); problémy oddelené ` \| ` |
| `age`, `age_over_89`, `sex`, `source` | metadáta |
| `model_autoencoder`, `model_classifier` | verzie modelov |
| `disclaimer` | upozornenie, že nejde o diagnózu |

Pravidlá zápisu:
- **chýbajúca hodnota = prázdna bunka** (napr. `age` pri veku > 89, `sex` pri neznámom pohlaví),
- logické hodnoty `true` / `false`,
- desatinné čísla v presnom zápise (Python `repr`), takže spätné načítanie dá **bitovo rovnaké** číslo
  (test: `0.1 + 0.2 = 0.30000000000000004` prežije cestu do CSV a späť),
- export odmietne problém kvality obsahujúci oddeľovač ` | ` aj neznámy názov modelu, aby sa CSV nedalo zle načítať.

## Verzia schémy 1.0 → 1.1

Pole `age_over_89` je nové, ale **voliteľné**, takže podľa pravidiel verzovania (#2) ide o verziu 1.1.
Výstupy vo verzii 1.0 sa dajú načítať aj naďalej (chýbajúce pole = `false`) a pri načítaní sa povýšia na 1.1.

Natrénované modely (checkpointy) obsahujú verziu schémy, s ktorou vznikli. Táto verzia sa pri načítaní
modelu **ignoruje**, lebo formát výstupu s váhami modelu nesúvisí. Porovnávajú sa iba zvody, labely,
frekvencia a dĺžka signálu. Bez toho by zmena schémy zneplatnila už natrénovaný autoencoder `ae_v1`.

## Verzia schémy 1.1 → 1.2 (#19)

Nové pole `anomaly.anomaly_score` (v CSV stĺpec `anomaly_score`) – **skóre anomálie** pomenované podľa
maturitného zadania („výstup zobrazí anomaly score“). Jeho hodnota je **rovná rekonštrukčnej chybe**
(`reconstruction_error`), kontrakt to pri vytvorení výsledku overí. Význam ani iné pole sa nemení,
ide iba o pridanie, preto 1.2. Staršie výstupy (1.0, 1.1) v JSON aj CSV sa načítajú ďalej –
chýbajúce `anomaly_score` sa doplní z `reconstruction_error` (nejde o odhad, je to tá istá hodnota)
a verzia sa povýši na 1.2. Schéma 1.1 ostáva v repozitári pre staršie exporty.
