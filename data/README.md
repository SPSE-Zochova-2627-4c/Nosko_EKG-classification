# Dáta

Obsah tohto priečinka sa necommituje (okrem tohto súboru).

## PTB-XL v1.0.3

```bash
uv run ekg-download          # iba 100 Hz + metadáta (predvolené)
uv run ekg-download --with-500hz
uv run ekg-download --keep-zip
uv run ekg-verify-data       # načíta všetkých 21 799 záznamov a overí ich
```

| | |
|---|---|
| Zdroj | https://physionet.org/content/ptb-xl/1.0.3/ |
| Súbor | ZIP z PhysioNet, 1,84 GB (100 Hz aj 500 Hz) |
| Kontrolné súčty | oficiálny `SHA256SUMS.txt` z PhysioNet – overí sa **každý** rozbalený súbor |
| Licencia | CC BY 4.0 |
| Záznam o stiahnutí | `data/raw/ptb-xl/DOWNLOAD_INFO.json` (verzia, zdroj, SHA256 ZIPu, dátum) |

Štruktúra po stiahnutí:

```
data/
├── raw/ptb-xl/
│   ├── ptbxl_database.csv     # 1 riadok = 1 záznam (ecg_id, patient_id, vek, pohlavie, scp_codes, strat_fold, …)
│   ├── scp_statements.csv     # popis 71 SCP kódov (diagnostic_class, rhythm, …)
│   ├── records100/            # 100 Hz signály vo formáte WFDB (.hea + .dat)
│   ├── SHA256SUMS.txt
│   ├── LICENSE.txt
│   └── DOWNLOAD_INFO.json
└── processed/                 # manifesty splitov, normalizačné parametre (issues #6, #8)
```

Prerušené sťahovanie stačí spustiť znova, pokračuje od miesta, kde skončilo. Ak sú dáta inde
(napr. iný disk na trénovacom PC), nastav premennú prostredia `EKG_DATA_DIR`.
