# Dáta

Obsah tohto priečinka sa necommituje (okrem tohto súboru). PTB-XL má približne 3 GB.

Plánovaná štruktúra:

```
data/
├── raw/ptb-xl/      # stiahnutý PTB-XL v1.0.3 (issue #3)
└── processed/       # manifesty splitov, normalizačné parametre (issues #6, #8)
```

Ak sú dáta na inom mieste (napr. iný disk na trénovacom PC), nastav premennú prostredia
`EKG_DATA_DIR`.
