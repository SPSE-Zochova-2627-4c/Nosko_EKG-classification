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
