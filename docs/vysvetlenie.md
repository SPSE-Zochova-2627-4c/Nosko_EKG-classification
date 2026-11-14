# Na čo sa modely pozerajú – mapy dôležitosti (issue #27)

Mapy ukazujú, **ktoré časti vstupného EKG ovplyvnili výstup modelu**. Nie sú to lekárske vysvetlenia,
nehovoria, či je výsledok správny, a ako ukazuje kontrola nižšie, treba ich čítať opatrne.
Modely ani prahy sa nemenili – mapy sa počítajú zo zmrazených modelov (#22).

```bash
uv run ekg-explain examples/00017_lr          # Model A + 5 labelov → outputs/vysvetlenie/
uv run ekg-explain --ecg-id 1440 --label MI   # iba vybraná mapa (test množina je zamknutá)
uv run ekg-explain --check                    # kontrola máp na validation (~18 min)
uv run streamlit run streamlit_app.py         # sekcia „Na čo sa modely pozerajú“
```

## Metódy

| | čo ukazuje | rozlíšenie |
|---|---|---|
| **Integrated Gradients** (Model B) | príspevok každej vzorky každého zvodu k logitu labelu; integrál gradientu po priamke od „rovnej čiary“ (nulový signál po normalizácii = priemer zvodu z train) k záznamu | 12 zvodov × 1000 vzoriek |
| **Grad-CAM** (Model B) | posledná konvolučná vrstva: váha kanála = priemerný gradient logitu, mapa = ReLU(Σ váha · aktivácia) | iba čas, 63 bodov (~160 ms) – zvody sa miešajú už v prvej vrstve |
| **Rekonštrukčná chyba v čase** (Model A) | (x − x̂)² v každej vzorke každého zvodu (normalizovaný priestor); priemer cez všetko je presne anomaly score | 12 × 1000 |

- Každý label má vlastnú mapu (multi-label: gradient logitu daného labelu).
- **Zobrazenie:** farebné pozadie za každým zvodom (iba kladná atribúcia = to, čo zvyšuje
  pravdepodobnosť), vyhladené kĺzavým priemerom 0,1 s, normované 99. percentilom cez všetky zvody,
  aby sa zvody dali porovnať. Pás nad zvodmi: Grad-CAM (výplň) a IG sčítané cez zvody (čiara).
  Pri každom zvode je **podiel** kladnej atribúcie (rovnomerne by bolo 1/12 ≈ 8 %).
- **Úplnosť IG:** súčet atribúcií má byť logit(záznam) − logit(rovná čiara). Na ukážkovom zázname 17
  bola odchýlka pre STTC 11 % pri 64 krokoch a 26 % pri 128, pri **256 krokoch do 4 %**; na 30 záznamoch
  validation je medián najväčšej odchýlky 0,4 %. Jeden záznam (6 máp) trvá ~3 s na CPU.
- Mapy sa počítajú z rovnakých modelov a preprocessingu ako výstup: test overuje, že anomaly score
  a pravdepodobnosti z `explain` sa zhodujú s `Pipeline.predict`.

## Kontrola na validation (`docs/vysledky/vysvetlenie/kontrola.json`)

**1. Lokalizácia infarktu.** Pre správne rozpoznané záznamy s jediným typom MI (iba IMI alebo iba
ASMI, ako v #14) som zmeral podiel kladnej MI atribúcie v dolných zvodoch (II, III, aVF) a predných
(V1 – V4). Rovnomerne by bolo 25 % a 33 %.

| typ MI | záznamov | dolné zvody (medián) | predné zvody (medián) |
|---|---|---|---|
| inferiórny (IMI) | 148 | **24 %** | 34 % |
| anteroseptálny (ASMI) | 107 | 19 % | **41 %** |

Pri ASMI je atribúcia v predných zvodoch nad rovnomerným podielom (41 % vs. 33 %) a mapy sa
medzi typmi posúvajú správnym smerom (dolné 24 → 19 %, predné 34 → 41 %). **Pri IMI však dolné
zvody nedostávajú viac ako rovnomerný podiel** – inferiórny infarkt mapa nelokalizuje.

**2. Porovnanie s náhodným modelom** (Adebayo et al. 2018). Ak by mapa vyzerala rovnako aj pre
sieť s náhodnými váhami, neukazuje, čo sa model naučil. 30 záznamov, Spearmanova korelácia máp
natrénovaného a náhodného modelu (medián cez záznamy a labely):

| metóda | korelácia | záver |
|---|---|---|
| Grad-CAM | −0,03 | závisí od natrénovaných váh ✔ |
| Integrated Gradients (absolútna hodnota, vyhladená) | **0,68** | z veľkej časti ju určuje samotný signál ✘ |

IG sa počíta ako (x − základná línia) × gradient, takže **veľké výchylky (QRS) dostanú veľkú
atribúciu takmer pri akomkoľvek modeli**. Z IG máp sa preto dá čítať hlavne **porovnanie zvodov
a úderov medzi sebou**, nie to, že „model sa pozerá na QRS“ – to by ukázal aj náhodný model.
Grad-CAM test prešiel, ale má iba časové rozlíšenie ~160 ms a nerozlišuje zvody.

## Skontrolované príklady (validation)

Obrázky v [`docs/vysledky/vysvetlenie/`](vysledky/vysvetlenie/). Podiely = kladná IG atribúcia po
zvodoch, „očakávanie“ = typický nález podľa učebnicových kritérií.

| záznam | prípad | P | najväčší podiel | očakávanie | dáva zmysel? |
|---|---|---|---|---|---|
| [1440](vysledky/vysvetlenie/zaznam_1440_MI.png) | ASMI (iba) | MI 0,998 | V1 20 %, V2 17 %, V4 17 %, V3 11 % (spolu 65 %) | QS komplexy vo V1 – V3 | **áno** |
| [2792](vysledky/vysvetlenie/zaznam_2792_MI.png) | IMI (iba) | MI 0,997 | V4 14 %, aVF 13 %, V5 12 %, III 11 %, II 9 % | Q vlny v II, III, aVF | **čiastočne** – dolné zvody 33 %, ale rovnako veľa vo V4 – V5 |
| [11614](vysledky/vysvetlenie/zaznam_11614_CD.png) | RBBB | CD 1,000 | V1 28 %, aVR 9 %, V6 9 % | rSR' vo V1, široké S v I a V6 | **áno** |
| [14047](vysledky/vysvetlenie/zaznam_14047_CD.png) | LBBB | CD 1,000 | V1 24 %, V2 14 %, aVL 11 %, V3 11 % | široký QRS, hlboké S vo V1, široké R v I, aVL | **áno** (V5 je v tomto zázname takmer plochý) |
| [1451](vysledky/vysvetlenie/zaznam_1451_HYP.png) | LVH | HYP 0,998 | V6 24 %, V5 14 %, V1 12 %, aVR 11 % | vysoké R vo V5/V6 + hlboké S vo V1 (Sokolow-Lyon) | **áno** |
| [18252](vysledky/vysvetlenie/zaznam_18252_STTC.png) | laterálna ischémia | STTC 0,981 | V6 16 %, V5 12 %, aVR 12 %, V2 11 % | zmeny ST/T vo V5, V6, I, aVL | **čiastočne** – zvody sedia, ale v čase je atribúcia na QRS, nie na úseku ST/T |
| [17](vysledky/vysvetlenie/zaznam_17_AFIB.png) | AFIB | AFIB 0,998 | V5 14 %, V1 12 %, V2 12 % | nepravidelné RR, chýbajúce P vlny | **nedá sa posúdiť** – atribúcia je na QRS niekoľkých úderov; Grad-CAM a IG označujú čiastočne iné údery |
| [12469](vysledky/vysvetlenie/zaznam_12469_MI.png) | **falošný MI** na prísnom NORM | MI 0,948 | V5 16 %, V1 12 %, aVF 10 %, III 10 % | – (zdravé EKG) | **nie** – atribúcia je sústredená na posledné ~2 s, kde nič typické pre MI nie je |
| [17](vysledky/vysvetlenie/zaznam_17_model_a.png) | Model A, AFIB | score 0,249 | chyba hlavne v II, III, aVF a vo V4 pri artefakte (~1,3 s a ~6,8 s) | fibrilačné vlny sú najlepšie vidieť v dolných zvodoch | **áno** |
| [352](vysledky/vysvetlenie/zaznam_352_model_a.png) | Model A, falošný poplach na NORM so šumom | score 0,078 (prah 0,038) | takmer celá chyba v prvých ~1,2 s, najmä V1 – V3 | artefakt (skok signálu) na začiatku záznamu | **áno** – mapa ukáže, že poplach spôsobil artefakt, nie srdce |

## Čo z toho vyplýva

- **Model A** (chyba v čase) je najľahšie čitateľný: ukazuje priamo, kde autoencoder nevedel
  signál zopakovať. Pri falošných poplachoch zo šumu (#26) to hneď odhalí artefakt.
- **Model B:** pri poruchách so zreteľným tvarom QRS (blokády ramienok, ASMI, LVH) sa podiely
  zvodov zhodujú s učebnicovými kritériami. Pri IMI, ischémii (ST/T) a AFIB mapy nepomáhajú
  a pri falošnom poplachu ukazujú miesto bez klinického významu.
- **IG odrážajú z veľkej časti samotný signál** (korelácia s náhodným modelom 0,68). Preto v práci
  aj v aplikácii hovorím iba „čo ovplyvnilo výstup“, nie „prečo model rozhodol“.
- Atribúcia sa často sústredí na niekoľko úderov namiesto všetkých rovnako – je to vlastnosť
  metódy (gradient konkrétneho vstupu), nie dôkaz, že ostatné údery sú zdravé.

## Kód a overenie

- `src/ekg/explain.py` – IG, Grad-CAM, chyba v čase, súhrny, obrázok, kontrola, CLI `ekg-explain`.
  Nový súbor v zmrazenom kóde, preto zapísaný v
  [`zmeny_po_teste.json`](vysledky/test/zmeny_po_teste.json); výstupy systému nemení.
- `streamlit_app.py` – sekcia „Na čo sa modely pozerajú“: výber Model A / label (predvolene
  najpravdepodobnejší pozitívny label, inak Model A), podiely zvodov, stiahnutie PNG.
- `tests/test_explain.py` (14 testov):
  - IG je pre lineárny model presne x · w,
  - úplnosť IG na ResNete,
  - Grad-CAM tvar, rozsah a iná mapa pre iný label,
  - vyhladenie, podiely zvodov, normovanie,
  - zhoda s výstupom pipeline,
  - obnovenie počtu vlákien, odmietnutie neplatného vstupu, obrázok, predvolený cieľ,
  - CLI odmietne test množinu,
  - CLI na ukážkovom súbore.
