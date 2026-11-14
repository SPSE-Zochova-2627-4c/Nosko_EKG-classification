# Ukážkové záznamy (PTB-XL)

Štyri pôvodné záznamy z PTB-XL v1.0.3 (100 Hz, formát WFDB: `.hea` + `.dat`), skopírované bez zmeny,
aby sa demo dalo spustiť **bez stiahnutia celého datasetu**. Všetky sú z **validation** množiny
(fold 9), nie z test množiny.

| súbor | ecg_id | skutočný label | čo ukazuje |
|---|---|---|---|
| `00010_lr` | 10 | NORM | oba modely v norme |
| `00017_lr` | 17 | AFIB | anomália, Model B označí AFIB (a falošne aj MI) |
| `00127_lr` | 127 | STTC | Model A v norme, Model B STTC – nesúlad modelov |
| `00352_lr` | 352 | NORM | falošný poplach autoencodera – nesúlad modelov |

Očakávané výstupy: [docs/demo.md](../docs/demo.md).

Zdroj: Wagner, P. et al. (2022). PTB-XL, a large publicly available electrocardiography dataset
(version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45 · Licencia
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
