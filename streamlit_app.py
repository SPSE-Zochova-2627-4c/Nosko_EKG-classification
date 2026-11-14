"""Web aplikácia: EKG, rekonštrukcia a výsledky oboch modelov (issue #20).

Spustenie:  uv run streamlit run streamlit_app.py
Potrebuje checkpointy ae_v1 a clf_v1 (GitHub Release). Ukážkové záznamy v examples/ fungujú
bez datasetu; ľubovoľný záznam PTB-XL potrebuje stiahnutý dataset (uv run ekg-download).
Všetka logika je v balíku ekg (inference, visualization, app_support) – tu sa iba zobrazuje.
"""

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from ekg import app_support, contract, export, inference
from ekg.visualization import plot_report, verdict

st.set_page_config(
    page_title="EKG – anomálie a poruchy", page_icon=":material/monitor_heart:", layout="wide"
)
st.title("Analýza 12-zvodového EKG")
st.caption(
    "Model A (autoencoder) vypočíta anomaly score podľa rekonštrukčnej chyby, "
    "Model B (1D ResNet) pravdepodobnosť porúch MI, STTC, CD, HYP a AFIB. "
    "Prahy oboch modelov sú nastavené na validation dátach PTB-XL."
)
st.warning(contract.DISCLAIMER, title="Nejde o lekársku diagnózu", icon=":material/warning:")


@st.cache_resource
def load_pipeline() -> inference.Pipeline:
    return inference.Pipeline.load()


EXAMPLES_DIR = Path(__file__).parent / "examples"


@st.cache_data(max_entries=8)
def analyze_example(ecg_id: int):
    rid = f"{ecg_id:05d}_lr"
    signal = inference.read_wfdb(EXAMPLES_DIR / rid)
    return analyze(signal, rid, contract.Metadata(source=f"PTB-XL (ukážka examples/{rid})"), None)


@st.cache_data(max_entries=32)
def analyze_ptbxl(ecg_id: int):
    [(rid, signal, meta, ann)] = list(inference.ptbxl_records([ecg_id]))
    return analyze(signal, rid, meta, ann)


@st.cache_data(max_entries=8)
def analyze_upload(files: list[tuple[str, bytes]]):
    rid, signal = app_support.signal_from_upload(files)
    return analyze(signal, rid, contract.Metadata(source=rid), None)


def analyze(signal, rid, meta, ann):
    pipe = load_pipeline()
    filtered, recon = pipe.reconstruct(signal)
    return pipe.predict(signal, rid, meta, ann), filtered, recon


try:
    load_pipeline()
except inference.PipelineError as exc:
    st.error(str(exc), title="Modely sa nepodarilo načítať", icon=":material/error:")
    st.stop()

# --- výber záznamu ----------------------------------------------------------------

source = st.segmented_control(
    "Zdroj záznamu",
    ["Ukážka", "PTB-XL", "Vlastný súbor"],
    default="Ukážka",
    required=True,
    key="source",
    help="Ukážka: 4 záznamy v priečinku examples/ (netreba dataset). "
    "PTB-XL: ľubovoľný záznam zo stiahnutého datasetu.",
)
analysis = None
try:
    if source == "Ukážka":
        example = st.selectbox(
            "Ukážkový záznam (PTB-XL, validation)",
            list(app_support.EXAMPLES),
            format_func=lambda i: f"{i} – {app_support.EXAMPLES[i]}",
            key="example",
        )
        analysis = analyze_example(example)
    elif source == "PTB-XL":
        ecg_id = st.number_input(
            "ecg_id záznamu PTB-XL",
            min_value=1,
            max_value=21837,
            value=127,
            step=1,
            key="ecg_id",
            help="Záznamy test množiny sú zamknuté pre záverečné vyhodnotenie (#22).",
        )
        analysis = analyze_ptbxl(int(ecg_id))
    else:
        uploaded = st.file_uploader(
            "WFDB záznam (.hea + .dat) alebo NumPy pole (.npy, tvar 12 × 1000 v mV)",
            type=["hea", "dat", "npy"],
            accept_multiple_files=True,
            key="upload",
        )
        if not uploaded:
            st.info("Nahraj súbory záznamu: 100 Hz, 12 zvodov, 10 s.", icon=":material/upload:")
            st.stop()
        analysis = analyze_upload([(f.name, f.getvalue()) for f in uploaded])
except contract.InvalidInputError as exc:
    st.error(str(exc), title="Neplatný vstup – analýza sa nespustila", icon=":material/error:")
    st.stop()
except PermissionError as exc:  # TestSetLockedError
    st.error(str(exc), title="Záznam nie je dostupný", icon=":material/lock:")
    st.stop()
except (OSError, KeyError) as exc:
    st.error(
        f"Záznam sa nepodarilo načítať: {exc}. Ak nemáš stiahnutý dataset PTB-XL "
        "(uv run ekg-download), použi zdroj Ukážka.",
        icon=":material/error:",
    )
    st.stop()

result, filtered, recon = analysis
a = result.anomaly

# --- výsledky -----------------------------------------------------------------------

st.subheader(f"Záznam {result.record_id}")
st.markdown(f"**{verdict(result)}**")
with st.container(horizontal=True):
    st.metric(
        "Anomaly score (Model A)",
        f"{a.anomaly_score:.4f}",
        help="Rekonštrukčná chyba autoencodera, priemer cez 12 zvodov.",
        border=True,
    )
    st.metric("Prah anomálie", f"{a.threshold:.4f}", border=True)
    st.metric("Model A", "anomália" if a.is_anomaly else "v norme", border=True)
    positive = [name for name, lab in result.labels.items() if lab.positive]
    st.metric("Model B – pozitívne", ", ".join(positive) or "žiadna", border=True)
    st.metric("Súlad modelov", "áno" if result.consistency.models_agree else "nie", border=True)

if not result.consistency.models_agree:
    st.warning(result.consistency.note, title="Modely nesúhlasia", icon=":material/compare_arrows:")
if result.quality.issues:
    st.info("; ".join(result.quality.issues), title="Kvalita signálu", icon=":material/graphic_eq:")

left, right = st.columns([3, 2])
with left:
    st.dataframe(
        app_support.results_table(result),
        hide_index=True,
        column_config={
            "Pravdepodobnosť": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Prah": st.column_config.NumberColumn(format="%.3f"),
        },
        alt="Pravdepodobnosti piatich porúch s prahmi a výsledkom",
    )
with right:
    st.dataframe(
        app_support.lead_errors_table(result),
        hide_index=True,
        column_config={"Chyba": st.column_config.NumberColumn(format="%.4f")},
        height=220,
        alt="Rekonštrukčná chyba po zvodoch",
    )

show_recon = st.toggle("Zobraziť rekonštrukciu autoencodera", value=True, key="show_recon")
fig = plot_report(filtered, recon, result, show_reconstruction=show_recon)
st.pyplot(
    fig,
    alt="12 zvodov EKG s rekonštrukciou, anomaly score a pravdepodobnosti porúch",
)
png = io.BytesIO()
fig.savefig(png, format="png", dpi=110)
plt.close(fig)

with st.container(horizontal=True):
    st.download_button(
        "Výsledok JSON",
        result.to_json(),
        file_name=f"ekg_{result.record_id}.json",
        mime="application/json",
        icon=":material/download:",
        on_click="ignore",
    )
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, export.CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(export.to_row(result))
    st.download_button(
        "Výsledok CSV",
        csv_buffer.getvalue(),
        file_name=f"ekg_{result.record_id}.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
    )
    st.download_button(
        "Obrázok PNG",
        png.getvalue(),
        file_name=f"ekg_{result.record_id}.png",
        mime="image/png",
        icon=":material/download:",
        on_click="ignore",
    )

st.caption(
    f"Modely: {result.model_versions['autoencoder']} · {result.model_versions['classifier']}"
)
