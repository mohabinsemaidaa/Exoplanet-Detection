"""Staged diagnostic script to isolate which dependency/operation segfaults
on Streamlit Community Cloud. Each stage prints a checkpoint before moving
on, so if the process crashes, the last visible checkpoint pinpoints it.
"""

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Exoplanet Detection", layout="wide", page_icon="🔭")

st.write("Stage -1: st.set_page_config OK")
st.write("Stage 0: Streamlit import OK")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

st.write("Stage 1: pandas import OK")

import numpy as np

st.write(f"Stage 2: numpy import OK (version {np.__version__})")

from src.preprocess import CORE_FEATURES, TARGET_COLUMN

st.write("Stage 2.5: from src.preprocess import CORE_FEATURES, TARGET_COLUMN OK")

from src.train import MODEL_COMPARISON_PATH, MODEL_PATH, load_clean_data

st.write("Stage 2.6: from src.train import ... OK (this imports sklearn.ensemble, "
         "sklearn.linear_model, sklearn.model_selection, sklearn.pipeline, "
         "sklearn.preprocessing all at once)")

from src.evaluate import CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH

st.write("Stage 2.7: from src.evaluate import ... OK (this imports matplotlib.pyplot "
         "and sklearn.metrics.ConfusionMatrixDisplay)")

df = load_clean_data()
st.write(f"Stage 3: CSV loaded via src.train.load_clean_data() OK, shape={df.shape}")

import joblib

st.write("Stage 5: joblib import OK")

model = joblib.load(MODEL_PATH)
st.write("Stage 6: model.pkl loaded via src.train.MODEL_PATH OK")

sample = df[CORE_FEATURES].head(1)
prediction = model.predict(sample)
st.write(f"Stage 7: prediction OK: {prediction[0]}")


def inject_theme():
    st.markdown(
        """
        <style>
        .hero {
            padding: 2.5rem 2rem;
            border-radius: 12px;
            background: radial-gradient(circle at top left, #1b2735 0%, #090a0f 100%);
            color: #e8ecf1;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>Exoplanet Detection</h1>
            <p>Diagnostic test of the hero banner's unsafe_allow_html injection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_theme()
show_hero()
st.write("Stage 8: inject_theme() + show_hero() (unsafe_allow_html CSS/HTML) OK")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
st.write("Stage 9: matplotlib figure created OK")

import plotly.express as px

fig3 = px.scatter(
    df,
    x="koi_period",
    y="koi_prad",
    color=TARGET_COLUMN,
    log_x=True,
    log_y=True,
    hover_data={feature: True for feature in CORE_FEATURES},
    opacity=0.6,
)
fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)
st.write("Stage 12: rich plotly chart created OK")

st.image(str(CONFUSION_MATRIX_PATH))
st.write("Stage 13: confusion_matrix.png loaded OK")

st.image(str(FEATURE_IMPORTANCE_PATH))
st.write("Stage 14: feature_importance.png loaded OK")

comparison_df = pd.read_csv(MODEL_COMPARISON_PATH).set_index("model")
st.bar_chart(comparison_df["accuracy"])
st.write("Stage 14.5: model_comparison.csv bar chart OK")

tab_a, tab_b = st.tabs(["Tab A", "Tab B"])
with tab_a:
    st.write("Tab A rendering")
    st.write(df.shape)
with tab_b:
    st.write("Tab B rendering")
    st.write(model.predict(sample)[0])
st.write("Stage 15: st.tabs() with simultaneous rendering OK")

st.success("ALL STAGES PASSED — no crash!")
