"""Staged diagnostic script to isolate which dependency/operation segfaults
on Streamlit Community Cloud. Each stage prints a checkpoint before moving
on, so if the process crashes, the last visible checkpoint pinpoints it.
"""

from pathlib import Path

import streamlit as st

st.write("Stage 0: Streamlit import OK")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

st.write("Stage 1: pandas import OK")

import numpy as np

st.write(f"Stage 2: numpy import OK (version {np.__version__})")

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "koi_clean.csv")
st.write(f"Stage 3: CSV loaded OK, shape={df.shape}")

import sklearn

st.write(f"Stage 4: scikit-learn import OK (version {sklearn.__version__})")

import joblib

st.write("Stage 5: joblib import OK")

model = joblib.load(PROJECT_ROOT / "data" / "processed" / "model.pkl")
st.write("Stage 6: model.pkl loaded OK")

sample = df.drop(columns=["koi_disposition"]).head(1)
prediction = model.predict(sample)
st.write(f"Stage 7: prediction OK: {prediction[0]}")

import matplotlib

st.write(f"Stage 8: matplotlib import OK (version {matplotlib.__version__})")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
st.write("Stage 9: matplotlib figure created OK")

import plotly

st.write(f"Stage 10: plotly import OK (version {plotly.__version__})")

import plotly.express as px

fig2 = px.scatter(df, x="koi_period", y="koi_prad")
st.write("Stage 11: plotly express chart created OK")

fig3 = px.scatter(
    df,
    x="koi_period",
    y="koi_prad",
    color="koi_disposition",
    color_discrete_map={
        "CONFIRMED": "#2ecc71",
        "CANDIDATE": "#f1c40f",
        "FALSE POSITIVE": "#e74c3c",
    },
    log_x=True,
    log_y=True,
    hover_data={col: True for col in df.columns},
    opacity=0.6,
)
fig3.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="Disposition",
)
st.plotly_chart(fig3, use_container_width=True)
st.write("Stage 12: rich plotly chart (log scale + hover_data + color map) created OK")

with open(PROJECT_ROOT / "reports" / "confusion_matrix.png", "rb") as f:
    st.image(f.read())
st.write("Stage 13: confusion_matrix.png loaded and displayed OK")

with open(PROJECT_ROOT / "reports" / "feature_importance.png", "rb") as f:
    st.image(f.read())
st.write("Stage 14: feature_importance.png loaded and displayed OK")

tab_a, tab_b = st.tabs(["Tab A", "Tab B"])
with tab_a:
    st.write("Tab A rendering")
    df_a = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "koi_clean.csv")
    st.write(df_a.shape)
with tab_b:
    st.write("Tab B rendering")
    model_b = joblib.load(PROJECT_ROOT / "data" / "processed" / "model.pkl")
    st.write(model_b.predict(sample)[0])
st.write("Stage 15: st.tabs() with simultaneous rendering OK")

st.success("ALL STAGES PASSED — no crash!")
