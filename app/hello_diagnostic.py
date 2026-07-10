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

st.success("ALL STAGES PASSED — no crash!")
