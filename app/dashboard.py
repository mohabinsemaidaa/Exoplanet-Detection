"""Streamlit dashboard for the Exoplanet Detection project."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH
from src.preprocess import CORE_FEATURES, TARGET_COLUMN
from src.train import MODEL_PATH, load_clean_data

st.set_page_config(page_title="Exoplanet Detection", layout="wide")


def show_overview():
    st.title("Exoplanet Detection")
    st.write(
        "This project predicts whether an astronomical observation from "
        "NASA's Kepler mission is likely to be a confirmed exoplanet, a "
        "candidate, or a false positive."
    )
    st.subheader("Dataset")
    st.write(
        "Kepler Objects of Interest (KOI) cumulative table, from the "
        "NASA Exoplanet Archive."
    )
    st.subheader("Approach")
    st.write(
        "Three models were compared — Logistic Regression, Random Forest, "
        "and Gradient Boosting — trained on 12 core physical features "
        "(orbital period, transit depth/duration, planet radius, stellar "
        "properties, and signal-to-noise ratio). The best-performing model "
        "is used for predictions in this dashboard."
    )


def show_dataset_exploration():
    st.header("Dataset Exploration")
    df = load_clean_data()
    st.write(f"{len(df)} rows after cleaning")

    st.subheader("Class Distribution")
    st.bar_chart(df[TARGET_COLUMN].value_counts())

    st.subheader("Feature Distribution")
    feature = st.selectbox("Choose a feature", CORE_FEATURES)
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=50)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)


def show_model_performance():
    st.header("Model Performance")
    st.subheader("Confusion Matrix")
    st.image(str(CONFUSION_MATRIX_PATH))
    st.subheader("Feature Importance")
    st.image(str(FEATURE_IMPORTANCE_PATH))


def show_prediction():
    st.header("Predict")
    st.write("Enter the observed values for a candidate signal:")

    inputs = {}
    columns = st.columns(2)
    for i, feature in enumerate(CORE_FEATURES):
        with columns[i % 2]:
            inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([inputs])[CORE_FEATURES]

        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0])

        st.success(f"Prediction: **{prediction}**")
        st.write(f"Confidence: {confidence:.1%}")


PAGES = {
    "Overview": show_overview,
    "Dataset Exploration": show_dataset_exploration,
    "Model Performance": show_model_performance,
    "Prediction": show_prediction,
}

page = st.sidebar.radio("Page", list(PAGES.keys()))
PAGES[page]()
