"""Streamlit dashboard for the Exoplanet Detection project."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH
from src.preprocess import CORE_FEATURES, TARGET_COLUMN
from src.train import MODEL_COMPARISON_PATH, MODEL_PATH, load_clean_data

st.set_page_config(page_title="Exoplanet Detection", layout="wide", page_icon="🔭")

EARTH_RADIUS_FEATURE = "koi_prad"

FEATURE_LABELS = {
    "koi_period": "Orbital Period (days)",
    "koi_duration": "Transit Duration (hours)",
    "koi_depth": "Transit Depth (ppm)",
    "koi_prad": "Planet Radius (Earth radii)",
    "koi_teq": "Equilibrium Temperature (K)",
    "koi_insol": "Insolation Flux (Earth flux)",
    "koi_model_snr": "Transit Signal-to-Noise",
    "koi_steff": "Stellar Effective Temperature (K)",
    "koi_slogg": "Stellar Surface Gravity (log g)",
    "koi_srad": "Stellar Radius (Solar radii)",
    "koi_kepmag": "Kepler Magnitude",
    "koi_impact": "Impact Parameter",
}

EXPLORER_FEATURES = ["koi_period", "koi_prad", "koi_depth", "koi_model_snr"]

DISPOSITION_COLORS = {
    "CONFIRMED": "#2ecc71",
    "CANDIDATE": "#f1c40f",
    "FALSE POSITIVE": "#e74c3c",
}


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
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: 0.02em;
        }
        .hero p {
            margin-top: 0.5rem;
            color: #a9b4c2;
            font-size: 1.05rem;
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
            <p>Classifying NASA Kepler Objects of Interest as confirmed exoplanets,
            candidates, or false positives, based on transit and stellar
            measurements from the Kepler mission.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_summary_cards(df: pd.DataFrame):
    counts = df[TARGET_COLUMN].value_counts()
    columns = st.columns(4)

    columns[0].metric("Total Observations", f"{len(df):,}")
    columns[1].metric("Confirmed Exoplanets", f"{counts.get('CONFIRMED', 0):,}")
    columns[2].metric("Candidates", f"{counts.get('CANDIDATE', 0):,}")
    columns[3].metric("False Positives", f"{counts.get('FALSE POSITIVE', 0):,}")


def show_overview():
    show_hero()
    df = load_clean_data()
    show_summary_cards(df)

    st.subheader("About This Project")
    st.write(
        "This project uses the Kepler Objects of Interest (KOI) cumulative "
        "table from the NASA Exoplanet Archive. Each row is a candidate "
        "transit signal, described by measurements such as orbital period, "
        "transit depth, planet radius, and stellar properties."
    )
    st.write(
        "Three models were compared — Logistic Regression, Random Forest, "
        "and Gradient Boosting — and the best performer is used for "
        "predictions in the **Make a Prediction** tab."
    )


def show_dataset_explorer():
    st.header("Dataset Explorer")
    df = load_clean_data()

    st.write(
        "Explore how confirmed exoplanets, candidates, and false positives "
        "differ across key physical measurements. Choose two features below."
    )

    col1, col2 = st.columns(2)
    x_feature = col1.selectbox(
        "X-axis", EXPLORER_FEATURES, index=0, format_func=lambda f: FEATURE_LABELS[f]
    )
    y_feature = col2.selectbox(
        "Y-axis", EXPLORER_FEATURES, index=1, format_func=lambda f: FEATURE_LABELS[f]
    )

    log_axes = st.checkbox(
        "Use log scale (helpful since these features span a wide range)",
        value=True,
    )

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=TARGET_COLUMN,
        color_discrete_map=DISPOSITION_COLORS,
        log_x=log_axes,
        log_y=log_axes,
        labels={**FEATURE_LABELS, TARGET_COLUMN: "Disposition"},
        hover_data={feature: True for feature in CORE_FEATURES},
        opacity=0.6,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Disposition",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_performance():
    st.header("Model Performance")

    st.subheader("Model Comparison")
    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH).set_index("model")
    st.bar_chart(comparison_df["accuracy"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        st.image(str(CONFUSION_MATRIX_PATH))
    with col2:
        st.subheader("Feature Importance")
        st.image(str(FEATURE_IMPORTANCE_PATH))


def show_result_card(prediction: str, confidence: float):
    message = f"Predicted disposition: **{prediction}** ({confidence:.1%} confidence)"
    if prediction == "CONFIRMED":
        st.success(message)
    elif prediction == "CANDIDATE":
        st.warning(message)
    else:
        st.error(message)


def show_earth_comparison(radius_in_earth_radii: float):
    if radius_in_earth_radii <= 0:
        return
    st.caption(
        f"🌍 At {radius_in_earth_radii:.2f} Earth radii, this candidate is "
        f"{radius_in_earth_radii:.1f}x the size of Earth."
    )


def show_prediction():
    st.header("Make a Prediction")
    st.write(
        "Enter the observed values for a candidate transit signal. Fields "
        "default to the dataset's median values as a starting point."
    )

    df = load_clean_data()
    medians = df[CORE_FEATURES].median()

    inputs = {}
    columns = st.columns(2)
    for i, feature in enumerate(CORE_FEATURES):
        with columns[i % 2]:
            inputs[feature] = st.number_input(
                FEATURE_LABELS[feature], value=float(medians[feature])
            )

    if st.button("Predict"):
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([inputs])[CORE_FEATURES]

        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0])

        show_result_card(prediction, confidence)
        show_earth_comparison(inputs[EARTH_RADIUS_FEATURE])


inject_theme()

overview_tab, explorer_tab, performance_tab, prediction_tab = st.tabs(
    ["Overview", "Dataset Explorer", "Model Performance", "Make a Prediction"]
)

with overview_tab:
    show_overview()

with explorer_tab:
    show_dataset_explorer()

with performance_tab:
    show_model_performance()

with prediction_tab:
    show_prediction()
