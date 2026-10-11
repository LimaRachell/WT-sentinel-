import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

st.set_page_config(
    page_title="WT Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Load model and preprocessing files
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("mlp_tuned.keras")

    scaler = joblib.load("scaler.pkl")

    with open("features.json", "r") as f:
        features = json.load(f)

    with open("threshold.txt", "r") as f:
        threshold = float(f.read())

    return model, scaler, features, threshold


model, scaler, feature_names, threshold = load_model()


# Header
st.title("🛡️ WT Sentinel")

st.subheader(
    "Wind Turbine SCADA Cyberattack Detection System"
)

st.write(
    "ML-based detection of cyberattacks in wind turbine SCADA data."
)


# Model information
st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Tuned MLP")
col2.metric("Features", "79")
col3.metric("Threshold", threshold)


# Upload
st.markdown("---")

st.header("SCADA Attack Detection")

uploaded_file = st.file_uploader(
    "Upload HAI 21.03 CSV",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"Loaded {len(df):,} records"
    )

    # Check required features
    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:

        st.error(
            f"Missing {len(missing)} required features."
        )

    else:

        # Select the 79 features
        X = df[feature_names].copy()

        # Convert to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Handle missing values
        X = X.fillna(X.median())

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        probabilities = model.predict(
            X_scaled,
            verbose=0
        ).ravel()

        # Apply threshold
        predictions = (
            probabilities >= threshold
        ).astype(int)

        # Counts
        total = len(predictions)

        attacks = int(
            np.sum(predictions == 1)
        )

        normal = int(
            np.sum(predictions == 0)
        )

        attack_percentage = (
            attacks / total * 100
        )

        # Results
        st.markdown("---")

        st.header("Detection Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Records",
            f"{total:,}"
        )

        c2.metric(
            "Normal",
            f"{normal:,}"
        )

        c3.metric(
            "Attacks",
            f"{attacks:,}"
        )

        c4.metric(
            "Attack Rate",
            f"{attack_percentage:.2f}%"
        )

        if attacks > 0:

            st.error(
                f" {attacks:,} potential attacks detected"
            )

        else:

            st.success(
                " No attacks detected"
            )

        # Detection table
        results = pd.DataFrame({

            "Attack Probability":
                probabilities,

            "Prediction":
                np.where(
                    predictions == 1,
                    "Attack",
                    "Normal"
                ),

            "Risk":
                np.where(
                    probabilities >= 0.95,
                    "Critical",

                    np.where(
                        probabilities >= threshold,
                        "High",
                        "Low"
                    )
                )
        })

        st.subheader("Detection Stream")

        st.dataframe(
            results.head(100),
            use_container_width=True
        )

        # Chart
        st.subheader(
            "Attack Probability"
        )

        st.line_chart(
            results[
                "Attack Probability"
            ].head(200)
        )


st.markdown("---")

st.caption(
    "WT Sentinel | HAI 21.03 | Tuned MLP"
)