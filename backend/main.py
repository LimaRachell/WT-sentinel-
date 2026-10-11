from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json
import io


# ==========================================
# WT SENTINEL - FASTAPI BACKEND
# ==========================================

app = FastAPI(
    title="WT Sentinel API",
    description="Wind Turbine SCADA Cyberattack Detection API",
    version="1.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model("mlp_tuned.keras")

scaler = joblib.load("scaler.pkl")

with open("features.json", "r") as f:
    feature_names = json.load(f)

with open("threshold.txt", "r") as f:
    threshold = float(f.read())


print("Model loaded successfully!")
print("Number of features:", len(feature_names))
print("Detection threshold:", threshold)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "WT Sentinel",
        "model": "Tuned MLP",
        "features": len(feature_names),
        "threshold": threshold
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():

    return {

        "model": "Tuned MLP",

        "features": len(feature_names),

        "threshold": threshold,

        "accuracy": 0.9540,

        "precision": 0.2976,

        "recall": 0.6991,

        "f1": 0.4174,

        "roc_auc": 0.8526
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded CSV
    contents = await file.read()

    df = pd.read_csv(io.BytesIO(contents))


    # ======================================
    # CHECK REQUIRED FEATURES
    # ======================================

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        return {
            "success": False,
            "error": "Missing required features",
            "missing_features": missing_features
        }


    # ======================================
    # SELECT FEATURES
    # ======================================

    X = df[feature_names].copy()


    # Convert everything to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # Handle missing values
    X = X.fillna(X.median())


    # ======================================
    # SCALE DATA
    # ======================================

    X_scaled = scaler.transform(X)


    # ======================================
    # MODEL PREDICTION
    # ======================================

    probabilities = model.predict(
        X_scaled,
        verbose=0
    ).ravel()


    # Apply threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)


    # ======================================
    # SUMMARY
    # ======================================

    total_records = len(predictions)

    normal_count = int(
        np.sum(predictions == 0)
    )

    attack_count = int(
        np.sum(predictions == 1)
    )

    attack_percentage = (
        attack_count /
        total_records *
        100
    )


    # ======================================
    # CREATE RESULTS
    # ======================================

    results = df.copy()

    results["attack_probability"] = probabilities

    results["prediction"] = np.where(
        predictions == 1,
        "Attack",
        "Normal"
    )

    results["risk"] = np.where(
        probabilities >= 0.95,
        "Critical",

        np.where(
            probabilities >= threshold,
            "High",
            "Low"
        )
    )


    # ======================================
    # RETURN RESULTS
    # ======================================

    return {

        "success": True,

        "summary": {

            "total_records": total_records,

            "normal_records": normal_count,

            "attack_records": attack_count,

            "attack_percentage": round(
                attack_percentage,
                2
            )
        },

        "predictions": results[
            [
                "attack_probability",
                "prediction",
                "risk"
            ]
        ].to_dict(
            orient="records"
        )
    }