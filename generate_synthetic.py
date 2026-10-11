import pandas as pd
import numpy as np
import json
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path.home() / "Downloads" / "test5.csv"

OUTPUT_FILE = Path.home() / "Downloads" / "synthetic_scada_test.csv"

N_NORMAL = 300
N_ATTACK = 300

RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# LOAD FEATURES USED BY YOUR MODEL
# ============================================================

with open("features.json", "r") as f:
    feature_names = json.load(f)

print(f"Model expects {len(feature_names)} features.")


# ============================================================
# LOAD HAI DATA
# ============================================================

print("Loading HAI 21.03 test5.csv...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# SELECT NORMAL HAI ROWS
# ============================================================

# test5 contains the original attack label.
normal_df = df[df["attack"] == 0].copy()

if len(normal_df) < N_NORMAL:
    raise ValueError("Not enough normal rows available.")


# Randomly select normal operating samples
normal_samples = normal_df.sample(
    n=N_NORMAL,
    random_state=RANDOM_STATE
).copy()


# ============================================================
# CREATE ATTACK SAMPLES
# ============================================================

attack_samples = normal_samples.copy()


# Select numerical feature columns
numeric_features = []

for feature in feature_names:

    if pd.api.types.is_numeric_dtype(
        attack_samples[feature]
    ):
        numeric_features.append(feature)


print(
    f"Numeric features available: "
    f"{len(numeric_features)}"
)


# ------------------------------------------------------------
# Inject synthetic anomalies
# ------------------------------------------------------------

for i in range(N_ATTACK):

    row_index = attack_samples.index[i]

    # Select several random SCADA features
    selected_features = rng.choice(
        numeric_features,
        size=min(8, len(numeric_features)),
        replace=False
    )

    for feature in selected_features:

        value = attack_samples.loc[
            row_index,
            feature
        ]

        # Avoid modifying missing values
        if pd.isna(value):
            continue

        # Strong but bounded perturbation
        magnitude = rng.uniform(
            2.0,
            5.0
        )

        direction = rng.choice(
            [-1, 1]
        )

        attack_samples.loc[
            row_index,
            feature
        ] = value * (
            1 + direction * magnitude
        )


# ============================================================
# LABELS
# ============================================================

normal_samples["synthetic_label"] = 0

attack_samples["synthetic_label"] = 1


# ============================================================
# COMBINE
# ============================================================

synthetic = pd.concat(
    [
        normal_samples,
        attack_samples
    ],
    ignore_index=True
)


# Shuffle
synthetic = synthetic.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# KEEP HAI-LIKE STRUCTURE
# ============================================================

# Keep timestamp if available,
# the 79 model features,
# and our synthetic label.

columns_to_keep = []

if "time" in synthetic.columns:
    columns_to_keep.append("time")

columns_to_keep += feature_names

columns_to_keep.append(
    "synthetic_label"
)

synthetic = synthetic[
    columns_to_keep
]


# ============================================================
# SAVE
# ============================================================

synthetic.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 50)
print("SYNTHETIC SCADA DATA CREATED")
print("=" * 50)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Shape:",
    synthetic.shape
)

print(
    "Normal:",
    (synthetic["synthetic_label"] == 0).sum()
)

print(
    "Synthetic Attack:",
    (synthetic["synthetic_label"] == 1).sum()
)

print()
print("File ready for WT Sentinel.")