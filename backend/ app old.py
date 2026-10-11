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



# ============================================================
# WT SENTINEL — WIND TURBINE SECURITY UI
# ============================================================

st.markdown("""
<style>
/* ---------- Global ---------- */
.stApp {
    background:
        radial-gradient(circle at 80% 10%, rgba(45, 212, 191, 0.08), transparent 30%),
        linear-gradient(135deg, #06131a 0%, #081d25 45%, #031015 100%);
    color: #e8f7f7;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #041117 0%, #071d24 100%);
    border-right: 1px solid rgba(74, 222, 206, 0.18);
}

[data-testid="stSidebar"] * {
    color: #dff8f5;
}

h1, h2, h3 {
    color: #eaffff !important;
}

.hero {
    padding: 22px 28px;
    border: 1px solid rgba(74, 222, 206, .20);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(10, 42, 51, .92), rgba(3, 20, 27, .85));
    box-shadow: 0 12px 45px rgba(0,0,0,.25);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: .5px;
}

.hero-sub {
    color: #94bfc1;
    font-size: 15px;
    margin-top: 4px;
}

.turbine-stage {
    min-height: 360px;
    border-radius: 25px;
    border: 1px solid rgba(74,222,206,.18);
    background:
        radial-gradient(circle at 50% 50%, rgba(45,212,191,.12), transparent 28%),
        linear-gradient(180deg, rgba(7,39,48,.88), rgba(3,17,23,.96));
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 18px;
}

.wind {
    position: absolute;
    width: 260px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,255,250,.35), transparent);
    animation: windmove 3s linear infinite;
}
.wind:nth-child(1) { top: 25%; left: -20%; animation-delay: 0s; }
.wind:nth-child(2) { top: 42%; left: -35%; animation-delay: 1s; width: 340px; }
.wind:nth-child(3) { top: 67%; left: -25%; animation-delay: 2s; width: 300px; }

@keyframes windmove {
    from { transform: translateX(0); opacity: 0; }
    20% { opacity: 1; }
    80% { opacity: 1; }
    to { transform: translateX(620px); opacity: 0; }
}

.turbine {
    position: relative;
    width: 170px;
    height: 260px;
}

.tower {
    position: absolute;
    width: 18px;
    height: 190px;
    left: 76px;
    bottom: 0;
    background: linear-gradient(90deg, #78999d, #e1f7f5, #6d8d91);
    clip-path: polygon(25% 0, 75% 0, 100% 100%, 0 100%);
    border-radius: 3px;
}

.nacelle {
    position: absolute;
    width: 72px;
    height: 35px;
    left: 55px;
    top: 48px;
    border-radius: 20px 25px 15px 15px;
    background: linear-gradient(145deg, #e6ffff, #6f9da0);
    box-shadow: 0 0 25px rgba(74,222,206,.20);
}

.hub {
    position: absolute;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #dff;
    left: 43px;
    top: 55px;
    z-index: 3;
    box-shadow: 0 0 18px rgba(255,255,255,.3);
}

.blades {
    position: absolute;
    left: 50px;
    top: 22px;
    width: 45px;
    height: 80px;
    transform-origin: 4px 43px;
    animation: spin 3s linear infinite;
    z-index: 2;
}

.blade {
    position: absolute;
    left: 2px;
    top: 40px;
    width: 95px;
    height: 8px;
    border-radius: 100% 10% 10% 100%;
    background: linear-gradient(90deg, #efffff, #789da0);
    transform-origin: 3px 4px;
}
.blade:nth-child(2) { transform: rotate(120deg); }
.blade:nth-child(3) { transform: rotate(240deg); }

@keyframes spin { to { transform: rotate(360deg); } }

.status-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(45,212,191,.10);
    border: 1px solid rgba(45,212,191,.35);
    color: #73f3df;
    font-weight: 700;
    font-size: 13px;
}

.attack-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(248,113,113,.12);
    border: 1px solid rgba(248,113,113,.35);
    color: #ff9999;
    font-weight: 700;
    font-size: 13px;
}

.panel {
    border: 1px solid rgba(74,222,206,.16);
    border-radius: 18px;
    padding: 20px;
    background: rgba(4, 25, 32, .72);
    margin-bottom: 18px;
}

.big-number {
    font-size: 30px;
    font-weight: 800;
    color: #eaffff;
}

.muted {
    color: #8eb4b7;
    font-size: 13px;
}

.explain {
    border-left: 4px solid #4ddcca;
    padding: 12px 16px;
    background: rgba(45,212,191,.07);
    border-radius: 0 12px 12px 0;
}

.source-node {
    text-align: center;
    padding: 15px;
    border: 1px solid rgba(74,222,206,.20);
    border-radius: 14px;
    background: rgba(7,39,48,.8);
    margin: 8px;
}

.playbook-step {
    padding: 13px 16px;
    margin: 8px 0;
    border-radius: 12px;
    background: rgba(7,39,48,.75);
    border: 1px solid rgba(74,222,206,.13);
}

.footer {
    text-align: center;
    color: #668e92;
    font-size: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌬️ WT SENTINEL")
    st.caption("Wind Turbine Cyber Defense Console")
    st.markdown("---")

    page = st.radio(
        "INVESTIGATION CONSOLE",
        [
            "⚡ Turbine Dashboard",
            "🔍 Why Attack?",
            "🌐 Attack Source",
            "🛡️ Response Playbook",
            "📊 SCADA Analytics"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 📡 DATA LINK")
    uploaded_file = st.file_uploader(
        "Load HAI 21.03 SCADA CSV",
        type=["csv"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown(
        "<div class='muted'>ML Engine<br><b>Tuned MLP</b><br><br>"
        "Feature Set<br><b>79 SCADA parameters</b></div>",
        unsafe_allow_html=True
    )

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">🌬️ WT Sentinel</div>
    <div class="hero-sub">
        Wind Turbine SCADA Cyberattack Detection & Response Platform
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Turbine visual ----------
turbine_html = """
<div class="turbine-stage">
    <div class="wind"></div><div class="wind"></div><div class="wind"></div>
    <div class="turbine">
        <div class="tower"></div>
        <div class="nacelle"></div>
        <div class="blades">
            <div class="blade"></div>
            <div class="blade"></div>
            <div class="blade"></div>
        </div>
        <div class="hub"></div>
    </div>
</div>
"""

# ---------- Load + predict ----------
df = None
results = None
total = attacks = normal = 0
attack_percentage = 0.0
probabilities = np.array([])
predictions = np.array([])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing = [feature for feature in feature_names if feature not in df.columns]

    if missing:
        st.error(f"Missing {len(missing)} required features.")
        st.caption("The uploaded CSV cannot be analyzed until the required SCADA features are present.")
    else:
        X = df[feature_names].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        X_scaled = scaler.transform(X)

        probabilities = model.predict(X_scaled, verbose=0).ravel()
        predictions = (probabilities >= threshold).astype(int)

        total = len(predictions)
        attacks = int(np.sum(predictions == 1))
        normal = int(np.sum(predictions == 0))
        attack_percentage = attacks / total * 100 if total else 0

        results = pd.DataFrame({
            "Attack Probability": probabilities,
            "Prediction": np.where(predictions == 1, "Attack", "Normal"),
            "Risk": np.where(
                probabilities >= 0.95,
                "Critical",
                np.where(probabilities >= threshold, "High", "Low")
            )
        })

# ---------- PAGE: TURBINE DASHBOARD ----------
if page == "⚡ Turbine Dashboard":
    st.markdown(turbine_html, unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("📡 Load a HAI 21.03 CSV from the sidebar to start the SCADA security analysis.")

        c1, c2, c3 = st.columns(3)
        c1.metric("SCADA Link", "STANDBY")
        c2.metric("ML Engine", "READY")
        c3.metric("Threat State", "WAITING")

    elif results is not None:
        status = "🔴 ATTACK DETECTED" if attacks > 0 else "🟢 TURBINE SECURE"

        st.markdown(
            f"<span class='{'attack-pill' if attacks > 0 else 'status-pill'}'>{status}</span>",
            unsafe_allow_html=True
        )
        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SCADA Records", f"{total:,}")
        c2.metric("Normal", f"{normal:,}")
        c3.metric("Potential Attacks", f"{attacks:,}")
        c4.metric("Attack Rate", f"{attack_percentage:.2f}%")

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### 📡 Live Detection Stream")
        st.dataframe(results.head(100), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PAGE: WHY ATTACK ----------
elif page == "🔍 Why Attack?":
    st.markdown("## 🔍 Why was this considered an attack?")
    st.caption("Explain the model's detection using the SCADA evidence available in the uploaded data.")

    if results is None:
        st.info("Load and analyze a CSV first. The attack explanation will appear here.")
    elif attacks == 0:
        st.success("🟢 No records crossed the configured attack threshold.")
        st.write("There is currently no detected attack event to explain.")
    else:
        attack_idx = int(np.argmax(probabilities))
        max_probability = float(probabilities[attack_idx])

        st.markdown(
            f"<div class='panel'><span class='attack-pill'>HIGH-RISK EVENT</span>"
            f"<h2>Attack probability: {max_probability:.2%}</h2>"
            f"<p class='muted'>Threshold used by WT Sentinel: {threshold:.4f}</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="explain">
        <b>Detection reasoning</b><br>
        WT Sentinel considers a record suspicious when the trained MLP produces
        an attack probability at or above the configured threshold. The
        explanation below is based on the detected SCADA event and does not
        claim an individual physical cause unless that information exists in
        the dataset.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚨 Detection Evidence")

        e1, e2, e3 = st.columns(3)
        e1.metric("Detected Attacks", f"{attacks:,}")
        e2.metric("Highest Probability", f"{max_probability:.2%}")
        e3.metric("Configured Threshold", f"{threshold:.2%}")

        st.markdown("### 📈 Attack Probability Profile")
        st.line_chart(results["Attack Probability"].head(200))

        st.markdown("### 🧾 Highest-Risk Records")
        top = results.sort_values("Attack Probability", ascending=False).head(10)
        st.dataframe(top, use_container_width=True, hide_index=True)

# ---------- PAGE: ATTACK SOURCE ----------
elif page == "🌐 Attack Source":
    st.markdown("## 🌐 Attack Source Investigation")
    st.caption("Trace the suspected attack path through the wind-turbine SCADA environment.")

    if results is None:
        st.info("Load and analyze a CSV first.")
    elif attacks == 0:
        st.success("🟢 No active attack was detected, so source investigation is not currently triggered.")
    else:
        st.markdown("""
        <div class="panel">
            <h3>⚠️ Source Attribution Status</h3>
            <p>
            <b>Inferred source:</b> SCADA data / communication layer
            </p>
            <p class="muted">
            This is an inference from the affected telemetry. The HAI 21.03
            input used here does not, by itself, establish an attacker IP address
            or a specific human/source identity.
            </p>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.markdown("<div class='source-node'>📡<br><b>SCADA Network</b><br><span class='muted'>Telemetry Layer</span></div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='source-node'>⚠️<br><b>Suspicious Data</b><br><span class='muted'>ML Detection</span></div>", unsafe_allow_html=True)
        with c:
            st.markdown("<div class='source-node'>🌬️<br><b>Wind Turbine</b><br><span class='muted'>Affected System</span></div>", unsafe_allow_html=True)

        st.markdown("### 🔎 Investigation Checklist")
        for item in [
            "Validate the abnormal SCADA measurements against the expected operating range.",
            "Compare suspicious records with neighboring normal records.",
            "Inspect the corresponding SCADA/controller logs.",
            "Check communication activity around the detected event.",
            "Correlate the event with turbine operational conditions."
        ]:
            st.markdown(f"☐ {item}")

# ---------- PAGE: RESPONSE PLAYBOOK ----------
elif page == "🛡️ Response Playbook":
    st.markdown("## 🛡️ Wind Turbine Incident Response Playbook")
    st.caption("Generate a response workflow from the detected security condition.")

    if results is None:
        st.info("Load and analyze a CSV first.")
    elif attacks == 0:
        st.success("🟢 No attack detected. A containment playbook is not currently required.")
    else:
        max_probability = float(np.max(probabilities))
        severity = "CRITICAL" if max_probability >= 0.95 else "HIGH"

        st.markdown(
            f"<div class='panel'><span class='attack-pill'>SEVERITY: {severity}</span>"
            f"<h2>SCADA Cyberattack Response</h2>"
            f"<p>Detected attack records: <b>{attacks:,}</b> | "
            f"Highest confidence: <b>{max_probability:.2%}</b></p></div>",
            unsafe_allow_html=True
        )

        if st.button("⚡ GENERATE RESPONSE PLAYBOOK", use_container_width=True):
            st.session_state["playbook_generated"] = True

        if st.session_state.get("playbook_generated", False):
            steps = [
                ("01", "🔍", "Validate SCADA readings", "Compare suspicious telemetry with trusted operating values."),
                ("02", "📊", "Check the baseline", "Compare the event against nearby normal operating records."),
                ("03", "📡", "Inspect SCADA communication", "Review controller, gateway and communication logs."),
                ("04", "🔒", "Contain the affected path", "Isolate or restrict the affected communication path according to the incident procedure."),
                ("05", "🧹", "Investigate the event", "Preserve relevant logs and determine the extent of the anomaly."),
                ("06", "🔄", "Restore trusted operation", "Return validated components/configuration to normal operation."),
                ("07", "👁️", "Continue monitoring", "Watch subsequent SCADA records for recurring anomalous behavior.")
            ]

            st.markdown("### 📋 Generated Playbook")
            for num, icon, title, desc in steps:
                st.markdown(
                    f"<div class='playbook-step'><b>{num} &nbsp; {icon} {title}</b>"
                    f"<br><span class='muted'>{desc}</span></div>",
                    unsafe_allow_html=True
                )

# ---------- PAGE: ANALYTICS ----------
elif page == "📊 SCADA Analytics":
    st.markdown("## 📊 SCADA Security Analytics")

    if results is None:
        st.info("Load and analyze a CSV first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", "Tuned MLP")
        c2.metric("Features", "79")
        c3.metric("Threshold", f"{threshold:.4f}")
        c4.metric("Records", f"{total:,}")

        st.markdown("### 📈 Attack Probability")
        st.line_chart(results["Attack Probability"].head(500))

        st.markdown("### 📋 Detection Stream")
        st.dataframe(results.head(500), use_container_width=True, hide_index=True)

st.markdown(
    "<div class='footer'>WT Sentinel · HAI 21.03 · Tuned MLP · Wind Turbine SCADA Cyber Defense</div>",
    unsafe_allow_html=True
)
