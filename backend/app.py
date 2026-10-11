import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import re
import json
from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

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




def build_playbook_pdf(playbook_text):
    """Create a readable PDF incident playbook in memory."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="WT Sentinel Incident Response Playbook", author="WT Sentinel"
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "WTTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=colors.HexColor("#0B5B61"),
        alignment=TA_CENTER, spaceAfter=8
    )
    heading_style = ParagraphStyle(
        "WTHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=colors.HexColor("#0B5B61"),
        spaceBefore=10, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "WTBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=12, spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        "WTBullet", parent=body_style, leftIndent=12, firstLineIndent=-8,
        spaceAfter=3
    )
    story = []
    lines = playbook_text.splitlines()
    if lines:
        story.append(Paragraph(escape(lines[0]), title_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B9A91")))
        story.append(Spacer(1, 8))
    for line in lines[1:]:
        clean = line.strip()
        if not clean:
            story.append(Spacer(1, 3))
        elif (clean.isupper() and len(clean) < 100) or re.match(r"^\\d+\\.\\s+[A-Z /()—-]+$", clean):
            story.append(Paragraph(escape(clean), heading_style))
        elif clean.endswith(":") and len(clean) < 70:
            story.append(Paragraph(escape(clean), heading_style))
        elif clean.startswith("- "):
            story.append(Paragraph("• " + escape(clean[2:]), bullet_style))
        else:
            story.append(Paragraph(escape(clean), body_style))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#54777A"))
        canvas.drawString(18 * mm, 9 * mm, "WT Sentinel | Incident triage — validate before operational action")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return buffer.getvalue()


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

        # The trained HAI MLP is binary-only. FDI is a working hypothesis,
        # not a trained or calibrated attack-type prediction.
        attack_type_hypothesis = (
            "False Data Injection (FDI)" if attacks > 0
            else "No attack crossed the MLP threshold"
        )

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

        if attacks > 0:
            st.markdown(
                f"<div class='panel'><span class='attack-pill'>PROVISIONAL ATTACK HYPOTHESIS</span>"
                f"<h2>🧪 {attack_type_hypothesis}</h2>"
                f"<p>The binary MLP flagged {attacks:,} records. Its highest attack score is "
                f"{float(np.max(probabilities)):.2%}. This score indicates Normal-vs-Attack "
                f"likelihood only; it is <b>not</b> the probability that the attack is FDI.</p>"
                f"<p class='muted'>FDI is an unverified working hypothesis because the current model "
                f"was not trained on FDI/replay/spoofing labels.</p></div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No records crossed the attack threshold. No attack type is assigned.")

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

        st.markdown("### 🧪 Working attack-type hypothesis")
        st.warning(
            "**False Data Injection (FDI) — provisional hypothesis for investigation.** "
            "The current HAI-trained MLP detects binary anomalies only; it does not classify "
            "FDI, replay, or spoofing. Do not treat this label as a verified attack type."
        )
        st.write(
            "To validate or reject the FDI hypothesis, compare flagged readings with independent "
            "or redundant sensors, expected physical relationships, approved operating limits, "
            "controller setpoints, and historian logs. This CSV alone cannot establish malicious intent."
        )

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
    st.caption(
        "Generate an evidence-led response plan for a wind-turbine company. "
        "Model detections are indicators for investigation, not proof of an intrusion."
    )

    if results is None:
        st.info("Load a compatible HAI 21.03 SCADA CSV from the sidebar first.")
    elif attacks == 0:
        st.success("No records crossed the configured MLP threshold.")
        st.warning(
            "No attack-triggered playbook was generated. This does not prove "
            "that the turbine or SCADA environment is free of cyber threats."
        )
    else:
        # ---------------- INCIDENT METRICS ----------------
        detected_indices = np.flatnonzero(predictions == 1)
        max_probability = float(np.max(probabilities))
        mean_probability = float(np.mean(probabilities[detected_indices]))
        attack_rate = (attacks / total * 100) if total else 0.0

        # A detection episode is a consecutive run of flagged rows.
        episodes = int(
            np.sum(
                (predictions == 1)
                & np.r_[True, predictions[:-1] == 0]
            )
        )

        # Prefer dataset timestamps; otherwise report row numbers rather than
        # inventing wall-clock attack times.
        first_detected_time = f"Record {int(detected_indices[0]) + 1}"
        last_detected_time = f"Record {int(detected_indices[-1]) + 1}"
        duration_hours = None
        incidents_per_hour = None
        timestamp_note = "No usable timestamp column; record positions are shown."

        if "time" in df.columns:
            parsed_times = pd.to_datetime(df["time"], errors="coerce", utc=True)
            first_event_time = parsed_times.iloc[int(detected_indices[0])]
            last_event_time = parsed_times.iloc[int(detected_indices[-1])]

            if pd.notna(first_event_time):
                first_detected_time = first_event_time.isoformat()
            if pd.notna(last_event_time):
                last_detected_time = last_event_time.isoformat()

            valid_times = parsed_times.dropna()
            if len(valid_times) >= 2:
                span_seconds = (
                    valid_times.max() - valid_times.min()
                ).total_seconds()
                if span_seconds > 0:
                    duration_hours = span_seconds / 3600.0
                    incidents_per_hour = episodes / duration_hours
                    timestamp_note = (
                        "Times are taken from the uploaded dataset and interpreted as UTC. "
                        "They may not be the current real-world time."
                    )
                else:
                    timestamp_note = (
                        "The timestamp span is zero or too short to calculate an hourly rate."
                    )

        # Rank unusual features using the same scaler used for the MLP.
        # This is a triage signal, not a causal explanation.
        top_features = []
        try:
            if "X_scaled" in globals() and len(detected_indices):
                feature_deviation = np.mean(
                    np.abs(X_scaled[detected_indices]), axis=0
                )
                ranked_features = np.argsort(feature_deviation)[::-1][:8]
                top_features = [
                    (feature_names[int(i)], float(feature_deviation[int(i)]))
                    for i in ranked_features
                ]
        except Exception:
            top_features = []

        severity = (
            "CRITICAL — urgent human review"
            if max_probability >= 0.95
            else "HIGH — investigate promptly"
        )

        # ---------------- OVERVIEW ----------------
        st.markdown(
            f"<div class='panel'><span class='attack-pill'>SEVERITY: {severity}</span>"
            f"<h3>Incident case: WT-SENTINEL-CSV-REVIEW</h3>"
            f"<p>Model: Tuned MLP | Threshold: {threshold:.2f} | "
            f"Highest binary attack score: {max_probability:.2%}</p>"
            f"<p><b>Working attack-type hypothesis:</b> False Data Injection (FDI) — provisional, unverified.</p>"
            f"<p>This is a model-generated triage case, not a confirmed cyber incident. "
            f"The binary MLP was not trained to distinguish FDI, replay, and spoofing.</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flagged Records", f"{attacks:,}")
        c2.metric("Flagged Record Rate", f"{attack_rate:.2f}%")
        c3.metric("Detection Episodes", f"{episodes:,}")
        c4.metric("Mean Score on Flagged Rows", f"{mean_probability:.2%}")

        st.markdown("### 1. Time of attack / incident window")
        t1, t2 = st.columns(2)
        t1.write(f"**First detected:** {first_detected_time}")
        t2.write(f"**Last detected:** {last_detected_time}")
        st.caption(timestamp_note)
        if incidents_per_hour is not None:
            st.write(f"**Estimated detection episodes per dataset hour:** {incidents_per_hour:.2f}")
        else:
            st.write("**Incident rate:** episode count is available; an hourly rate cannot be reliably calculated from these timestamps.")

        st.info(
            f"Incident rate here means {attacks:,} flagged records out of {total:,} "
            f"records ({attack_rate:.2f}%). It is not a rate of independently confirmed attacks. "
            "Consecutive flagged rows are grouped as one detection episode."
        )

        # ---------------- SOURCE AND CAUSE ----------------
        st.markdown("### 2. Source, affected area, and likely cause")
        st.warning(
            "**Source attribution is not available from this CSV model alone.** "
            "The MLP does not identify an attacker IP, device, account, or person. "
            "The items below are investigation leads, not confirmed causes."
        )

        st.markdown("**Initial source assessment**")
        st.write(
            "- **Observed source of evidence:** the uploaded SCADA/process telemetry records.\n"
            "- **Possible area to investigate:** sensor/telemetry integrity, controller/process behaviour, "
            "or SCADA communications.\n"
            "- **How to confirm:** correlate the dataset time window with authorised firewall, "
            "gateway, historian, HMI, controller, identity, and maintenance logs."
        )

        st.markdown("**Most unusual model input features (triage clues)**")
        if top_features:
            feature_table = pd.DataFrame(
                top_features,
                columns=["Feature", "Mean absolute scaled deviation on flagged rows"]
            )
            st.dataframe(feature_table, use_container_width=True, hide_index=True)
            st.caption(
                "A larger scaled deviation indicates a feature worth reviewing. "
                "It does not prove that the feature caused the alert or was manipulated."
            )
        else:
            st.write("Feature deviation ranking could not be calculated for this upload.")

        st.markdown("**Possible causes to investigate**")
        for cause in [
            "Malicious manipulation or replay of sensor/telemetry values.",
            "Sensor fault, calibration drift, missing values, or maintenance activity.",
            "Abnormal operating conditions or an unusual process transition.",
            "Unauthorised configuration/command changes or a communications disruption."
        ]:
            st.markdown(f"- {cause}")

        # ---------------- OUTCOME / REMEDY ----------------
        st.markdown("### FDI hypothesis — checks to validate or reject it")
        for check in [
            "Compare flagged sensor values with independent sensors, redundant measurements, and engineering limits.",
            "Check whether related process variables changed consistently with turbine physics and operating state.",
            "Review historian/controller logs for unexplained value changes, setpoint changes, engineering access, and configuration edits.",
            "Check calibration, maintenance, missing-value handling, and communication faults as non-malicious explanations."
        ]:
            st.markdown(f"- {check}")
        st.caption("These checks investigate an FDI hypothesis; they do not prove that an FDI attack occurred.")

        st.markdown("### 3. Potential outcome and operational consequence")
        st.write(
            "Depending on what the investigation confirms, a real incident could affect:"
        )
        for consequence in [
            "**Monitoring and visibility:** inaccurate readings, missing alarms, or reduced operator confidence.",
            "**Control availability:** delayed or unavailable remote supervision or authorised control functions.",
            "**Energy production:** curtailed output or downtime during investigation and recovery.",
            "**Equipment and safety:** equipment stress or unsafe operating conditions if malicious activity affects control; this is not established by the ML alert.",
            "**Business operations:** response effort, inspection, reporting, regulatory/customer obligations, and reputational impact."
        ]:
            st.markdown(f"- {consequence}")

        st.markdown("### 4. Recommended company response")
        response_phases = [
            (
                "A — Triage and safety",
                [
                    "Notify the authorised turbine operator, OT security lead, and incident coordinator.",
                    "Check alarms and flagged readings against independent measurements and approved operating limits.",
                    "Preserve the original CSV, timestamps, model scores, relevant logs, and chain-of-custody details.",
                    "Treat the alert as a lead for investigation; do not make turbine control or shutdown decisions from the MLP score alone."
                ]
            ),
            (
                "B — Investigate and contain",
                [
                    "Identify the turbine, site, asset, and SCADA components involved using operational inventories and logs.",
                    "Correlate the event window with controller, gateway, historian, firewall, remote-access, and maintenance records.",
                    "Check for unexpected setpoint/configuration changes, communication failures, sensor faults, and access anomalies.",
                    "Have authorised OT personnel choose proportionate containment under the site's approved response plan; consider safety, availability, and fail-safe behaviour before isolating any link."
                ]
            ),
            (
                "C — Eradicate and recover",
                [
                    "Fix the confirmed root cause; rotate credentials or remove unauthorised access only through approved procedures where relevant.",
                    "Restore configurations/software from verified trusted sources and follow vendor/site change-control requirements.",
                    "Validate sensor integrity, controller behaviour, alarms, and safe operation before returning affected equipment to service.",
                    "Monitor for recurrence and document decisions, evidence, downtime, recovery actions, and lessons learned."
                ]
            )
        ]

        for phase_title, phase_steps in response_phases:
            st.markdown(f"**{phase_title}**")
            for step_number, step in enumerate(phase_steps, start=1):
                st.markdown(
                    f"<div class='playbook-step'>{step_number}. {step}</div>",
                    unsafe_allow_html=True
                )

        st.error(
            "Operational safety: do not automatically stop a turbine, change PLC/setpoints, "
            "disable protection, or isolate OT communications based solely on this ML alert. "
            "Follow the turbine manufacturer's guidance and the company's authorised OT incident plan."
        )

        # ---------------- LOSS ESTIMATE ----------------
        st.markdown("### 5. Potential resource loss — editable scenario estimate")
        st.caption(
            "Enter company-specific assumptions to estimate an illustrative exposure. "
            "These are not measured losses and should not be presented as actual damage."
        )

        loss_col1, loss_col2, loss_col3 = st.columns(3)
        with loss_col1:
            turbine_capacity_mw = st.number_input(
                "Affected capacity (MW)", min_value=0.0, value=2.0, step=0.5,
                help="Use the combined rated capacity of turbines believed to be affected."
            )
            outage_hours = st.number_input(
                "Assumed downtime (hours)", min_value=0.0, value=2.0, step=0.5
            )
        with loss_col2:
            capacity_factor = st.number_input(
                "Expected capacity factor (%)", min_value=0.0, max_value=100.0,
                value=35.0, step=5.0
            )
            electricity_price = st.number_input(
                "Electricity value (currency/kWh)", min_value=0.0, value=5.0, step=0.5
            )
        with loss_col3:
            response_cost = st.number_input(
                "Response cost (currency)", min_value=0.0, value=25000.0, step=5000.0
            )
            repair_cost = st.number_input(
                "Inspection / repair allowance (currency)", min_value=0.0,
                value=0.0, step=5000.0
            )

        estimated_lost_mwh = (
            turbine_capacity_mw * outage_hours * (capacity_factor / 100.0)
        )
        estimated_lost_revenue = estimated_lost_mwh * 1000.0 * electricity_price
        estimated_total_exposure = estimated_lost_revenue + response_cost + repair_cost

        l1, l2, l3 = st.columns(3)
        l1.metric("Illustrative energy not generated", f"{estimated_lost_mwh:,.2f} MWh")
        l2.metric("Illustrative lost energy value", f"{estimated_lost_revenue:,.2f}")
        l3.metric("Total scenario exposure", f"{estimated_total_exposure:,.2f}")

        st.caption(
            "Calculation: capacity (MW) × downtime (hours) × capacity factor = estimated MWh; "
            "MWh × 1,000 × energy value per kWh = estimated lost energy value. "
            "Total scenario exposure adds the entered response and repair allowances. "
            "It excludes unquantified safety, environmental, contractual, and reputational impacts. "
            "Replace the defaults with approved company figures before using this estimate."
        )

        # ---------------- DOWNLOADABLE PLAYBOOK ----------------
        st.markdown("### 6. Generate and export the incident playbook")
        # Build the playbook automatically for the current uploaded CSV.
        if uploaded_file is not None:
            st.session_state["playbook_generated"] = True
        if st.button("⚡ REFRESH COMPANY RESPONSE PLAYBOOK", use_container_width=True):
            st.session_state["playbook_generated"] = True

        if st.session_state.get("playbook_generated", False):
            top_feature_lines = (
                "\n".join(
                    f"- {name}: mean absolute scaled deviation {score:.3f}"
                    for name, score in top_features
                )
                if top_features else "- Not available"
            )

            playbook_text = f"""WT SENTINEL — WIND TURBINE SCADA INCIDENT RESPONSE PLAYBOOK
Case reference: WT-SENTINEL-CSV-REVIEW
Status: MODEL-GENERATED ALERT — INVESTIGATION REQUIRED
Working attack-type hypothesis: FALSE DATA INJECTION (FDI) — PROVISIONAL, UNVERIFIED
Important: the trained MLP predicts binary Normal-vs-Attack only. Its score is not a
probability that the attack is FDI. The HAI labels used here do not distinguish FDI/replay/spoofing.

1. DETECTION SUMMARY
Dataset records analysed: {total}
Flagged records: {attacks}
Flagged-record rate: {attack_rate:.2f}%
Consecutive detection episodes: {episodes}
Configured MLP threshold: {threshold:.4f}
Highest model score among all records: {max_probability:.4f}
Mean score on flagged records: {mean_probability:.4f}

2. TIME WINDOW
First detected dataset time/record: {first_detected_time}
Last detected dataset time/record: {last_detected_time}
{timestamp_note}
Episode rate per dataset hour: {f"{incidents_per_hour:.2f}" if incidents_per_hour is not None else "Not calculated: usable time span unavailable."}

3. SOURCE / ATTRIBUTION
The model flags SCADA/process telemetry records. The model does not identify an attacker,
IP address, account, device, or human identity. Correlate the time window with authorised
controller, historian, HMI, gateway, firewall, identity, remote-access, and maintenance logs.
Possible areas to investigate: sensor/telemetry integrity, controller/process behaviour,
SCADA communications, or benign sensor/maintenance/operating anomalies.

4. FEATURE TRIAGE CLUES (NOT PROOF OF CAUSE)
{top_feature_lines}

5. POSSIBLE OUTCOMES IF A REAL INCIDENT IS CONFIRMED
- Reduced trust in SCADA measurements or alarms.
- Loss of monitoring/remote supervision or control availability.
- Curtailment/downtime and reduced energy production.
- Potential equipment or safety consequences depending on affected process and safeguards.
- Investigation, recovery, contractual, reporting, and reputational costs.

5A. FDI HYPOTHESIS VALIDATION (NOT A CONFIRMED ATTACK TYPE)
- Compare flagged values with independent/redundant sensors and engineering limits.
- Check whether related process variables agree with expected physical relationships.
- Review controller, historian, setpoint, engineering-access, and configuration logs.
- Rule out sensor calibration faults, maintenance, missing values, and communication issues.

6. RESPONSE STEPS
TRIAGE:
- Notify authorised turbine operations, OT security, and incident coordination personnel.
- Verify flagged readings against independent measurements and approved operating limits.
- Preserve original data, timestamps, model outputs, and relevant logs.

INVESTIGATION / CONTAINMENT:
- Identify affected turbine/site/assets from company inventories and operational logs.
- Correlate controller, gateway, historian, firewall, remote-access, and maintenance records.
- Check sensor faults, communication failures, unexpected configuration/setpoint changes, and access anomalies.
- Choose containment only through authorised site procedures after considering safety and availability.

RECOVERY:
- Remediate the confirmed root cause using approved procedures.
- Restore trusted configurations/software where required and validate safe operation.
- Monitor for recurrence and record the timeline, decisions, downtime, and lessons learned.

7. ILLUSTRATIVE RESOURCE-LOSS SCENARIO (NOT ACTUAL LOSS)
Assumed affected capacity: {turbine_capacity_mw:.2f} MW
Assumed downtime: {outage_hours:.2f} hours
Assumed capacity factor: {capacity_factor:.1f}%
Assumed electricity value: {electricity_price:.2f} currency/kWh
Estimated energy not generated: {estimated_lost_mwh:.2f} MWh
Estimated lost energy value: {estimated_lost_revenue:.2f}
Entered response allowance: {response_cost:.2f}
Entered inspection/repair allowance: {repair_cost:.2f}
Illustrative total scenario exposure: {estimated_total_exposure:.2f}

These estimates depend entirely on the assumptions entered by the user. They are not
verified financial loss, equipment damage, or a prediction that a disaster will occur.
They exclude unquantified safety, environmental, contractual, and reputational effects.

8. SAFETY AND LIMITATIONS
This is decision support based on an offline CSV and a binary MLP. A flagged row is not
proof of a cyberattack, attack type, root cause, or physical impact. Do not automatically
stop a turbine, change PLC/setpoints, disable protection, or isolate OT communications
solely from this alert. Follow manufacturer guidance and the company's approved OT
incident-response and safety procedures.
"""

            st.text_area(
                "Generated playbook preview",
                value=playbook_text,
                height=420
            )
            pdf_bytes = build_playbook_pdf(playbook_text)
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    "📄 Download playbook as PDF",
                    data=pdf_bytes,
                    file_name="WT_Sentinel_Incident_Response_Playbook.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with download_col2:
                st.download_button(
                    "📝 Download playbook as TXT",
                    data=playbook_text,
                    file_name="WT_Sentinel_Incident_Response_Playbook.txt",
                    mime="text/plain",
                    use_container_width=True
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
