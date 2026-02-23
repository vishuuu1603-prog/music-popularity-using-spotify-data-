import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# PAGE CONFIG + ADVANCED 3D STYLE
# ===============================
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

st.markdown("""
<style>
body {
    background-color: #ffffff;
}
.card {
    background: linear-gradient(145deg, #ffffff, #f2f2f2);
    border-radius: 18px;
    padding: 28px;
    box-shadow:
        8px 8px 18px rgba(0,0,0,0.12),
       -8px -8px 18px rgba(255,255,255,0.9);
    margin-bottom: 30px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
with open("model_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
default_threshold = saved["threshold"]

# ===============================
# NAVIGATION BAR
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Use Cases", "Charts", "Prediction", "About"]
)

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎵 Spotify Song Popularity Prediction")

    st.write("""
    This application predicts whether a Spotify track is likely to become **popular**
    using a **probability‑based machine learning model** trained on real Spotify data.

    The system evaluates **audio features and album metadata** and produces
    interpretable, data‑driven predictions.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# USE CASES PAGE
# ===============================
elif page == "Use Cases":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📌 Use Cases")

    st.markdown("""
    - 🎤 Artists optimizing tracks before release  
    - 🎧 Producers evaluating hit potential  
    - 📊 Music analytics and trend analysis  
    - 🏢 Streaming platform insights  
    - 🎓 Academic and ML projects  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# CHARTS PAGE (NEW)
# ===============================
elif page == "Charts":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📊 Model Insights & Charts")

    st.subheader("1️⃣ Popularity Decision Threshold")
    st.progress(int(default_threshold * 100))
    st.caption(f"Default model threshold = {default_threshold}")

    st.subheader("2️⃣ Conceptual Feature Influence")
    feature_df = pd.DataFrame({
        "Feature Group": [
            "Energy & Loudness",
            "Danceability",
            "Tempo & Duration",
            "Acousticness",
            "Instrumentalness"
        ],
        "Relative Influence": [85, 75, 65, 40, 30]
    })
    st.bar_chart(feature_df.set_index("Feature Group"))

    st.subheader("3️⃣ Prediction Zones")
    zone_df = pd.DataFrame({
        "Probability": [0.0, default_threshold, 1.0],
        "Zone": ["Non‑Popular", "Decision Boundary", "Popular"]
    })
    st.line_chart(zone_df.set_index("Zone"))

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎯 Predict Song Popularity")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.0)
        energy = st.slider("Energy", 0.0, 1.0, 0.0)
        loudness = st.slider("Loudness", -60.0, 0.0, -60.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.0)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.0)
        tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 40.0)

    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.0)
        valence = st.slider("Valence", 0.0, 1.0, 0.0)
        duration_ms = st.number_input("Duration (ms)", 30000, 600000, 30000)
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    # 🔧 Threshold control (KEY FOR NON‑POPULAR)
    # Force conservative decision
    threshold = st.slider("Decision Threshold", 0.7, 0.95, 0.9)

# Debug proof
    st.write("Raw probability:", prob)

    if st.button("🚀 Predict"):
        input_dict = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_ms": duration_ms,
            "track_duration_min": duration_ms / 60000,
            "album_type": 1 if album_type == "single" else 0,
            "explicit": 1 if explicit == "Yes" else 0
        }

        df = pd.DataFrame([input_dict])

        for f in features:
            if f not in df.columns:
                df[f] = 0
        df = df[features]

        X_scaled = scaler.transform(df)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob > threshold)

        st.subheader("📊 Prediction Result")
        st.progress(int(prob * 100))
        st.metric("Popularity Probability", f"{prob:.3f}")

        if pred == 1:
            st.success("🔥 POPULAR")
        else:
            st.warning("❄️ NOT POPULAR")

        st.caption(f"Rule used: probability > {threshold}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")

    st.write("""
    This project demonstrates a **real‑world machine learning classification system**
    using Logistic Regression, feature scaling, and probability‑based decision making.

    The model is optimized for **imbalanced Spotify data** and prioritizes recall.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
