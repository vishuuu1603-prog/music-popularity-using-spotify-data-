import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# PAGE CONFIG + STYLE
# ===============================
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

st.markdown("""
<style>
body { background-color: white; }
.card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
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
# NAV BAR
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Use Cases", "Prediction", "About"]
)

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🎵 Spotify Song Popularity Prediction")

    st.write("""
    This system predicts whether a Spotify track is likely to become **popular**
    using machine learning trained on real Spotify audio data.

    The prediction is **probability‑based**, not rule‑based.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# USE CASES
# ===============================
elif page == "Use Cases":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📌 Use Cases")

    st.markdown("""
    - Artists optimizing songs before release  
    - Producers evaluating hit potential  
    - Music analytics & trend analysis  
    - Streaming platform insights  
    - Academic ML projects  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION
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
        tempo = st.number_input("Tempo", 40.0, 250.0, 40.0)

    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.0)
        valence = st.slider("Valence", 0.0, 1.0, 0.0)
        duration_ms = st.number_input("Duration (ms)", 30000, 600000, 30000)
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit", ["No", "Yes"])

    # 🔧 Threshold tester
    threshold = st.slider(
        "Decision Threshold",
        0.1, 0.9, float(default_threshold),
        help="Lower = more POPULAR predictions"
    )

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

        # Match training features
        for f in features:
            if f not in df.columns:
                df[f] = 0
        df = df[features]

        X_scaled = scaler.transform(df)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob > threshold)

        st.subheader("📊 Prediction Result")
        st.progress(min(int(prob * 100), 100))
        st.metric("Popularity Probability", f"{prob:.3f}")

        if pred == 1:
            st.success("🔥 POPULAR (probability above threshold)")
        else:
            st.warning("❄️ NOT POPULAR (probability below threshold)")

        st.caption(
            f"Decision Rule: probability > {threshold} → Popular"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")

    st.write("""
    This project demonstrates a **real‑world ML classification system**
    using Logistic Regression, feature scaling, and probability thresholds.

    Designed for **academic and professional analysis**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
