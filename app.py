import streamlit as st
import pandas as pd
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    layout="wide"
)

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
body {
    background-color: white;
}
.card {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}
h1, h2 {
    color: #1DB954;
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
# NAVIGATION
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
    A machine‑learning system that predicts whether a song is likely to become
    **popular** based on Spotify audio and metadata features.

    Predictions are **probability‑based**, not rule‑based.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# USE CASES
# ===============================
elif page == "Use Cases":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📌 Applications")
    st.markdown("""
    • Song hit‑potential analysis  
    • Artist release strategy  
    • Music analytics platforms  
    • Record label A&R decisions  
    • Academic ML demonstrations  
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
        danceability = st.slider("Danceability", 0.0, 1.0, 0.2)
        energy = st.slider("Energy", 0.0, 1.0, 0.2)
        loudness = st.slider("Loudness", -60.0, 0.0, -40.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.8)

    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.7)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.2)
        tempo = st.number_input("Tempo", 40.0, 250.0, 70.0)
        duration_ms = st.number_input("Duration (ms)", 30000, 600000, 180000)
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit", ["No", "Yes"])

    threshold = st.slider(
        "Decision Threshold",
        0.4, 0.8, 0.65,
        help="Higher threshold → harder to be POPULAR"
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

        for f in features:
            if f not in df.columns:
                df[f] = 0

        df = df[features]
        X = scaler.transform(df)

        prob = model.predict_proba(X)[0][1]
        pred = prob >= threshold

        st.subheader("📊 Result")
        st.metric("Popularity Probability", f"{prob:.3f}")

        if pred:
            st.success("🔥 POPULAR")
        else:
            st.warning("❄️ NOT POPULAR")

        st.caption(f"Rule: probability ≥ {threshold}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")
    st.write("""
    Built using Logistic Regression with feature scaling and
    probability‑based decision thresholds.

    Designed for **real‑world ML understanding**, not gimmicks.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
