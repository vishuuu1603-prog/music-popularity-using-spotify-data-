import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# ----------------------------------
# Load trained artifacts
# ----------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("model_pipeline.pkl")

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
threshold = artifacts["threshold"]

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Predict", "ℹ️ About"]
)

# ----------------------------------
# HOME
# ----------------------------------
if page == "🏠 Home":
    st.title("🎧 Spotify Music Popularity Prediction")

    st.markdown("""
    This app predicts whether a song is **POPULAR or NOT**
    using a **Logistic Regression** model trained on Spotify audio features.

    ### Target Variable
    - `popular` → **1 = Popular**, **0 = Not Popular**

    ### Model Logic
    - Predicts probability
    - Uses threshold **0.40** (same as training)
    """)

# ----------------------------------
# PREDICTION PAGE
# ----------------------------------
elif page == "🎯 Predict":
    st.title("🎯 Predict Song Popularity")

    st.markdown("### 🎶 Enter Song Features")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)

    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)
        duration_ms = st.number_input("Duration (ms)", 30000, 600000, 210000)

    # Create input dataframe EXACTLY as training
    input_df = pd.DataFrame([[
        danceability,
        energy,
        loudness,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        duration_ms
    ]], columns=features)

    st.markdown("---")

    if st.button("🎯 Predict"):
        scaled_input = scaler.transform(input_df)

        # Predict probability (CORRECT)
        prob = model.predict_proba(scaled_input)[0][1]

        prediction = int(prob > threshold)

        st.subheader("📊 Result")

        st.write(f"**Popularity Probability:** `{prob:.2f}`")
        st.write(f"**Threshold Used:** `{threshold}`")

        if prediction == 1:
            st.success("🔥 POPULAR SONG")
        else:
            st.warning("⚠️ NOT POPULAR")

# ----------------------------------
# ABOUT
# ----------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About")

    st.markdown("""
    **Model**
    - Logistic Regression
    - Class-weight balanced
    - Threshold-based classification

    **Target**
    - `popular` (binary)

    **Built with**
    - Python
    - Scikit-learn
    - Streamlit

    **Author**
    - Vishva Patel
    """)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Spotify Popularity Prediction App")
