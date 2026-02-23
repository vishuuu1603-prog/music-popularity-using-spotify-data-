import streamlit as st
import pandas as pd
import pickle

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Spotify Song Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# ======================================
# LOAD MODEL PIPELINE
# ======================================
@st.cache_resource
def load_pipeline():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

saved = load_pipeline()

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
threshold = saved["threshold"]

# ======================================
# SIDEBAR NAVIGATION
# ======================================
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Predict", "ℹ️ About"]
)

# ======================================
# HOME PAGE
# ======================================
if page == "🏠 Home":
    st.title("🎧 Spotify Song Popularity Prediction")

    st.markdown("""
    This app predicts whether a Spotify song will be **POPULAR (1)** or  
    **NOT POPULAR (0)** using a **Logistic Regression model**.

    ### 🔍 Model Details
    - Target: `popular`
    - Model: Logistic Regression (balanced)
    - Threshold: **0.40**
    - Scaled numeric features only

    👉 Go to **Predict** to test a song.
    """)

# ======================================
# PREDICTION PAGE
# ======================================
elif page == "🎯 Predict":
    st.title("🎯 Predict Song Popularity")

    st.markdown("### 🎵 Track & Album Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        track_number = st.number_input("Track Number", min_value=1, value=1)
        explicit = st.selectbox("Explicit", ["No", "Yes"])
        album_type = st.selectbox("Album Type", ["album", "single"])

    with col2:
        album_total_tracks = st.number_input(
            "Album Total Tracks", min_value=1, max_value=100, value=10
        )
        album_release_year = st.number_input(
            "Album Release Year", min_value=1950, max_value=2026, value=2020
        )

    with col3:
        duration_ms = st.number_input(
            "Track Duration (ms)", min_value=30000, max_value=600000, value=210000
        )

    st.markdown("### 🎶 Audio Features")

    col4, col5 = st.columns(2)

    with col4:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)

    with col5:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)

    # ======================================
    # FEATURE ENGINEERING (SAME AS TRAINING)
    # ======================================
    explicit_val = 1 if explicit == "Yes" else 0
    album_type_val = 1 if album_type == "single" else 0
    track_duration_min = duration_ms / 60000

    input_dict = {
        "track_number": track_number,
        "explicit": explicit_val,
        "album_type": album_type_val,
        "album_total_tracks": album_total_tracks,
        "album_release_year": album_release_year,
        "track_duration_min": track_duration_min,
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[features]

    st.markdown("---")

    # ======================================
    # PREDICTION
    # ======================================
    if st.button("🎯 Predict Popularity"):
        X_scaled = scaler.transform(input_df)
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(prob > threshold)

        st.subheader("📊 Prediction Result")
        st.write(f"**Popularity Probability:** `{prob:.2f}`")
        st.write(f"**Threshold Used:** `{threshold}`")

        if prediction == 1:
            st.success("🔥 POPULAR SONG")
        else:
            st.warning("⚠️ NOT POPULAR")

# ======================================
# ABOUT PAGE
# ======================================
elif page == "ℹ️ About":
    st.title("ℹ️ About")

    st.markdown("""
    **Spotify Song Success Prediction**

    - Logistic Regression (balanced)
    - Scaled numeric features
    - Threshold-based classification

    **Target:** `popular`  
    **Built with:** Python · Scikit-learn · Streamlit  
    """)

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("Spotify Popularity Prediction App")
